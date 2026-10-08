# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any

import torch
from torch.distributed.tensor import DTensor

_LARGE_FP32_STATE_KEYS = frozenset(("exp_avg", "exp_avg_sq", "master_param"))


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _scalar(value: float | int | torch.Tensor) -> float:
    return (
        float(value.detach().item())
        if isinstance(value, torch.Tensor)
        else float(value)
    )


def _transformer_engine() -> tuple[Any, Any]:
    # Import the framework package before its extension: newer releases register
    # ``transformer_engine_torch`` while importing ``transformer_engine.pytorch``.
    try:
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
    except ImportError as error:
        raise ModuleNotFoundError(
            "FusedAdam with beta1 > 0 and beta2 > 0 on CUDA runs Transformer "
            "Engine's multi-tensor Adam; install it with pip install 'wm[te]' "
            "or use a container that ships it"
        ) from error
    return te, tex


def _same_fp32_state_layout(
    value: torch.Tensor,
    parameter: torch.Tensor,
) -> bool:
    if value.dtype != torch.float32 or value.device != parameter.device:
        return False
    if isinstance(value, DTensor) != isinstance(parameter, DTensor):
        return False
    if isinstance(value, DTensor) and (
        value.device_mesh != parameter.device_mesh
        or value.placements != parameter.placements
    ):
        return False
    value_local = _local_tensor(value)
    parameter_local = _local_tensor(parameter)
    return (
        value.shape == parameter.shape
        and value_local.shape == parameter_local.shape
        and value_local.layout == parameter_local.layout
    )


def _restore_fp32_state_tensor(
    value: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    if _same_fp32_state_layout(value, parameter):
        return value
    return value.detach().to(device=parameter.device, dtype=torch.float32)


class FusedAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float | torch.Tensor = 1e-3,
        *,
        bias_correction: bool = True,
        betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999),
        eps: float | torch.Tensor = 1e-10,
        adam_w_mode: bool = True,
        weight_decay: float | torch.Tensor = 0.0,
        amsgrad: bool = False,
        use_te_capturable_kernel: bool = False,
        master_weights: bool = False,
    ) -> None:
        if amsgrad:
            raise ValueError("FusedAdam does not implement AMSGrad")
        if master_weights and not use_te_capturable_kernel:
            raise ValueError(
                "FP32 master weights require use_te_capturable_kernel=True"
            )

        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        self._validate_group(defaults)
        super().__init__(params, defaults)
        self.adam_w_mode = bool(adam_w_mode)
        self.use_te_capturable_kernel = bool(use_te_capturable_kernel)
        self.master_weights = bool(master_weights)
        self._overflow_buffers: dict[torch.device, torch.Tensor] = {}
        # Fail at construction, not at the first step, when the backend is missing.
        if any(
            self._uses_first_moment(_scalar(group["betas"][0]))
            and self._uses_second_moment(_scalar(group["betas"][1]))
            and any(parameter.is_cuda for parameter in group["params"])
            for group in self.param_groups
        ):
            _transformer_engine()

    @staticmethod
    def _validate_group(group: dict[str, Any]) -> None:
        def scalar(name: str, value: Any) -> float:
            if torch.is_tensor(value) and value.numel() != 1:
                raise ValueError(f"FusedAdam {name} must be a scalar")
            try:
                result = _scalar(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"FusedAdam {name} must be a finite scalar") from error
            if not math.isfinite(result):
                raise ValueError(f"FusedAdam {name} must be finite")
            return result

        for name in ("lr", "eps", "weight_decay"):
            if scalar(name, group[name]) < 0:
                raise ValueError(f"FusedAdam {name} must be non-negative")
        betas = group["betas"]
        if (
            not isinstance(betas, Sequence)
            or isinstance(betas, (str, bytes))
            or len(betas) != 2
        ):
            raise ValueError("FusedAdam betas must contain two scalars")
        for index, beta in enumerate(betas):
            if not 0.0 <= scalar(f"beta{index + 1}", beta) < 1.0:
                raise ValueError(f"FusedAdam beta{index + 1} must be in [0, 1)")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self._validate_group({**self.defaults, **param_group})
        super().add_param_group(param_group)

    @staticmethod
    def _uses_first_moment(beta1: float) -> bool:
        return beta1 != 0.0

    @staticmethod
    def _uses_second_moment(beta2: float) -> bool:
        return beta2 != 0.0

    @classmethod
    def _initialise_state(
        cls,
        parameter: torch.Tensor,
        state: dict[str, Any],
        *,
        beta1: float,
        beta2: float,
        master_weights: bool,
    ) -> None:
        if "step" not in state:
            # Step is control-plane state, not parameter-shaped numerical
            # state.  Keeping one scalar per parameter on CUDA would make the
            # Python bucket assembly below call ``.item()`` across the device
            # boundary once per parameter on every update.  A CPU scalar
            # preserves intermittent-gradient semantics and remains a plain
            # replicated scalar in DCP checkpoints.
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
        if cls._uses_first_moment(beta1):
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            # Old checkpoints may contain the formerly unconditional slot.
            state.pop("exp_avg", None)
        if cls._uses_second_moment(beta2):
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            # Also clean up checkpoints produced before beta2=0 was supported.
            state.pop("exp_avg_sq", None)
        if master_weights and "master_param" not in state:
            state["master_param"] = parameter.detach().clone().to(torch.float32)

    @staticmethod
    def _foreach_step(
        *,
        gradients: list[torch.Tensor],
        parameters: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        master_parameters: list[torch.Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        use_first_moment: bool,
        use_second_moment: bool,
        eps: float,
        weight_decay: float,
        adam_w_mode: bool,
        bias_correction: bool,
        step: int,
    ) -> None:
        if not parameters:
            return

        work_gradients = [gradient.float() for gradient in gradients]
        update_parameters = master_parameters or parameters
        if not adam_w_mode and weight_decay != 0.0:
            work_gradients = torch._foreach_add(
                work_gradients, update_parameters, alpha=weight_decay
            )

        if use_first_moment:
            torch._foreach_lerp_(exp_avgs, work_gradients, 1.0 - beta1)
            numerators = exp_avgs
        else:
            # m_t = g_t and the first-moment bias correction is exactly one.
            numerators = work_gradients
        if use_second_moment:
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(
                exp_avg_sqs, work_gradients, work_gradients, value=1.0 - beta2
            )
            denominators = torch._foreach_sqrt(exp_avg_sqs)
        else:
            # For beta2=0, v_t = g_t**2 and its bias correction is one.  The
            # absolute gradient is a temporary value, not checkpoint state.
            denominators = [gradient.abs() for gradient in work_gradients]

        if bias_correction:
            correction1 = 1.0 - beta1**step
            correction2_sqrt = (1.0 - beta2**step) ** 0.5 if use_second_moment else 1.0
        else:
            correction1 = 1.0
            correction2_sqrt = 1.0

        torch._foreach_div_(denominators, correction2_sqrt)
        torch._foreach_add_(denominators, eps)
        if adam_w_mode and weight_decay != 0.0:
            torch._foreach_mul_(update_parameters, 1.0 - lr * weight_decay)
        torch._foreach_addcdiv_(
            update_parameters,
            numerators,
            denominators,
            value=-lr / correction1,
        )
        if master_parameters:
            torch._foreach_copy_(parameters, master_parameters)

    def _cuda_fused_step(
        self,
        *,
        gradients: list[torch.Tensor],
        parameters: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        master_parameters: list[torch.Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        use_first_moment: bool,
        use_second_moment: bool,
        eps: float,
        weight_decay: float,
        bias_correction: bool,
        step: int,
        device: torch.device,
    ) -> None:
        if not parameters:
            return
        if not use_first_moment or not use_second_moment:
            raise ValueError("Transformer Engine path requires both Adam moments")

        te, tex = _transformer_engine()

        overflow = self._overflow_buffers.get(device)
        if overflow is None:
            overflow = torch.zeros(1, dtype=torch.int32, device=device)
            self._overflow_buffers[device] = overflow

        if self.use_te_capturable_kernel:
            kernel = (
                tex.multi_tensor_adam_capturable_master
                if self.master_weights
                else tex.multi_tensor_adam_capturable
            )
            tensor_lists = [gradients, parameters, exp_avgs, exp_avg_sqs]
            if self.master_weights:
                tensor_lists.append(master_parameters)
            te.optimizers.multi_tensor_applier(
                kernel,
                overflow,
                tensor_lists,
                torch.tensor(lr, dtype=torch.float32, device=device),
                beta1,
                beta2,
                eps,
                torch.tensor(step, dtype=torch.int32, device=device),
                int(self.adam_w_mode),
                int(bias_correction),
                weight_decay,
                torch.ones(1, dtype=torch.float32, device=device),
            )
            return

        te.optimizers.multi_tensor_applier(
            tex.multi_tensor_adam,
            overflow,
            [gradients, parameters, exp_avgs, exp_avg_sqs],
            lr,
            beta1,
            beta2,
            eps,
            step,
            int(self.adam_w_mode),
            int(bias_correction),
            weight_decay,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Schedulers and callers may edit groups after construction. Validate
        # every group first so an invalid later group cannot partially update.
        for group in self.param_groups:
            self._validate_group(group)
        for group in self.param_groups:
            # Control scalars belong to the group, not individual parameters
            # or step buckets. In particular, float(CUDA_tensor) synchronizes.
            beta1, beta2 = (float(beta) for beta in group["betas"])
            use_first_moment = self._uses_first_moment(beta1)
            use_second_moment = self._uses_second_moment(beta2)
            lr = _scalar(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            bias_correction = bool(group["bias_correction"])
            buckets: dict[
                tuple[torch.device, torch.dtype, int],
                tuple[
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                    list[torch.Tensor],
                ],
            ] = {}

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients")
                if parameter.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                ):
                    raise TypeError(
                        "FusedAdam supports float16, bfloat16, and float32 parameters"
                    )

                state = self.state[parameter]
                self._initialise_state(
                    parameter,
                    state,
                    beta1=beta1,
                    beta2=beta2,
                    master_weights=self.master_weights,
                )
                step = int(_scalar(state["step"])) + 1
                state["step"].fill_(step)
                local_parameter = _local_tensor(parameter)
                local_gradient = _local_tensor(gradient)
                if local_parameter.numel() == 0 or local_gradient.numel() == 0:
                    continue

                # Transformer Engine's multi-tensor Adam kernel accepts one
                # scalar step per launch. Parameters can legitimately have
                # different steps when gradients are intermittent, so retain
                # fusion by grouping only parameters with the same step.
                key = (local_parameter.device, parameter.dtype, step)
                bucket = buckets.setdefault(key, ([], [], [], [], []))
                bucket[0].append(local_gradient)
                bucket[1].append(local_parameter)
                if use_first_moment:
                    bucket[2].append(_local_tensor(state["exp_avg"]))
                if use_second_moment:
                    bucket[3].append(_local_tensor(state["exp_avg_sq"]))
                if self.master_weights:
                    bucket[4].append(_local_tensor(state["master_param"]))

            for (device, _dtype, step), bucket in buckets.items():
                common = {
                    "gradients": bucket[0],
                    "parameters": bucket[1],
                    "exp_avgs": bucket[2],
                    "exp_avg_sqs": bucket[3],
                    "master_parameters": bucket[4],
                    "lr": lr,
                    "beta1": beta1,
                    "beta2": beta2,
                    "use_first_moment": use_first_moment,
                    "use_second_moment": use_second_moment,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "bias_correction": bias_correction,
                    "step": step,
                }
                if device.type == "cuda" and use_first_moment and use_second_moment:
                    self._cuda_fused_step(device=device, **common)
                else:
                    self._foreach_step(
                        adam_w_mode=self.adam_w_mode,
                        **common,
                    )

        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        stashed: dict[Any, dict[str, torch.Tensor]] = {}
        saved_param_groups: tuple[tuple[Any, ...], ...] = ()

        def extract_large_state(
            _optimizer: torch.optim.Optimizer,
            public_state_dict: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal saved_param_groups

            groups = public_state_dict.get("param_groups", ())
            for group in groups:
                self._validate_group(group)
            saved_param_groups = tuple(
                tuple(group.get("params", ())) for group in groups
            )
            parameter_ids = set(chain.from_iterable(saved_param_groups))
            saved_state = public_state_dict.get("state")
            if not isinstance(saved_state, dict):
                return public_state_dict

            stripped_state = saved_state.copy()
            changed = False
            for old_id in parameter_ids:
                parameter_state = saved_state.get(old_id)
                if not isinstance(parameter_state, dict):
                    continue
                large_state = {
                    key: value
                    for key, value in parameter_state.items()
                    if key in _LARGE_FP32_STATE_KEYS and isinstance(value, torch.Tensor)
                }
                if not large_state:
                    continue
                stashed[old_id] = large_state
                stripped_parameter_state = parameter_state.copy()
                for key in large_state:
                    stripped_parameter_state.pop(key)
                stripped_state[old_id] = stripped_parameter_state
                changed = True

            if not changed:
                return public_state_dict
            stripped = public_state_dict.copy()
            stripped["state"] = stripped_state
            return stripped

        def restore_large_state(optimizer: torch.optim.Optimizer) -> None:
            current_parameters = tuple(
                tuple(group["params"]) for group in optimizer.param_groups
            )
            id_map = dict(
                zip(
                    chain.from_iterable(saved_param_groups),
                    chain.from_iterable(current_parameters),
                    strict=True,
                )
            )
            parameter_betas = {
                parameter: tuple(float(beta) for beta in group["betas"])
                for group in optimizer.param_groups
                for parameter in group["params"]
            }

            for old_id, values in stashed.items():
                parameter = id_map.get(old_id)
                if parameter is None:
                    continue
                beta1, beta2 = parameter_betas[parameter]
                parameter_state = optimizer.state[parameter]
                for key, value in values.items():
                    if key == "exp_avg" and not self._uses_first_moment(beta1):
                        continue
                    if key == "exp_avg_sq" and not self._uses_second_moment(beta2):
                        continue
                    parameter_state[key] = _restore_fp32_state_tensor(
                        value,
                        parameter,
                    )

            # Preserve the existing compatibility behavior for legacy
            # checkpoints whose zero-beta groups still contain a moment slot.
            for group in optimizer.param_groups:
                beta1, beta2 = (float(beta) for beta in group["betas"])
                for parameter in group["params"]:
                    if parameter not in optimizer.state:
                        continue
                    if not self._uses_first_moment(beta1):
                        optimizer.state[parameter].pop("exp_avg", None)
                    if not self._uses_second_moment(beta2):
                        optimizer.state[parameter].pop("exp_avg_sq", None)

        pre_handle = self.register_load_state_dict_pre_hook(extract_large_state)
        post_handle = self.register_load_state_dict_post_hook(
            restore_large_state,
            prepend=True,
        )
        try:
            super().load_state_dict(state_dict)
        finally:
            pre_handle.remove()
            post_handle.remove()


__all__ = ["FusedAdam"]
