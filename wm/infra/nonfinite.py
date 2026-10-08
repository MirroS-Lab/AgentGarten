# SPDX-License-Identifier: Apache-2.0

"""Agree across ranks on whether an update saw a non-finite gradient."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

__all__ = ["synchronize_found_nonfinite"]


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def _scan_nonfinite_gradients(
    optimizers: tuple[torch.optim.Optimizer, ...],
) -> list[torch.Tensor]:
    grouped: dict[torch.device, dict[torch.dtype, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[int] = set()
    for optimizer in optimizers:
        for parameter_group in optimizer.param_groups:
            for parameter in parameter_group["params"]:
                identity = id(parameter)
                if identity in seen:
                    continue
                seen.add(identity)
                gradient = parameter.grad
                if gradient is None:
                    continue
                gradient = _local_tensor(gradient)
                if gradient.is_sparse:
                    gradient = gradient.coalesce().values()
                # FSDP/HSDP may legitimately assign an empty local shard for
                # a small parameter (notably the discriminator head).  Empty
                # gradients contain no non-finite values, and CUDA's BF16
                # fallback ``foreach_norm(inf)`` has no identity for them.
                if gradient.numel() == 0:
                    continue
                grouped[gradient.device][gradient.dtype].append(gradient)

    found_per_device: list[torch.Tensor] = []
    for device, gradients_by_dtype in grouped.items():
        found = torch.zeros((), dtype=torch.float32, device=device)
        inverse_scale = torch.ones((), dtype=torch.float32, device=device)
        for gradients in gradients_by_dtype.values():
            try:
                torch._amp_foreach_non_finite_check_and_unscale_(
                    gradients, found, inverse_scale
                )
            except RuntimeError:
                # Some older CUDA builds do not implement the AMP foreach
                # finite check for BF16. Keep that compatibility path entirely
                # device-side: one predicate kernel per gradient followed by a
                # single device reduction, rather than synchronizing Python
                # once for every parameter.
                infinity_norms = torch._foreach_norm(gradients, float("inf"))
                finite = torch.isfinite(torch.stack(infinity_norms)).all()
                found.copy_(
                    torch.maximum(found, torch.logical_not(finite).to(found.dtype))
                )
        found_per_device.append(found)
    return found_per_device


def synchronize_found_nonfinite(
    grad_scaler: torch.amp.GradScaler,
    optimizers: tuple[torch.optim.Optimizer, ...],
    *,
    fallback_device: torch.device,
) -> bool:
    found_inf_per_optimizer: list[dict[torch.device, torch.Tensor]] = []
    if grad_scaler.is_enabled():
        # Compatibility shim for the only GradScaler state without a public
        # accessor. Every optimizer has already passed through unscale_ here.
        found_inf_per_optimizer = [
            grad_scaler._found_inf_per_device(optimizer) for optimizer in optimizers
        ]
        found_inf_tensors = [
            found_inf
            for per_device in found_inf_per_optimizer
            for found_inf in per_device.values()
        ]
    else:
        found_inf_tensors = []
    found_inf_tensors.extend(_scan_nonfinite_gradients(optimizers))

    if found_inf_tensors:
        consensus = found_inf_tensors[0].detach().clone()
        for found_inf in found_inf_tensors[1:]:
            consensus.copy_(
                torch.maximum(consensus, found_inf.to(device=consensus.device))
            )
    else:
        consensus = torch.zeros((), dtype=torch.float32, device=fallback_device)

    consensus.clamp_(min=0, max=1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(consensus, op=dist.ReduceOp.MAX)
        consensus.clamp_(min=0, max=1)

    if grad_scaler.is_enabled():
        # Propagate the collective decision back into every optimizer state.
        # This makes scaler.step and the single scaler.update agree on all ranks.
        for per_device in found_inf_per_optimizer:
            if per_device:
                for device, found_inf in per_device.items():
                    found_inf.copy_(consensus.to(device=device))
            else:
                per_device[consensus.device] = consensus.clone()

    return bool(consensus.item())
