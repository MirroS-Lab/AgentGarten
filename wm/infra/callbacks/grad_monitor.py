# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

import torch

from wm.infra import io as easy_io
from wm.infra import log
from wm.infra.distributed import get_rank, is_rank0

from .base import Callback
from .grad_clip import _validate_max_norm, clip_grad_norm_, clip_grad_norm_groups_


def _small_metadata(value: Any, *, max_items: int = 64) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.numel() > max_items:
            return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        host = tensor.to(device="cpu")
        return host.item() if host.numel() == 1 else host.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _small_metadata(item, max_items=max_items)
            for key, item in islice(value.items(), max_items)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _small_metadata(item, max_items=max_items)
            for item in islice(value, max_items)
        ]
    return str(value)


def _active_optimizer_parameter_groups(
    model_ddp: torch.nn.Module,
    first_optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
) -> dict[str, tuple[torch.Tensor, ...]] | None:
    training_model = getattr(model_ddp, "training_model", None)
    if training_model is None:
        training_model = model_ddp
    get_names = getattr(training_model, "get_optimizer_names", None)
    optimizers = getattr(training_model, "optimizer_dict", None)
    if not callable(get_names) or optimizers is None:
        # A plain module without an optimizer selector is monitored root-wide.
        return None
    names = tuple(get_names(iteration))
    if any(name not in optimizers for name in names) or all(
        optimizers[name] is not first_optimizer for name in names
    ):
        return None
    return {
        str(name): tuple(
            parameter
            for parameter_group in optimizers[name].param_groups
            for parameter in parameter_group["params"]
        )
        for name in names
    }


class GradClipMonitor(Callback):
    def __init__(
        self,
        max_norm: float | None = 1.0,
        *,
        spike_threshold: float | None = None,
        spike_ratio: float | None = 5.0,
        warmup_updates: int = 10,
        ema_decay: float = 0.95,
        output_dir: str | None = None,
        sample_keys: Sequence[str] = (
            "data_key",
            "latent_path",
            "sample_index",
            "caption_start_frame",
            "window_start_frame",
        ),
    ) -> None:
        super().__init__()
        self.max_norm = None if max_norm is None else _validate_max_norm(max_norm)
        self.spike_threshold = (
            None if spike_threshold is None else float(spike_threshold)
        )
        self.spike_ratio = None if spike_ratio is None else float(spike_ratio)
        self.warmup_updates = max(0, int(warmup_updates))
        self.ema_decay = min(1.0, max(0.0, float(ema_decay)))
        self.output_dir = None if output_dir is None else str(output_dir)
        self.sample_keys = tuple(str(key) for key in sample_keys)

        self._sample_records: list[dict[str, Any]] = []
        self._measurement_done = False
        self._pending_norm: torch.Tensor | None = None
        self._pending_role_norms: dict[str, torch.Tensor] = {}
        self._pending_clip_coefficient: float | None = None
        self._pending_spike = False
        self._pending_optimizer_names: set[str] = set()
        self._norm_ema: float | None = None
        self._updates_seen = 0
        self._event_index = 0
        self._writer_enabled = True

    def on_training_step_start(
        self, model: Any, data: dict[str, Any], iteration: int = 0
    ) -> None:
        del model, iteration
        record = {
            key: _small_metadata(data[key]) for key in self.sample_keys if key in data
        }
        if record:
            self._sample_records.append(record)

    def _is_spike(self, norm: float) -> tuple[bool, float | None]:
        baseline = self._norm_ema
        finite = math.isfinite(norm)
        absolute = self.spike_threshold is not None and norm >= self.spike_threshold
        relative = (
            finite
            and self.spike_ratio is not None
            and self._updates_seen >= self.warmup_updates
            and baseline is not None
            and baseline > 0.0
            and norm >= baseline * self.spike_ratio
        )
        spike = not finite or absolute or relative

        if finite:
            self._norm_ema = (
                norm
                if baseline is None
                else self.ema_decay * baseline + (1.0 - self.ema_decay) * norm
            )
        self._updates_seen += 1
        return spike, baseline

    def _write_spike(
        self,
        *,
        iteration: int,
        norm: float,
        baseline: float | None,
    ) -> None:
        if self.output_dir is None or not self._writer_enabled:
            return

        rank = get_rank()
        self._event_index += 1
        filename = (
            f"event_{self._event_index:06d}_iter_{iteration:09d}_rank_{rank:04d}.json"
        )
        payload = {
            "iteration": iteration,
            "rank": rank,
            "grad_norm": norm,
            "grad_norm_ema_before_update": baseline,
            "spike_threshold": self.spike_threshold,
            "spike_ratio": self.spike_ratio,
            "microbatches": list(self._sample_records),
        }
        try:
            path = str(
                easy_io.join_path(
                    self.output_dir,
                    filename,
                )
            )
            easy_io.atomic_put_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                path,
            )
        except Exception as error:  # monitoring must not interrupt training
            self._writer_enabled = False
            log.warning(
                "Disabling gradient spike writer after failure: "
                f"{type(error).__name__}: {error}"
            )

    def on_before_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del scheduler, grad_scaler
        # DMD may expose more than one active optimizer at a boundary.  Their
        # gradients belong to one atomic update, so scan/clip the model once.
        if self._measurement_done:
            return
        self._measurement_done = True

        clip_limit = float("inf") if self.max_norm is None else self.max_norm
        parameter_groups = _active_optimizer_parameter_groups(
            model_ddp,
            optimizer,
            iteration=iteration,
        )
        if parameter_groups is None:
            total_norm = clip_grad_norm_(model_ddp.parameters(), clip_limit)
            role_norms: dict[str, torch.Tensor] = {}
        else:
            total_norm, role_norms = clip_grad_norm_groups_(
                parameter_groups,
                clip_limit,
            )
        if total_norm is None:
            return

        self._pending_norm = total_norm.detach()
        self._pending_role_norms = {
            role: norm.detach() for role, norm in role_norms.items()
        }
        norm = float(total_norm.detach().float().cpu())
        if math.isinf(clip_limit):
            self._pending_clip_coefficient = 1.0
        elif math.isfinite(norm):
            self._pending_clip_coefficient = min(
                1.0,
                self.max_norm / (norm + 1.0e-6),
            )
        else:
            self._pending_clip_coefficient = 0.0

        spike, baseline = self._is_spike(norm)
        self._pending_spike = spike
        if spike:
            logical_iteration = iteration + 1
            self._write_spike(
                iteration=logical_iteration,
                norm=norm,
                baseline=baseline,
            )
            if is_rank0():
                log.warning(
                    f"Gradient spike at iteration {logical_iteration}: "
                    f"norm={norm:.6g}, previous_ema={baseline}"
                )

    def on_after_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        del model_ddp, optimizer, scheduler, iteration
        self._pending_optimizer_names.add(str(name))

    def on_optimizer_step_skipped(
        self,
        model_ddp: torch.nn.Module,
        optimizer_names: tuple[str, ...],
        iteration: int = 0,
    ) -> None:
        del model_ddp, iteration
        self._pending_optimizer_names.update(optimizer_names)

    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, loss, iteration
        if not self._measurement_done:
            return

        if self._pending_norm is not None:
            reference = self._pending_norm
            output_batch["metrics/grad_norm"] = reference
            if self._pending_role_norms:
                for role, norm in self._pending_role_norms.items():
                    output_batch[f"metrics/{role}/grad_norm"] = norm
                    output_batch[f"metrics/{role}/grad_clip_coefficient"] = (
                        reference.new_tensor(self._pending_clip_coefficient)
                    )
            elif len(self._pending_optimizer_names) == 1:
                optimizer_name = next(iter(self._pending_optimizer_names))
                if optimizer_name != "net":
                    output_batch[f"metrics/{optimizer_name}/grad_norm"] = reference
            output_batch["metrics/grad_clip_coefficient"] = reference.new_tensor(
                self._pending_clip_coefficient
            )
            output_batch["metrics/grad_spike"] = reference.new_tensor(
                float(self._pending_spike)
            )

        self._sample_records.clear()
        self._measurement_done = False
        self._pending_norm = None
        self._pending_role_norms.clear()
        self._pending_clip_coefficient = None
        self._pending_spike = False
        self._pending_optimizer_names.clear()


__all__ = ["GradClipMonitor"]
