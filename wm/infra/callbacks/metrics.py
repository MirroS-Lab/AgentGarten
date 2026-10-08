# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch
import torch.distributed as dist

from wm.infra import log
from wm.infra.distributed import is_rank0

from .base import Callback

_METRIC_PREFIX = "metrics/"
_TRAIN_LOSS_KEY = "train/loss"
_VALIDATION_LOSS_KEY = "validation/loss"


class MetricSink(Protocol):
    def write(self, metrics: Mapping[str, float], *, iteration: int) -> None: ...


class LoguruMetricSink:
    def write(self, metrics: Mapping[str, float], *, iteration: int) -> None:
        values = " | ".join(
            f"{name}={value:.6g}" for name, value in sorted(metrics.items())
        )
        log.info(f"iteration={iteration} | {values}")


class ScalarMetricCallback(Callback):
    def __init__(
        self,
        every_n: int | None = 100,
        sinks: Sequence[MetricSink | None] | None = None,
        *,
        report_update_speed: bool = False,
    ) -> None:
        super().__init__()
        self.every_n = None if every_n is None else int(every_n)
        self.sinks = (
            tuple(sink for sink in sinks if sink is not None)
            if sinks is not None
            else (LoguruMetricSink(),)
        )
        self._train: dict[str, torch.Tensor] = {}
        self._train_gauges: dict[str, torch.Tensor] = {}
        self._validation: dict[str, torch.Tensor] = {}
        self._train_ignored: set[str] = set()
        self._validation_ignored: set[str] = set()
        self._warned_ignored: set[str] = set()
        self._disabled_sinks: set[int] = set()
        self.report_update_speed = bool(report_update_speed)
        self._update_window_started_at: float | None = None
        self._update_window_size = 0
        self._pending_learning_rates: dict[int, tuple[torch.Tensor, ...]] = {}

    def on_before_dataloading(self, iteration: int = 0) -> None:
        del iteration
        if self.report_update_speed and self._update_window_started_at is None:
            self._update_window_started_at = time.perf_counter()

    def _take_update_speed_metrics(self) -> dict[str, float]:
        if self._update_window_started_at is None or self._update_window_size == 0:
            return {}
        elapsed = time.perf_counter() - self._update_window_started_at
        update_time = elapsed / self._update_window_size
        self._update_window_started_at = None
        self._update_window_size = 0
        return {
            "performance/update_time_s": update_time,
            "performance/updates_per_s": 1.0 / update_time,
        }

    @staticmethod
    def _scalar_tensor(value: Any, *, reference: torch.Tensor) -> torch.Tensor | None:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return value.detach().reshape(()).float()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.tensor(
                float(value), dtype=torch.float32, device=reference.device
            )
        return None

    @staticmethod
    def _add(
        accumulator: dict[str, torch.Tensor],
        name: str,
        value: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
    ) -> None:
        finite = torch.isfinite(value)
        safe_value = torch.where(finite, value, torch.zeros_like(value))
        if weight is None:
            # Training observations have unit weight. Avoid creating and
            # processing an all-ones tensor for every scalar on every batch.
            total = safe_value
            count = finite.to(dtype=torch.float32)
            invalid_count = (~finite).to(dtype=torch.float32)
        else:
            weight = weight.to(device=value.device, dtype=torch.float32)
            weight = torch.where(
                torch.isfinite(weight) & (weight >= 0),
                weight,
                torch.zeros_like(weight),
            )
            total = safe_value * weight
            count = finite.to(dtype=torch.float32) * weight
            invalid_count = ((~finite) & (weight > 0)).to(dtype=torch.float32)
        record = torch.stack((total, count, invalid_count))
        current = accumulator.get(name)
        if current is None:
            accumulator[name] = record
        else:
            if current.device != record.device:
                current = current.to(record.device)
                accumulator[name] = current
            current.add_(record)

    def _accumulate(
        self,
        accumulator: dict[str, torch.Tensor],
        ignored: set[str],
        output_batch: Mapping[str, Any],
        loss: torch.Tensor,
        *,
        split: str,
        weight: torch.Tensor | None = None,
    ) -> None:
        loss_name = _TRAIN_LOSS_KEY if split == "train" else _VALIDATION_LOSS_KEY
        loss_value = self._scalar_tensor(loss, reference=loss)
        if loss_value is None:
            ignored.add(f"{loss_name} (non-scalar loss)")
        else:
            self._add(accumulator, loss_name, loss_value, weight=weight)

        for source_name, raw_value in output_batch.items():
            if not isinstance(source_name, str) or not source_name.startswith(
                _METRIC_PREFIX
            ):
                continue
            suffix = source_name.removeprefix(_METRIC_PREFIX)
            if not suffix:
                ignored.add("metrics/ (empty metric name)")
                continue
            name = f"{split}/{suffix}"
            if name == loss_name:
                ignored.add(
                    f"{source_name} (reserved; the callback records the provided loss)"
                )
                continue
            value = self._scalar_tensor(raw_value, reference=loss)
            if value is None:
                ignored.add(f"{source_name} (expected one scalar tensor or number)")
                continue
            self._add(accumulator, name, value, weight=weight)

    def on_before_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model_ddp, scheduler, grad_scaler, iteration
        self._pending_learning_rates[id(optimizer)] = tuple(
            (
                value.detach().reshape(()).float().clone()
                if torch.is_tensor(value)
                else torch.tensor(float(value), dtype=torch.float32)
            )
            for value in (group["lr"] for group in optimizer.param_groups)
        )

    def on_after_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        del model_ddp, scheduler, iteration
        learning_rates = self._pending_learning_rates.pop(id(optimizer), ())
        prefix = "" if name == "net" else f"{name}/"
        for group_index, learning_rate in enumerate(learning_rates):
            self._train_gauges[f"train/{prefix}scheduler/lr_{group_index}"] = (
                learning_rate
            )

    def on_optimizer_step_skipped(
        self,
        model_ddp: torch.nn.Module,
        optimizer_names: tuple[str, ...],
        iteration: int = 0,
    ) -> None:
        del model_ddp, optimizer_names, iteration
        self._pending_learning_rates.clear()

    @staticmethod
    def _reduction_device(
        accumulator: Mapping[str, torch.Tensor],
    ) -> torch.device:
        if dist.is_available() and dist.is_initialized():
            if dist.get_backend() == "nccl":
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device("cpu")
        if accumulator:
            return next(iter(accumulator.values())).device
        return torch.device("cpu")

    @staticmethod
    def _global_metadata(
        local_names: Sequence[str], local_ignored: set[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not (dist.is_available() and dist.is_initialized()):
            return tuple(sorted(local_names)), tuple(sorted(local_ignored))
        gathered: list[tuple[list[str], list[str]] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            (list(local_names), sorted(local_ignored)),
        )
        names = {
            name
            for rank_metadata in gathered
            for name in (() if rank_metadata is None else rank_metadata[0])
        }
        ignored = {
            message
            for rank_metadata in gathered
            for message in (() if rank_metadata is None else rank_metadata[1])
        }
        return tuple(sorted(names)), tuple(sorted(ignored))

    def _write_to_sinks(self, metrics: Mapping[str, float], *, iteration: int) -> None:
        for sink in self.sinks:
            identity = id(sink)
            if identity in self._disabled_sinks:
                continue
            try:
                sink.write(metrics, iteration=iteration)
            except Exception as error:  # logging must not discard training progress
                self._disabled_sinks.add(identity)
                log.warning(
                    f"Disabling metric sink {type(sink).__name__} after write failure: "
                    f"{type(error).__name__}: {error}"
                )

    def _flush(
        self,
        accumulator: dict[str, torch.Tensor],
        ignored: set[str],
        *,
        iteration: int,
        gauges: dict[str, torch.Tensor] | None = None,
        extra_metrics: Mapping[str, float] | None = None,
    ) -> None:
        gauges = {} if gauges is None else gauges
        names, ignored_names = self._global_metadata(
            sorted(set(accumulator) | set(gauges)), ignored
        )
        device = self._reduction_device(accumulator)
        packed = torch.zeros((len(names), 3), dtype=torch.float32, device=device)
        for index, name in enumerate(names):
            local = accumulator.get(name)
            if local is not None:
                packed[index].copy_(local.to(device=device))
                continue
            gauge = gauges.get(name)
            if gauge is not None:
                gauge = gauge.to(device=device, dtype=torch.float32)
                finite = torch.isfinite(gauge)
                packed[index, 0] = torch.where(finite, gauge, torch.zeros_like(gauge))
                packed[index, 1] = finite.to(dtype=torch.float32)
                packed[index, 2] = (~finite).to(dtype=torch.float32)

        if names and dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        accumulator.clear()
        gauges.clear()
        ignored.clear()
        if not is_rank0():
            return

        new_ignored = set(ignored_names).difference(self._warned_ignored)
        self._warned_ignored.update(new_ignored)
        for message in sorted(new_ignored):
            log.warning(f"Ignoring callback metric {message}")

        metrics: dict[str, float] = {}
        nonfinite: dict[str, int] = {}
        if names:
            values = packed.cpu().tolist()
            for name, (total, count, invalid_count) in zip(names, values, strict=True):
                if count > 0:
                    metrics[name] = total / count
                if invalid_count > 0:
                    nonfinite[name] = int(invalid_count)
        if extra_metrics is not None:
            metrics.update(extra_metrics)
        if nonfinite:
            details = ", ".join(
                f"{name}={count}" for name, count in sorted(nonfinite.items())
            )
            log.warning(
                f"Excluded non-finite metric observations at iteration {iteration}: "
                f"{details}"
            )
        if metrics:
            self._write_to_sinks(metrics, iteration=iteration)

    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, iteration
        self._accumulate(
            self._train,
            self._train_ignored,
            output_batch,
            loss,
            split="train",
        )

    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, output_batch, loss
        if self.report_update_speed:
            self._update_window_size += 1
        if (
            self.every_n is not None
            and self.every_n > 0
            and iteration % self.every_n == 0
        ):
            self._flush(
                self._train,
                self._train_ignored,
                iteration=iteration,
                gauges=self._train_gauges,
                extra_metrics=self._take_update_speed_metrics(),
            )

    def on_train_end(self, model: Any, iteration: int = 0) -> None:
        del model
        self._flush(
            self._train,
            self._train_ignored,
            iteration=iteration,
            gauges=self._train_gauges,
            extra_metrics=self._take_update_speed_metrics(),
        )

    def on_validation_start(
        self, model: Any, dataloader_val: Any, iteration: int = 0
    ) -> None:
        del model, dataloader_val, iteration
        self._validation.clear()
        self._validation_ignored.clear()

    def on_validation_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, iteration
        weight = self._scalar_tensor(
            data_batch.get("validation_metric_weight", 1.0),
            reference=loss,
        )
        if weight is None:
            self._validation_ignored.add(
                "validation_metric_weight (expected one scalar tensor or number)"
            )
            weight = torch.zeros((), device=loss.device, dtype=torch.float32)
        self._accumulate(
            self._validation,
            self._validation_ignored,
            output_batch,
            loss,
            split="validation",
            weight=weight,
        )

    def on_validation_end(self, model: Any, iteration: int = 0) -> None:
        del model
        self._flush(
            self._validation,
            self._validation_ignored,
            iteration=iteration,
        )

    def on_app_end(self) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as error:  # logging cleanup must not fail the job
                log.warning(
                    f"Failed to close metric sink {type(sink).__name__}: "
                    f"{type(error).__name__}: {error}"
                )


__all__ = ["LoguruMetricSink", "MetricSink", "ScalarMetricCallback"]
