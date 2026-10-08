# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import torch

from wm.infra import log
from wm.infra.distributed import get_rank
from wm.infra.model import LifecycleCadence
from wm.infra.rng_state import isolated_rng
from wm.infra.utils import map_tensors, preserve_training_mode

from .base import Callback

_SampleSource = Literal["train", "validation"]


class ArtifactWriter(Protocol):
    def write(
        self,
        value: Any,
        *,
        output_dir: str,
        split: str,
        iteration: int,
        rank: int,
        sample_batch_index: int = 0,
    ) -> None: ...


class ArtifactWriterGroup:
    def __init__(self, writers: Sequence[ArtifactWriter | None]) -> None:
        self.writers = tuple(writer for writer in writers if writer is not None)
        self._disabled_writers: set[int] = set()

    def write(
        self,
        value: Any,
        *,
        output_dir: str,
        split: str,
        iteration: int,
        rank: int,
        sample_batch_index: int = 0,
    ) -> None:
        for writer in self.writers:
            identity = id(writer)
            if identity in self._disabled_writers:
                continue
            try:
                writer.write(
                    value,
                    output_dir=output_dir,
                    split=split,
                    iteration=iteration,
                    rank=rank,
                    sample_batch_index=sample_batch_index,
                )
            except Exception as error:
                self._disabled_writers.add(identity)
                log.warning(
                    f"Disabling artifact writer {type(writer).__name__} after "
                    f"write failure: {type(error).__name__}: {error}"
                )

    def close(self) -> None:
        for writer in self.writers:
            close = getattr(writer, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as error:
                log.warning(
                    f"Failed to close artifact writer {type(writer).__name__}: "
                    f"{type(error).__name__}: {error}"
                )


def _clone_sample_batch(value: Any) -> Any:
    return map_tensors(value, lambda tensor: tensor.detach().clone())


def _sample_seed_from_batch(value: Any) -> int | tuple[int, ...]:
    if torch.is_tensor(value):
        values = value.detach().to(device="cpu").reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return int(value)
    if not values:
        raise ValueError("sample seed batch metadata must not be empty")
    seeds = tuple(int(seed) for seed in values)
    return seeds[0] if len(seeds) == 1 else seeds


class PeriodicSampleCallback(Callback):
    def __init__(
        self,
        every_n: int | None,
        output_dir: str,
        writer: ArtifactWriter,
        *,
        source: _SampleSource = "validation",
        max_batches: int = 1,
        sampling_options: Mapping[str, Any] | None = None,
        seed_batch_key: str | None = "validation_sample_seed",
        artifact_formatter: Any | None = None,
    ) -> None:
        super().__init__()
        if source not in ("train", "validation"):
            raise ValueError(
                f"sample source must be 'train' or 'validation', got {source!r}"
            )
        self.every_n = None if every_n is None else int(every_n)
        self.output_dir = str(output_dir)
        self.writer = writer
        self.source: _SampleSource = source
        if int(max_batches) < 0:
            raise ValueError(f"max_batches must be non-negative, got {max_batches}")
        self.max_batches = int(max_batches)
        self.sampling_options = dict(sampling_options or {})
        self.seed_batch_key = None if seed_batch_key is None else str(seed_batch_key)
        self.artifact_formatter = artifact_formatter
        self._writer_enabled = True
        self._validation_pending = False
        self._validation_batch_index = 0

    def _is_due(self, iteration: int, model: Any | None = None) -> bool:
        if self.every_n is None or self.every_n <= 0:
            return False
        if self.trainer is not None:
            iteration = int(iteration) - self.trainer.periodic_iteration_offset
        # Startup validation is explicitly requested by Trainer and retains
        # the historical iteration-zero sample for every objective.
        if int(iteration) == 0:
            return True
        if isinstance(model, LifecycleCadence):
            return bool(model.is_cadence_due(int(iteration), self.every_n))
        return int(iteration) % self.every_n == 0

    @staticmethod
    @torch.inference_mode()
    def _call_sample(
        model: Any, data_batch: dict[str, Any], sampling_options: Mapping[str, Any]
    ) -> Any:
        return model.sample(data_batch, **sampling_options)

    def _sampling_options_for_batch(
        self, data_batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        options = dict(self.sampling_options)
        if self.seed_batch_key is not None and self.seed_batch_key in data_batch:
            options["seed"] = _sample_seed_from_batch(data_batch[self.seed_batch_key])
        return options

    def _write(
        self,
        value: Any,
        *,
        model: Any,
        data_batch: dict[str, Any],
        split: str,
        iteration: int,
        sample_batch_index: int,
        sampling_options: Mapping[str, Any],
    ) -> None:
        if value is None or not self._writer_enabled:
            return
        context = getattr(model, "context_parallel_context", None)
        if context is not None and not context.is_leader:
            return
        artifact_rank = get_rank() if context is None else int(context.artifact_rank)
        try:
            if self.artifact_formatter is not None:
                value = self.artifact_formatter(
                    value,
                    model=model,
                    data_batch=data_batch,
                    sampling_options=sampling_options,
                )
            self.writer.write(
                value,
                output_dir=self.output_dir,
                split=split,
                iteration=iteration,
                rank=artifact_rank,
                sample_batch_index=sample_batch_index,
            )
        except Exception as error:  # artifact IO must not discard a model update
            self._writer_enabled = False
            log.warning(
                f"Disabling artifact writer {type(self.writer).__name__} after "
                f"write failure: {type(error).__name__}: {error}"
            )

    @isolated_rng()
    def _sample_training_batch(
        self, model: Any, data_batch: dict[str, Any], *, iteration: int
    ) -> None:
        batch = _clone_sample_batch(data_batch)
        sampling_options = self._sampling_options_for_batch(batch)
        with preserve_training_mode(model):
            model.eval()
            value = self._call_sample(model, batch, sampling_options)
        self._write(
            value,
            model=model,
            data_batch=batch,
            split="train",
            iteration=iteration,
            sample_batch_index=0,
            sampling_options=sampling_options,
        )

    @isolated_rng()
    def _sample_validation_batch(
        self,
        model: Any,
        data_batch: dict[str, Any],
        *,
        iteration: int,
        sample_batch_index: int,
    ) -> None:
        batch = _clone_sample_batch(data_batch)
        sampling_options = self._sampling_options_for_batch(batch)
        # Trainer owns eval mode and isolates the whole validation pass.
        # Also isolate each preview so enabling it (including leader-only
        # formatters/writers) cannot change later validation batches' RNG.
        value = self._call_sample(model, batch, sampling_options)
        self._write(
            value,
            model=model,
            data_batch=batch,
            split="validation",
            iteration=iteration,
            sample_batch_index=sample_batch_index,
            sampling_options=sampling_options,
        )

    def _run_sample_batch(
        self,
        model: Any,
        data_batch: dict[str, Any],
        *,
        split: _SampleSource,
        iteration: int,
        sample_batch_index: int = 0,
    ) -> None:
        started_at = time.perf_counter()
        log.info(
            f"Sample start: split={split}, iteration={iteration}, "
            f"batch={sample_batch_index}",
            rank0_only=False,
        )
        try:
            if split == "train":
                self._sample_training_batch(model, data_batch, iteration=iteration)
            else:
                self._sample_validation_batch(
                    model,
                    data_batch,
                    iteration=iteration,
                    sample_batch_index=sample_batch_index,
                )
        except Exception:
            elapsed = time.perf_counter() - started_at
            log.exception(
                f"Sample failed: split={split}, iteration={iteration}, "
                f"batch={sample_batch_index}, "
                f"elapsed_s={elapsed:.3f}",
                rank0_only=False,
            )
            raise
        elapsed = time.perf_counter() - started_at
        log.info(
            f"Sample end: split={split}, iteration={iteration}, "
            f"batch={sample_batch_index}, "
            f"elapsed_s={elapsed:.3f}",
            rank0_only=False,
        )

    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del output_batch, loss
        if self.source == "train" and self._is_due(iteration, model):
            self._run_sample_batch(
                model,
                data_batch,
                split="train",
                iteration=iteration,
            )

    def on_validation_start(
        self, model: Any, dataloader_val: Any, iteration: int = 0
    ) -> None:
        del dataloader_val
        self._validation_batch_index = 0
        self._validation_pending = (
            self.source == "validation"
            and self.max_batches > 0
            and self._is_due(iteration, model)
        )

    def on_validation_step_end(
        self,
        model: Any,
        data: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del output_batch, loss
        if not self._validation_pending:
            return
        sample_batch_index = self._validation_batch_index
        self._validation_batch_index += 1
        if sample_batch_index >= self.max_batches:
            self._validation_pending = False
            return
        # Trainer prepares this batch once before validation_step. Sampling
        # afterwards reuses those latents instead of encoding raw video again.
        self._run_sample_batch(
            model,
            data,
            split="validation",
            iteration=iteration,
            sample_batch_index=sample_batch_index,
        )
        if self._validation_batch_index >= self.max_batches:
            self._validation_pending = False

    def on_validation_end(self, model: Any, iteration: int = 0) -> None:
        del model, iteration
        self._validation_pending = False
        self._validation_batch_index = 0

    def on_app_end(self) -> None:
        close = getattr(self.writer, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as error:
            log.warning(
                f"Failed to close artifact writer {type(self.writer).__name__}: "
                f"{type(error).__name__}: {error}"
            )


__all__ = [
    "ArtifactWriter",
    "ArtifactWriterGroup",
    "PeriodicSampleCallback",
]
