# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any
from weakref import WeakSet, WeakValueDictionary

from wm.infra import log
from wm.infra.distributed import is_rank0

from .media import (
    _cpu_media_artifacts,
    _encode_cpu_video,
    _grid_options,
)


def _import_wandb() -> Any:
    return importlib.import_module("wandb")


# Metric and media sinks share one process-wide W&B run.  Validation media can
# be emitted while the asynchronous scalar worker is draining its queue, so
# serialize initialization and SDK calls across the two sink instances.
_WANDB_INIT_LOCK = threading.Lock()
_WANDB_LOG_LOCK = threading.Lock()


@dataclass
class _SharedRun:
    sdk: Any
    clients: WeakSet = field(default_factory=WeakSet)
    run: Any = None
    initialized: bool = False
    owned: bool = False


_SHARED_RUNS: WeakValueDictionary[int, _SharedRun] = WeakValueDictionary()


class _WandbRunClient:
    def __init__(
        self,
        project: str,
        group: str | None = None,
        name: str | None = None,
        mode: str | None = None,
        step_metrics: Mapping[str, str] | None = None,
    ) -> None:
        self.project = project
        self.group = group
        self.name = name
        self.mode = mode
        self.step_metrics = dict(step_metrics or {})
        self._wandb: Any | None = None
        self._shared_run: _SharedRun | None = None
        self._closed = False
        self._step_metrics_defined = False

    def _reserve_run(self) -> _SharedRun | None:
        with _WANDB_INIT_LOCK:
            if self._closed:
                return None
            if self._shared_run is not None:
                return self._shared_run
            try:
                wandb = _import_wandb()
            except ModuleNotFoundError as error:
                if error.name != "wandb":
                    raise
                raise ModuleNotFoundError(
                    "W&B logging requires the optional dependency; install wm[wandb]"
                ) from error
            shared = _SHARED_RUNS.get(id(wandb))
            if shared is None:
                shared = _SharedRun(wandb)
                _SHARED_RUNS[id(wandb)] = shared
            shared.clients.add(self)
            self._shared_run = shared
            self._wandb = wandb
            return shared

    def _initialize_locked(self, shared: _SharedRun) -> Any:
        wandb = shared.sdk
        if not shared.initialized:
            if getattr(wandb, "run", None) is None:
                api_key = os.environ.get("WANDB_API_KEY")
                if api_key:
                    wandb.login(key=api_key, relogin=False)
                try:
                    wandb.init(
                        project=self.project,
                        group=self.group,
                        name=self.name,
                        mode=self.mode,
                    )
                finally:
                    # Even partial SDK initialization belongs to this group
                    # and must be released if later setup raises.
                    shared.run = getattr(wandb, "run", None)
                    shared.initialized = shared.run is not None
                    shared.owned = shared.initialized
            else:
                shared.run = wandb.run
                shared.initialized = True
        if getattr(wandb, "run", None) is not shared.run:
            raise RuntimeError("The shared W&B run was replaced or finished externally")
        if not self._step_metrics_defined:
            define_metric = getattr(wandb, "define_metric", None)
            if callable(define_metric):
                for pattern, step_metric in self.step_metrics.items():
                    define_metric(pattern, step_metric=step_metric)
            self._step_metrics_defined = True
        return wandb

    def _initialize(self) -> Any | None:
        shared = self._reserve_run()
        if shared is None:
            return None
        with _WANDB_LOG_LOCK:
            if self._closed:
                return None
            return self._initialize_locked(shared)

    def _log(self, payload: Mapping[str, Any], *, iteration: int) -> None:
        shared = self._reserve_run()
        if shared is None:
            return
        with _WANDB_LOG_LOCK:
            if self._closed:
                return
            wandb = self._initialize_locked(shared)
            wandb.log(dict(payload), step=int(iteration))

    def close(self) -> None:
        if not is_rank0():
            return
        with _WANDB_INIT_LOCK:
            if self._closed:
                return
            self._closed = True
            shared = self._shared_run
            self._shared_run = None
            self._wandb = None
            if shared is None:
                return
            shared.clients.discard(self)
            if shared.clients:
                return
            # Lock order is registry -> SDK. SDK calls never acquire the
            # registry lock; a last close can safely wait for an in-flight log.
            with _WANDB_LOG_LOCK:
                try:
                    if (
                        shared.owned
                        and shared.run is not None
                        and getattr(shared.sdk, "run", None) is shared.run
                    ):
                        shared.sdk.finish()
                finally:
                    _SHARED_RUNS.pop(id(shared.sdk), None)


class WandbMetricSink(_WandbRunClient):
    def __init__(
        self,
        project: str,
        group: str | None = None,
        name: str | None = None,
        mode: str | None = None,
        *,
        async_write: bool = False,
        step_metrics: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            project=project,
            group=group,
            name=name,
            mode=mode,
            step_metrics=step_metrics,
        )
        self.async_write = bool(async_write)
        self._queue: SimpleQueue[tuple[dict[str, float], int] | object] | None = None
        self._worker: threading.Thread | None = None
        self._worker_stop = object()
        self._worker_error: BaseException | None = None
        self._worker_lock = threading.Lock()
        self._closing = False

    def _run_async(self) -> None:
        queue = self._queue
        if queue is None:
            return
        while True:
            item = queue.get()
            if item is self._worker_stop:
                return
            try:
                metrics, iteration = item  # type: ignore[misc]
                self._log(metrics, iteration=iteration)
            except BaseException as error:  # logging must not stop training
                self._worker_error = error
                log.warning(
                    "Disabling asynchronous W&B metric writes after failure: "
                    f"{type(error).__name__}: {error}"
                )
                # Drain queued payloads without blocking the training thread
                # on an unavailable network/service.  The sentinel is kept so
                # ``close`` can join this worker deterministically.
                while queue.get() is not self._worker_stop:
                    pass
                return

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        self._queue = SimpleQueue()
        self._worker = threading.Thread(
            target=self._run_async,
            name="wm-wandb-metrics",
            daemon=True,
        )
        self._worker.start()

    def write(self, metrics: Mapping[str, float], *, iteration: int) -> None:
        if not is_rank0():
            return
        if not self.async_write:
            self._log(metrics, iteration=iteration)
            return
        with self._worker_lock:
            if self._closing or self._closed or self._worker_error is not None:
                return
            try:
                self._reserve_run()
            except Exception as error:
                self._worker_error = error
                log.warning(
                    "Disabling asynchronous W&B metric writes after failure: "
                    f"{type(error).__name__}: {error}"
                )
                return
            self._ensure_worker()
            assert self._queue is not None
            # Copy the tiny scalar payload so the caller can immediately
            # release its reduction tensors and continue the next update.
            self._queue.put((dict(metrics), int(iteration)))

    def close(self) -> None:
        if self.async_write:
            with self._worker_lock:
                worker = self._worker
                queue = self._queue
                if not self._closing and worker is not None and queue is not None:
                    queue.put(self._worker_stop)
                self._closing = True
            if worker is not None and worker.ident is not None:
                worker.join()
            if self._worker_error is not None:
                log.warning(
                    "W&B metric worker stopped: "
                    f"{type(self._worker_error).__name__}: {self._worker_error}"
                )
        super().close()


class WandbArtifactWriter(_WandbRunClient):
    def __init__(
        self,
        project: str,
        group: str | None = None,
        name: str | None = None,
        mode: str | None = None,
        *,
        layout: str = "BCTHW",
        float_range: str = "minus_one_one",
        fps: int = 16,
        frame_chunk_size: int = 8,
        grid_layout: Sequence[Any] | None = None,
        grid_condition_keys: Sequence[Any] | None = None,
        grid_reference_key: Any | None = None,
        grid_name: str = "grid",
        save_individual_artifacts: bool = True,
        video_quality: int = 5,
    ) -> None:
        super().__init__(project=project, group=group, name=name, mode=mode)
        self.layout = layout
        self.float_range = float_range
        self.fps = int(fps)
        self.frame_chunk_size = int(frame_chunk_size)
        (
            self.grid_layout,
            self.grid_condition_keys,
            self.grid_reference_key,
        ) = _grid_options(
            grid_layout,
            grid_condition_keys,
            grid_reference_key,
        )
        self.grid_name = str(grid_name)
        self.save_individual_artifacts = bool(save_individual_artifacts)
        self.video_quality = int(video_quality)

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
        del output_dir, rank
        if not is_rank0():
            return
        wandb = self._initialize()
        if wandb is None:
            return
        payload: dict[str, Any] = {}
        for artifact_name, batch_index, batch_size, pixels in _cpu_media_artifacts(
            value,
            layout=self.layout,
            float_range=self.float_range,
            frame_chunk_size=self.frame_chunk_size,
            grid_layout=self.grid_layout,
            grid_condition_keys=self.grid_condition_keys,
            grid_reference_key=self.grid_reference_key,
            grid_name=self.grid_name,
            save_individual_artifacts=self.save_individual_artifacts,
        ):
            suffix = "" if batch_size == 1 else f"/{batch_index:03d}"
            key = f"{split}/batch_{int(sample_batch_index):04d}/{artifact_name}{suffix}"
            if pixels.shape[0] == 1:
                payload[key] = wandb.Image(pixels[0].numpy())
            else:
                # Raw TCHW frames make W&B invoke MoviePy. MoviePy 1.x is
                # incompatible with decorator 5.x and can lose ``fps`` before
                # constructing ffmpeg arguments. Reuse the local EasyIO
                # encoder and give W&B an already encoded artifact instead.
                encoded = _encode_cpu_video(
                    pixels,
                    fps=self.fps,
                    quality=self.video_quality,
                )
                payload[key] = wandb.Video(encoded, format="mp4")
        if payload:
            self._log(payload, iteration=int(iteration))


__all__ = ["WandbArtifactWriter", "WandbMetricSink"]
