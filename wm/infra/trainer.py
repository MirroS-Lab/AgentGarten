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

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

from wm.infra.batch_transport import fetch_batch, prepare_batch
from wm.infra.callbacks import CallbackGroup
from wm.infra.checkpoint import DistributedCheckpointer
from wm.infra.distributed import DistributedDataParallel, ddp_sync_grad, is_rank0
from wm.infra.model import Model
from wm.infra.nonfinite import synchronize_found_nonfinite
from wm.infra.rng_state import isolated_rng
from wm.infra.utils import preserve_training_mode, to

__all__ = ["Trainer", "TrainingStepResult"]

Batch = dict[str, Any]


@contextmanager
def _preserve_validation_cursor(loader: Iterable[Any]) -> Iterator[None]:
    state_dict = getattr(loader, "state_dict", None)
    restore = getattr(loader, "load_state_dict", None)
    stateful = callable(state_dict) and callable(restore)
    state = deepcopy(state_dict()) if stateful else None
    try:
        yield
    finally:
        if stateful:
            restore(state)


def _checkpoint_registry_subset(
    registry: Mapping[str, Any],
    names: Iterable[str] | str | None,
    *,
    label: str,
) -> dict[str, Any]:
    if names is None:
        return dict(registry)
    requested = (names,) if isinstance(names, str) else tuple(names)
    requested = tuple(dict.fromkeys(str(name) for name in requested))
    missing = tuple(name for name in requested if name not in registry)
    if missing:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown checkpoint {label} name(s): {', '.join(missing)}; "
            f"available: {available}"
        )
    return {name: registry[name] for name in requested}


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    output_batch: Batch
    loss: torch.Tensor
    grad_accum_step: int
    did_update: bool
    found_nonfinite: bool = False


def _unwrapped_model(model: torch.nn.Module) -> Model:
    if isinstance(model, DistributedDataParallel):
        return model.training_model  # type: ignore[return-value]
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise TypeError(
            "Trainer closures cannot run through stock PyTorch DDP. Use "
            "wm.infra.distributed.wrap_ddp(), whose forward_closure() enters "
            "the DDP forward/reducer lifecycle."
        )
    if not isinstance(model, Model):
        raise TypeError("Trainer requires a wm.infra.Model training root")
    return model


def _with_context_parallel_topology(
    model: Model, runtime_metadata: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    topology = model.context_parallel_context.checkpoint_topology()
    if topology is None:
        return runtime_metadata
    runtime = dict(runtime_metadata or {})
    mesh = dict(runtime.get("mesh") or {})
    mesh["context_parallel"] = topology
    runtime["mesh"] = mesh
    return runtime


class Trainer:
    def __init__(
        self,
        *,
        grad_accum_steps: int = 1,
        callbacks: CallbackGroup | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least one")
        self.grad_accum_steps = int(grad_accum_steps)
        self.periodic_iteration_offset = 0
        self.device = None if device is None else torch.device(device)
        self.callbacks = (
            callbacks
            if callbacks is not None
            else CallbackGroup(config=None, trainer=self)
        )
        for callback in self.callbacks:
            callback.trainer = self

    # ------------------------------------------------------------------ data
    def _prepare_data_batch(self, model: Model, data_batch: Batch) -> Batch:
        if self.device is not None:
            data_batch = to(data_batch, device=self.device)
        return model.prepare_data_batch(data_batch)

    def _next_batch(
        self, model: Model, iterator: Iterator[Batch], *, sequence_index: int
    ) -> Batch | None:
        # ``None`` once the iterator is exhausted (on every rank together).
        context = model.context_parallel_context
        data_batch, exhausted, owner_rank = fetch_batch(
            context, iterator, sequence_index=sequence_index
        )
        if exhausted:
            return None
        return prepare_batch(
            context,
            data_batch,
            owner_rank=owner_rank,
            prepare=partial(self._prepare_data_batch, model),
        )

    # ----------------------------------------------------------------- train
    def train(
        self,
        model: Model,
        dataloader_train: Iterable[Batch],
        dataloader_val: Iterable[Batch] | None = None,
        *,
        grad_scaler: torch.amp.GradScaler,
        max_iterations: int | None,
        checkpointer: DistributedCheckpointer | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_components: Iterable[str] | None = None,
        checkpoint_optimizer_names: Iterable[str] | str | None = None,
        checkpoint_scheduler_names: Iterable[str] | str | None = None,
        checkpoint_model_key_prefix: str | None = None,
        checkpoint_interval: int | None = None,
        validation_interval: int | None = None,
        periodic_iteration_offset: int = 0,
        max_val_steps: int | None = None,
        validate_at_start: bool = False,
        save_final_checkpoint: bool = True,
        model_wrapper: Callable[[Model], torch.nn.Module] | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Run the training loop; the keywords mirror the ``train`` config section."""
        if periodic_iteration_offset < 0:
            raise ValueError("periodic_iteration_offset must be non-negative")
        self.periodic_iteration_offset = int(periodic_iteration_offset)
        try:
            iteration = self._train(
                model,
                dataloader_train,
                dataloader_val,
                grad_scaler=grad_scaler,
                max_iterations=max_iterations,
                checkpointer=checkpointer,
                load_options={
                    "path": checkpoint_path,
                    "components": checkpoint_components,
                    "optimizer_names": checkpoint_optimizer_names,
                    "scheduler_names": checkpoint_scheduler_names,
                    "model_key_prefix": checkpoint_model_key_prefix,
                },
                checkpoint_interval=checkpoint_interval,
                validation_interval=validation_interval,
                max_val_steps=max_val_steps,
                validate_at_start=validate_at_start,
                save_final_checkpoint=save_final_checkpoint,
                model_wrapper=model_wrapper,
                runtime_metadata=_with_context_parallel_topology(
                    model, runtime_metadata
                ),
            )
        except BaseException as error:
            self._close(checkpointer, error)
            raise
        self._close(checkpointer, None)
        return iteration

    def _close(
        self,
        checkpointer: DistributedCheckpointer | None,
        training_error: BaseException | None,
    ) -> None:
        # Finish the pending checkpoint and release callbacks. A failure here
        # never masks the training error; without one, it is raised.
        cleanup_error: BaseException | None = None
        if checkpointer is not None:
            try:
                if training_error is None:
                    checkpointer.finalize()
                else:
                    checkpointer.finalize_after_error()
            except BaseException as error:
                cleanup_error = error
                if training_error is not None:
                    warnings.warn(
                        "Checkpoint finalization failed while handling a "
                        f"training error: {type(error).__name__}: {error}",
                        stacklevel=2,
                    )
        try:
            self.callbacks.on_app_end()
        except Exception as callback_error:
            if training_error is not None:
                warnings.warn(
                    "Callback cleanup failed while handling a training error: "
                    f"{type(callback_error).__name__}: {callback_error}",
                    stacklevel=2,
                )
            elif cleanup_error is None:
                cleanup_error = callback_error
            else:
                warnings.warn(
                    "Callback cleanup also failed after checkpoint "
                    f"finalization: {type(callback_error).__name__}: "
                    f"{callback_error}",
                    stacklevel=2,
                )
        if training_error is None and cleanup_error is not None:
            raise cleanup_error

    def _restore(
        self,
        checkpointer: DistributedCheckpointer,
        model: Model,
        grad_scaler: torch.amp.GradScaler,
        dataloader_train: Iterable[Batch],
        *,
        load_options: Mapping[str, Any],
        runtime_metadata: Mapping[str, Any] | None,
    ) -> int:
        load_kwargs = {
            "path": load_options["path"],
            "components": load_options["components"],
            "dataloader": dataloader_train,
            "runtime_metadata": runtime_metadata,
        }
        if load_options["model_key_prefix"] is not None:
            load_kwargs["model_key_prefix"] = load_options["model_key_prefix"]
        result = checkpointer.load(
            model,
            _checkpoint_registry_subset(
                model.optimizer_dict, load_options["optimizer_names"], label="optimizer"
            ),
            _checkpoint_registry_subset(
                model.scheduler_dict, load_options["scheduler_names"], label="scheduler"
            ),
            grad_scaler,
            **load_kwargs,
        )
        return result.iteration

    def _train(
        self,
        model: Model,
        dataloader_train: Iterable[Batch],
        dataloader_val: Iterable[Batch] | None,
        *,
        grad_scaler: torch.amp.GradScaler,
        max_iterations: int | None,
        checkpointer: DistributedCheckpointer | None,
        load_options: Mapping[str, Any],
        checkpoint_interval: int | None,
        validation_interval: int | None,
        max_val_steps: int | None,
        validate_at_start: bool,
        save_final_checkpoint: bool,
        model_wrapper: Callable[[Model], torch.nn.Module] | None,
        runtime_metadata: Mapping[str, Any] | None,
    ) -> int:
        iteration = 0
        if checkpointer is not None:
            iteration = self._restore(
                checkpointer,
                model,
                grad_scaler,
                dataloader_train,
                load_options=load_options,
                runtime_metadata=runtime_metadata,
            )

        def running() -> bool:
            return max_iterations is None or iteration < max_iterations

        def new_train_iterator() -> Iterator[Batch]:
            iterator = iter(dataloader_train)
            # The checkpoint RNG is restored only after the loader has started
            # its workers, so worker start-up does not shift the stream.
            if checkpointer is not None:
                checkpointer.restore_pending_rng_state()
            return iterator

        def save() -> None:
            checkpointer.save(
                model,
                model.optimizer_dict,
                model.scheduler_dict,
                grad_scaler,
                iteration=iteration,
                dataloader=dataloader_train,
                runtime_metadata=runtime_metadata,
            )

        def cadence_due(interval: int | None) -> bool:
            return model.is_cadence_due(
                iteration - self.periodic_iteration_offset, interval
            )

        # DDP must be constructed after DCP has restored optimizer state. FSDP2
        # callers leave this factory unset because their roots were
        # parallelized before optimizer construction.
        model_ddp = model if model_wrapper is None else model_wrapper(model)
        # Validate the execution-root contract before changing module mode or
        # entering any callback/data lifecycle.
        _unwrapped_model(model_ddp)
        model_ddp.train()
        self.callbacks.on_train_start(model, iteration=iteration)

        # Validation sinks such as W&B start background threads on rank 0.
        # Forking DataLoader workers after that point can deadlock only rank 0
        # while its peers wait in the next model collective. Start the first
        # epoch workers after restore but before validation; no training batch
        # is consumed until validation has completed.
        train_iterator = new_train_iterator() if running() else None
        if validate_at_start and dataloader_val is not None:
            self.validate(
                model, dataloader_val, iteration=iteration, max_val_steps=max_val_steps
            )

        grad_accum_step = 0
        successful_updates = 0
        last_saved_iteration: int | None = None
        while running():
            if train_iterator is None:
                train_iterator = new_train_iterator()
            epoch_had_batch = False
            while running():
                self.callbacks.on_before_dataloading(iteration=iteration)
                data_batch = self._next_batch(
                    model,
                    train_iterator,
                    sequence_index=iteration * self.grad_accum_steps + grad_accum_step,
                )
                if data_batch is None:
                    break
                epoch_had_batch = True
                if not model_ddp.training:
                    model_ddp.train()
                result = self.training_step(
                    model_ddp,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_step=grad_accum_step,
                )
                grad_accum_step = result.grad_accum_step
                if not result.did_update:
                    continue
                iteration += 1
                successful_updates += 1

                if checkpointer is not None and cadence_due(checkpoint_interval):
                    save()
                    last_saved_iteration = iteration
                if dataloader_val is not None and cadence_due(validation_interval):
                    # Publish the completed snapshot before potentially long
                    # evaluation. Otherwise an async save can leave `latest`
                    # stale until the next save, starving external evaluators.
                    # All ranks enter this lifecycle boundary together; never
                    # run checkpointer completion collectives in a background
                    # callback alongside model collectives.
                    if checkpointer is not None:
                        checkpointer.finalize()
                    self.validate(
                        model,
                        dataloader_val,
                        iteration=iteration,
                        max_val_steps=max_val_steps,
                    )
            train_iterator = None
            if running() and not epoch_had_batch:
                if is_rank0():
                    warnings.warn(
                        "Training dataloader produced no batches; ending training.",
                        stacklevel=2,
                    )
                break

        if (
            checkpointer is not None
            and save_final_checkpoint
            and successful_updates > 0
            and last_saved_iteration != iteration
        ):
            save()
        self.callbacks.on_train_end(model, iteration=iteration)
        return iteration

    # ------------------------------------------------------------------ step
    def training_step(
        self,
        model_ddp: torch.nn.Module,
        grad_scaler: torch.amp.GradScaler,
        data_batch: dict[str, Any],
        *,
        iteration: int = 0,
        grad_accum_step: int = 0,
    ) -> TrainingStepResult:
        if not 0 <= grad_accum_step < self.grad_accum_steps:
            raise ValueError(
                "grad_accum_step must be in "
                f"[0, {self.grad_accum_steps}), got {grad_accum_step}"
            )

        model = _unwrapped_model(model_ddp)
        bindings = model.get_optimizer_bindings(iteration)
        active_optimizers = tuple(binding.optimizer for binding in bindings)
        optimizer_names = tuple(binding.name for binding in bindings)

        self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)

        is_last_accum = grad_accum_step == self.grad_accum_steps - 1
        output_batch: dict[str, Any] = {}
        detached_loss: torch.Tensor | None = None
        saw_last_closure = False
        closure_count = 0

        for _name, closure, is_last_closure in model.training_step_closures(
            data_batch, iteration
        ):
            if saw_last_closure:
                raise RuntimeError(
                    "training_step_closures() yielded a closure after "
                    "is_last=True; the final marker must be on the last closure"
                )
            if not isinstance(is_last_closure, bool):
                raise TypeError(
                    "training_step_closures() is_last must be a bool, got "
                    f"{type(is_last_closure).__name__}"
                )
            closure_count += 1
            saw_last_closure = is_last_closure

            sync_this_backward = is_last_accum and bool(is_last_closure)
            with ddp_sync_grad(model_ddp, sync_this_backward):
                if isinstance(model_ddp, DistributedDataParallel):
                    output, loss = model_ddp.forward_closure(closure)
                else:
                    output, loss = closure()

                output_batch.update(output)

                grad_scaler.scale(loss / self.grad_accum_steps).backward()

            loss_value = loss.detach()
            detached_loss = (
                loss_value if detached_loss is None else detached_loss + loss_value
            )

        if closure_count == 0 or detached_loss is None:
            raise RuntimeError("training_step_closures() yielded no closures")
        if not saw_last_closure:
            raise RuntimeError(
                "training_step_closures() must mark exactly its final closure "
                "with is_last=True"
            )

        next_grad_accum_step = grad_accum_step + 1
        if next_grad_accum_step < self.grad_accum_steps:
            self.callbacks.on_training_step_batch_end(
                model,
                data_batch,
                output_batch,
                detached_loss,
                iteration=iteration,
            )
            return TrainingStepResult(
                output_batch=output_batch,
                loss=detached_loss,
                grad_accum_step=next_grad_accum_step,
                did_update=False,
            )

        for optimizer in active_optimizers:
            grad_scaler.unscale_(optimizer)

        for binding in bindings:
            optimizer, scheduler = binding.optimizer, binding.scheduler
            self.callbacks.on_before_optimizer_step(
                model_ddp,
                optimizer,
                scheduler,
                grad_scaler,
                iteration=iteration,
            )

        found_nonfinite = synchronize_found_nonfinite(
            grad_scaler,
            active_optimizers,
            fallback_device=detached_loss.device,
        )
        did_update = not found_nonfinite

        if did_update:
            for binding in bindings:
                name, optimizer, scheduler = (
                    binding.name,
                    binding.optimizer,
                    binding.scheduler,
                )
                grad_scaler.step(optimizer)
                scheduler.step()
                self.callbacks.on_after_optimizer_step(
                    model_ddp,
                    name,
                    optimizer,
                    scheduler,
                    iteration=iteration,
                )
        else:
            self.callbacks.on_optimizer_step_skipped(
                model_ddp, optimizer_names, iteration=iteration
            )

        for binding in bindings:
            optimizer, scheduler = binding.optimizer, binding.scheduler
            self.callbacks.on_before_zero_grad(
                model_ddp, optimizer, scheduler, iteration=iteration
            )
            optimizer.zero_grad(set_to_none=True)

        # Every active optimizer shares this GradScaler, so update exactly once.
        grad_scaler.update()

        self.callbacks.on_training_step_batch_end(
            model,
            data_batch,
            output_batch,
            detached_loss,
            iteration=iteration,
        )
        if did_update:
            self.callbacks.on_training_step_end(
                model,
                data_batch,
                output_batch,
                detached_loss,
                iteration=iteration + 1,
            )

        return TrainingStepResult(
            output_batch=output_batch,
            loss=detached_loss,
            grad_accum_step=0,
            did_update=did_update,
            found_nonfinite=found_nonfinite,
        )

    # -------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(
        self,
        model: Model,
        dataloader_val: Iterable[Batch],
        *,
        iteration: int = 0,
        max_val_steps: int | None = None,
    ) -> None:
        with (
            preserve_training_mode(model),
            _preserve_validation_cursor(dataloader_val),
            isolated_rng(),
        ):
            self.callbacks.on_validation_start(
                model, dataloader_val, iteration=iteration
            )
            model.eval()
            iterator = iter(dataloader_val)
            batch_index = 0
            while max_val_steps is None or batch_index < max_val_steps:
                data_batch = self._next_batch(
                    model, iterator, sequence_index=batch_index
                )
                if data_batch is None:
                    break
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(
                    model, data_batch, output_batch, loss, iteration=iteration
                )
                batch_index += 1
            self.callbacks.on_validation_end(model, iteration=iteration)
