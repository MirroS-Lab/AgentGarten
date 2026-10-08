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

import re
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemWriter,
)
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from wm.infra import io as easy_io
from wm.infra import log
from wm.infra.checkpoint.common import warn_rank0
from wm.infra.checkpoint.optimizer_state import (
    OPTIMIZER_STATE_OPTIONS,
    UnsavedOptimizerState,
    flattened_optimizer_group_options,
    optimizer_group_options,
    optimizer_options_differ,
    optimizer_state_for_save,
    remove_unsaved_optimizer_template_state,
    restore_optimizer_group_options,
    restore_unsaved_optimizer_state,
    snapshot_unsaved_optimizer_state,
)
from wm.infra.checkpoint.plan import (
    ModelRestorePlan,
    plan_model_initializations,
    prepare_model_restore_plan,
)
from wm.infra.checkpoint.plan import model_prefix as _model_prefix
from wm.infra.checkpoint.scheduler_state import (
    LAMBDA_LR_ADAPTER,
    restore_closed_form_scheduler_progress,
    restore_grad_scaler_progress,
    restore_lambda_lr_progress,
    scheduler_adapter_name,
    scheduler_state_for_save,
)
from wm.infra.checkpoint.sidecar import (
    capture_dataloader_snapshot,
    restore_dataloader_snapshot,
    save_checkpoint_metadata,
    save_dataloader_snapshot,
    warn_runtime_metadata_changes,
)
from wm.infra.checkpoint.source import (
    TENSOR_COMPONENTS,
    LoadSource,
    component_fields,
    component_names,
    normalise_components,
)
from wm.infra.distributed import is_rank0
from wm.infra.model import Model
from wm.infra.rng_state import (
    capture_distributed_rng_state,
    isolated_rng,
    restore_distributed_rng_state,
    validate_distributed_rng_state,
)


@dataclass(frozen=True, slots=True)
class LoadResult:
    checkpoint_path: str | None
    iteration: int = 0
    loaded_components: tuple[str, ...] = ()
    missing_components: tuple[str, ...] = ()
    missing_model_keys: tuple[str, ...] = ()
    unexpected_model_keys: tuple[str, ...] = ()
    checkpoint_model_keys: frozenset[str] = frozenset()
    loaded_model_keys: frozenset[str] = frozenset()
    dataloader_restored: bool = False
    rng_restore_pending: bool = False


@dataclass(slots=True)
class _PendingSave:
    future: Future[Any]
    stager: DefaultStager
    checkpoint_name: str
    checkpoint_path: Path
    iteration: int


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    while isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return model


def _match_rng_blob_shapes(
    rng_state: dict[str, Any],
    metadata: Metadata,
) -> None:
    ranks = rng_state.get("ranks")
    if not isinstance(ranks, Mapping):
        return
    for rank_name, rank_state in ranks.items():
        if not isinstance(rank_state, Mapping):
            continue
        for field in ("python_blob", "numpy_blob"):
            key = f"trainer.rng_state.ranks.{rank_name}.{field}"
            entry = metadata.state_dict_metadata.get(key)
            current = rank_state.get(field)
            if entry is None or not torch.is_tensor(current):
                continue
            shape = tuple(int(dimension) for dimension in entry.size)
            rank_state[field] = torch.empty(
                shape,
                dtype=current.dtype,
                device="cpu",
            )


@dataclass(slots=True)
class _ModelLoad:
    model_state: dict[str, Any] | None = None
    overlay_plan: ModelRestorePlan | None = None
    checkpoint_model_keys: frozenset[str] = frozenset()
    missing_model_keys: tuple[str, ...] = ()
    unexpected_model_keys: tuple[str, ...] = ()
    warned_empty_overlay: bool = False


@dataclass(slots=True)
class _OptimizerLoad:
    names: tuple[str, ...] = ()
    states: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    options: dict[str, tuple[dict[str, Any], ...]] = dataclass_field(
        default_factory=dict
    )
    flat_group_options: dict[str, dict[str, Any]] = dataclass_field(
        default_factory=dict
    )
    unsaved: dict[str, UnsavedOptimizerState] = dataclass_field(default_factory=dict)
    has_saved_state: dict[str, bool] = dataclass_field(default_factory=dict)


@dataclass(slots=True)
class _SchedulerLoad:
    names: tuple[str, ...] = ()
    states: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)


@dataclass(slots=True)
class _TrainerLoad:
    state: dict[str, Any] | None = None
    current_scaler_state: dict[str, Any] | None = None


class DistributedCheckpointer:
    def __init__(
        self,
        save_dir: str | Path,
        *,
        async_mode: str = "disabled",
        async_timeout_seconds: float = 3600.0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.latest_path = self.save_dir / "latest"
        self._pending_rng_state: dict[str, Any] | None = None
        if async_mode not in {"disabled", "thread"}:
            raise ValueError("async_mode must be 'disabled' or 'thread'")
        self.async_mode = async_mode
        self.async_timeout_seconds = float(async_timeout_seconds)
        self._pending_save: _PendingSave | None = None
        self._async_process_group = None
        if (
            self.async_mode == "thread"
            and dist.is_available()
            and dist.is_initialized()
        ):
            # DCP's asynchronous writer operates on staged CPU tensors and
            # requires a CPU-capable collective backend. Keep training on its
            # existing NCCL group and dedicate one all-rank Gloo group to IO.
            self._async_process_group = dist.new_group(backend="gloo")

    @staticmethod
    def _uses_cuda_storage(model: torch.nn.Module) -> bool:
        for tensor in chain(model.parameters(), model.buffers()):
            local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            return local.device.type == "cuda"
        return False

    def _new_async_stager(self, model: torch.nn.Module) -> DefaultStager:
        use_cuda = self._uses_cuda_storage(model)
        return DefaultStager(
            StagingOptions(
                use_pinned_memory=use_cuda,
                use_shared_memory=False,
                # Match Cosmos Framework's safe snapshot boundary: D2H is
                # complete before training mutates live weights again; only
                # storage IO runs in the background.
                use_async_staging=False,
                use_non_blocking_copy=use_cuda,
            )
        )

    def _publish_latest(self, checkpoint_name: str) -> None:
        if dist.is_available() and dist.is_initialized():
            publication_error: BaseException | None = None
            if is_rank0():
                try:
                    easy_io.atomic_put_text(f"{checkpoint_name}\n", self.latest_path)
                except BaseException as error:
                    publication_error = error
            status = [
                None
                if publication_error is None
                else f"{type(publication_error).__name__}: {publication_error}"
            ]
            dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                if publication_error is not None:
                    raise publication_error
                raise RuntimeError(f"Rank 0 failed to publish latest: {status[0]}")
        else:
            easy_io.atomic_put_text(f"{checkpoint_name}\n", self.latest_path)

    def _collect_async_failures(
        self,
        local_error: BaseException | None,
    ) -> tuple[tuple[int, str, str], ...]:
        failure = (
            None
            if local_error is None
            else (
                type(local_error).__name__,
                " ".join(str(local_error).split())[:400],
            )
        )
        if not (dist.is_available() and dist.is_initialized()):
            return () if failure is None else ((0, *failure),)
        gathered: list[tuple[str, str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            failure,
            group=self._async_process_group,
        )
        return tuple(
            (rank, failure_type, message)
            for rank, failure in enumerate(gathered)
            if failure is not None
            for failure_type, message in (failure,)
        )

    def _wait_for_pending_save(self, *, coordinate: bool = True) -> None:
        pending = self._pending_save
        if pending is None:
            return

        local_error: BaseException | None = None
        try:
            pending.future.result(timeout=self.async_timeout_seconds)
        except FutureTimeoutError as error:
            local_error = TimeoutError(
                f"asynchronous DCP save exceeded {self.async_timeout_seconds:g} seconds"
            )
            local_error.__cause__ = error
        except BaseException as error:
            local_error = error

        failures = (
            self._collect_async_failures(local_error)
            if coordinate
            else (
                ()
                if local_error is None
                else (
                    (
                        dist.get_rank() if dist.is_initialized() else 0,
                        type(local_error).__name__,
                        str(local_error),
                    ),
                )
            )
        )
        timed_out = any(
            failure_type == "TimeoutError" for _, failure_type, _ in failures
        )
        if not timed_out:
            pending.stager.close()
            self._pending_save = None
        if failures:
            summary = "; ".join(
                f"rank {rank}: {failure_type}: {message}"
                for rank, failure_type, message in failures[:4]
            )
            if len(failures) > 4:
                summary += f"; ... ({len(failures)} ranks failed)"
            raise RuntimeError(
                f"Asynchronous checkpoint {pending.checkpoint_name} failed: {summary}"
            ) from local_error

        if coordinate:
            # Completion runs outside save()'s capture/restore region, possibly
            # after more training. Preserve the current streams around external
            # publication, including its exception paths.
            with isolated_rng():
                self._publish_latest(pending.checkpoint_name)

    def finalize(self) -> None:
        self._wait_for_pending_save()

    def finalize_after_error(self) -> None:
        self._wait_for_pending_save(coordinate=False)

    def initialize_model(
        self,
        model: torch.nn.Module,
        initializations: Mapping[str, Mapping[str, Any]],
        *,
        resume: bool,
    ) -> None:
        for initialization in plan_model_initializations(
            initializations, resume=resume
        ):
            self.load(model, components=("model",), **initialization.load_kwargs())

    def save(
        self,
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
        schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        *,
        iteration: int,
        dataloader: Any | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if self.async_mode == "thread":
            self._wait_for_pending_save()
        model = _unwrap_ddp(model)
        # Storage implementations and loader snapshots are external code;
        # a save must not perturb the trajectory that follows it.
        optimizers = {} if optimizers is None else optimizers
        schedulers = {} if schedulers is None else schedulers
        checkpoint_name = f"iter_{int(iteration):09d}"
        checkpoint_path = self.save_dir / checkpoint_name
        tensor_path = checkpoint_path / "tensor"
        checkpoint_rng_state: dict[str, Any] | None = None

        try:
            # Keep every external capture operation inside the same protected
            # region.  A loader's state_dict/resume_signature may consume a
            # global RNG or raise; neither case should leave the training
            # process at a different RNG position than before save().
            checkpoint_rng_state = capture_distributed_rng_state()
            self.save_dir.mkdir(parents=True, exist_ok=True)
            dataloader_snapshot = (
                capture_dataloader_snapshot(dataloader, runtime_metadata)
                if dataloader is not None
                else None
            )
            model_state = get_model_state_dict(
                model, options=StateDictOptions(strict=False)
            )
            if isinstance(model, Model):
                model_state = model.checkpoint_model_state_dict(model_state)
            state: dict[str, Any] = {
                "model": model_state,
                "trainer": {
                    "iteration": int(iteration),
                    "rng_state": checkpoint_rng_state,
                },
            }
            if optimizers:
                optimizer_states = {
                    name: optimizer_state_for_save(model, optimizer)
                    for name, optimizer in optimizers.items()
                    if optimizer.state
                }
                if optimizer_states:
                    state["optimizers"] = optimizer_states
            if schedulers:
                state["schedulers"] = {
                    name: scheduler_state_for_save(scheduler)
                    for name, scheduler in schedulers.items()
                }
            if grad_scaler is not None:
                state["trainer"]["grad_scaler"] = grad_scaler.state_dict()

            if self.async_mode == "thread":
                # Sidecars are small, rank-local, and may contain arbitrary
                # Python objects. Publish them synchronously from the same
                # captured boundary; the pointer remains unchanged until the
                # background tensor write also succeeds.
                if dataloader_snapshot is not None:
                    save_dataloader_snapshot(checkpoint_path, dataloader_snapshot)
                save_checkpoint_metadata(
                    checkpoint_path,
                    iteration=iteration,
                    runtime_metadata=runtime_metadata,
                )
                stager = self._new_async_stager(model)
                try:
                    response = dcp.async_save(
                        state,
                        # DefaultStager has already produced an immutable CPU
                        # snapshot. Disable FileSystemWriter's GPU copy-ahead
                        # so its background upload never synchronizes the CUDA
                        # stream used by the next training iteration.
                        storage_writer=FileSystemWriter(
                            tensor_path,
                            per_thread_copy_ahead=0,
                        ),
                        planner=DefaultSavePlanner(
                            dedup_save_to_lowest_rank=True,
                            enable_plan_caching=True,
                        ),
                        process_group=self._async_process_group,
                        async_stager=stager,
                        no_dist=not (dist.is_available() and dist.is_initialized()),
                    )
                except BaseException:
                    stager.close()
                    raise
                future = getattr(response, "upload_completion", response)
                self._pending_save = _PendingSave(
                    future=future,
                    stager=stager,
                    checkpoint_name=checkpoint_name,
                    checkpoint_path=checkpoint_path,
                    iteration=int(iteration),
                )
            else:
                dcp.save(
                    state,
                    checkpoint_id=str(tensor_path),
                    planner=DefaultSavePlanner(
                        dedup_save_to_lowest_rank=True,
                        enable_plan_caching=True,
                    ),
                    no_dist=not (dist.is_available() and dist.is_initialized()),
                )
                if dataloader_snapshot is not None:
                    save_dataloader_snapshot(checkpoint_path, dataloader_snapshot)
                save_checkpoint_metadata(
                    checkpoint_path,
                    iteration=iteration,
                    runtime_metadata=runtime_metadata,
                )
                self._publish_latest(checkpoint_name)
            return str(checkpoint_path)

        except Exception:
            rank = dist.get_rank() if dist.is_initialized() else 0
            log.exception(
                f"[rank={rank}] Failed to save checkpoint at iteration={iteration}, path={checkpoint_path}"
            )
            raise

        finally:
            if checkpoint_rng_state is not None:
                restore_distributed_rng_state(checkpoint_rng_state)

    def load(
        self,
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
        schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        *,
        path: str | Path | None = "latest",
        components: Iterable[str] | None = None,
        model_key_prefix: str | None = None,
        model_key_prefixes: Iterable[str] | None = None,
        source_model_key_prefix: str | None = None,
        source_model_key_patterns: Iterable[str] | None = None,
        dataloader: Any | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> LoadResult:
        model = _unwrap_ddp(model)
        source_patterns = tuple(
            re.compile(pattern) for pattern in (source_model_key_patterns or ())
        )
        additional_model_key_prefixes: tuple[str, ...] = ()
        if model_key_prefixes is not None:
            destinations = tuple(str(prefix) for prefix in model_key_prefixes)
            if not destinations:
                raise ValueError("model_key_prefixes must contain a destination")
            if model_key_prefix is not None:
                raise ValueError(
                    "use either model_key_prefix or model_key_prefixes, not both"
                )
            model_key_prefix = destinations[0]
            additional_model_key_prefixes = destinations[1:]
        self.finalize()
        self._pending_rng_state = None
        selected = normalise_components(
            components,
            include_dataloader=dataloader is not None,
        )
        checkpoint_path = self._resolve_path(path)
        if checkpoint_path is None:
            return LoadResult(checkpoint_path=None)
        warn_runtime_metadata_changes(checkpoint_path, runtime_metadata)

        optimizers = {} if optimizers is None else optimizers
        schedulers = {} if schedulers is None else schedulers
        source = LoadSource.open(checkpoint_path, source_model_key_prefix)
        missing_components = tuple(
            component
            for component in selected
            if component in TENSOR_COMPONENTS
            and component not in source.saved_components
        )
        if missing_components:
            warn_rank0(
                "Checkpoint has no selected component(s): "
                + ", ".join(missing_components)
                + "; keeping them fresh."
            )

        # Each selected component contributes templates to one collective DCP
        # read, then applies the loaded values in model, optimizer, scheduler,
        # trainer, dataloader order.
        state: dict[str, Any] = {}
        loaded_components: list[str] = []
        model_load = _ModelLoad()
        if source.has("model", selected):
            model_load = self._prepare_model_load(
                model,
                source,
                state,
                source_prefix=source_model_key_prefix,
                destination_prefix=model_key_prefix,
                additional_destination_prefixes=additional_model_key_prefixes,
                source_patterns=source_patterns,
            )
            if model_load.model_state:
                loaded_components.append("model")
        optimizer_load = _OptimizerLoad()
        if source.has("optimizers", selected):
            optimizer_load = self._prepare_optimizer_load(
                model, optimizers, source, state
            )
            if optimizer_load.states:
                loaded_components.append("optimizers")
        scheduler_load = _SchedulerLoad()
        if source.has("schedulers", selected):
            scheduler_load = self._prepare_scheduler_load(schedulers, source, state)
        trainer_load = _TrainerLoad()
        if source.has("trainer", selected):
            trainer_load = self._prepare_trainer_load(grad_scaler, source, state)

        if state:
            dcp.load(
                state,
                storage_reader=source.reader,
                planner=DefaultLoadPlanner(allow_partial_load=True),
                no_dist=not (dist.is_available() and dist.is_initialized()),
            )

        loaded_model_keys = self._apply_model_load(
            model,
            model_load,
            source,
            state,
        )
        self._apply_optimizer_load(model, optimizers, optimizer_load)
        if self._apply_scheduler_load(schedulers, scheduler_load, source):
            loaded_components.append("schedulers")
        iteration = 0
        if trainer_load.state is not None:
            iteration, restored_iteration = self._apply_trainer_load(
                grad_scaler, trainer_load, source, dataloader
            )
            if restored_iteration:
                loaded_components.append("trainer")

        dataloader_restored = False
        if "dataloader" in selected:
            dataloader_restored = restore_dataloader_snapshot(
                checkpoint_path,
                dataloader,
                runtime_metadata,
            )
            if dataloader_restored:
                loaded_components.append("dataloader")
            else:
                missing_components = (*missing_components, "dataloader")

        return LoadResult(
            checkpoint_path=str(checkpoint_path),
            iteration=iteration,
            loaded_components=tuple(loaded_components),
            missing_components=missing_components,
            missing_model_keys=model_load.missing_model_keys,
            unexpected_model_keys=model_load.unexpected_model_keys,
            checkpoint_model_keys=model_load.checkpoint_model_keys,
            loaded_model_keys=loaded_model_keys,
            dataloader_restored=dataloader_restored,
            rng_restore_pending=self._pending_rng_state is not None,
        )

    @staticmethod
    def _prepare_model_load(
        model: torch.nn.Module,
        source: LoadSource,
        state: dict[str, Any],
        *,
        source_prefix: str | None,
        destination_prefix: str | None,
        additional_destination_prefixes: tuple[str, ...],
        source_patterns: tuple[re.Pattern[str], ...],
    ) -> _ModelLoad:
        overlay_model = (
            source.is_huggingface
            or source_prefix is not None
            or bool(source_patterns)
            or bool(additional_destination_prefixes)
        )
        live_model_state = get_model_state_dict(
            model, options=StateDictOptions(strict=False)
        )
        if not overlay_model and isinstance(model, Model):
            live_model_state = model.checkpoint_model_state_dict(live_model_state)
        plan = prepare_model_restore_plan(
            live_state=live_model_state,
            saved_keys=(
                source.state_keys
                if source.flat_model
                else component_fields(source.state_keys, "model")
            ),
            overlay=overlay_model,
            source_prefix=source_prefix,
            destination_prefixes=(destination_prefix, *additional_destination_prefixes),
            source_patterns=source_patterns,
        )
        load = _ModelLoad(
            model_state={
                entry.template_key: live_model_state[entry.template_key]
                for entry in plan.entries
            },
            checkpoint_model_keys=plan.checkpoint_model_keys,
            missing_model_keys=plan.missing_model_keys,
            unexpected_model_keys=plan.unexpected_model_keys,
        )
        if load.model_state:
            if overlay_model:
                load.overlay_plan = plan
                source_template = {
                    entry.source_key: live_model_state[entry.template_key]
                    for entry in plan.entries
                }
                if source.flat_model:
                    state.update(source_template)
                else:
                    state["model"] = source_template
            else:
                state["model"] = load.model_state
        elif overlay_model:
            source_label = _model_prefix(source_prefix).removesuffix(".") or "<root>"
            destination_label = (
                _model_prefix(destination_prefix).removesuffix(".") or "<root>"
            )
            warn_rank0(
                "Checkpoint model overlay matched no live keys: "
                f"source={source_label}, destination={destination_label}; "
                "keeping the destination fresh "
                f"(missing={len(load.missing_model_keys)}, "
                f"unexpected={len(load.unexpected_model_keys)})."
            )
            load.warned_empty_overlay = True
        return load

    def _apply_model_load(
        self,
        model: torch.nn.Module,
        load: _ModelLoad,
        source: LoadSource,
        state: dict[str, Any],
    ) -> frozenset[str]:
        model_state = load.model_state
        if load.overlay_plan is not None:
            loaded_source_state = state if source.flat_model else state["model"]
            model_state = {
                target: loaded_source_state[entry.source_key]
                for entry in load.overlay_plan.entries
                for target in entry.targets
            }
        loaded_model_keys: frozenset[str] = frozenset()
        if model_state is not None and model_state:
            loaded_model_keys = frozenset(model_state)
            set_model_state_dict(
                model,
                model_state,
                options=StateDictOptions(strict=False),
            )
        if not load.warned_empty_overlay:
            self._warn_model_difference(
                load.missing_model_keys, load.unexpected_model_keys
            )
        return loaded_model_keys

    def _prepare_optimizer_load(
        self,
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
        source: LoadSource,
        state: dict[str, Any],
    ) -> _OptimizerLoad:
        saved_optimizer_names = component_names(source.state_keys, "optimizers")
        load = _OptimizerLoad(
            names=self._warn_named_difference(
                "Optimizer", set(optimizers), saved_optimizer_names
            )
        )
        for name in load.names:
            optimizer = optimizers[name]
            load.options[name] = optimizer_group_options(optimizer)
            saved_optimizer_fields = component_fields(
                source.state_keys, "optimizers", name
            )
            load.has_saved_state[name] = any(
                field.startswith("state.") for field in saved_optimizer_fields
            )
            load.unsaved[name] = snapshot_unsaved_optimizer_state(
                model,
                optimizer,
                saved_optimizer_fields,
            )
            optimizer_state = get_optimizer_state_dict(
                model, optimizer, options=OPTIMIZER_STATE_OPTIONS
            )
            load.states[name] = optimizer_state
            load.flat_group_options[name] = flattened_optimizer_group_options(
                optimizer_state
            )
        if load.states:
            state["optimizers"] = load.states
        return load

    @staticmethod
    def _apply_optimizer_load(
        model: torch.nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
        load: _OptimizerLoad,
    ) -> None:
        for name in load.names:
            optimizer = optimizers[name]
            loaded_flat_options = flattened_optimizer_group_options(load.states[name])
            checkpoint_differed = optimizer_options_differ(
                loaded_flat_options, load.flat_group_options[name]
            )
            # Restore current per-parameter group options before the public
            # setter rebuilds live groups. This matters when the current
            # recipe regroups parameters: the setter chooses one option from
            # each current group while translating the flattened state.
            load.states[name].update(load.flat_group_options[name])
            remove_unsaved_optimizer_template_state(
                optimizer,
                load.states[name],
                load.unsaved[name],
            )
            if load.has_saved_state[name]:
                set_optimizer_state_dict(
                    model,
                    optimizer,
                    load.states[name],
                    options=OPTIMIZER_STATE_OPTIONS,
                )
            restore_optimizer_group_options(
                name,
                optimizer,
                load.options[name],
                checkpoint_differed=checkpoint_differed,
            )
            restore_unsaved_optimizer_state(optimizer, load.unsaved[name])

    def _prepare_scheduler_load(
        self,
        schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler],
        source: LoadSource,
        state: dict[str, Any],
    ) -> _SchedulerLoad:
        saved_scheduler_names = component_names(source.state_keys, "schedulers")
        matching_names = self._warn_named_difference(
            "Scheduler", set(schedulers), saved_scheduler_names
        )
        load = _SchedulerLoad()
        supported_names = []
        for name in matching_names:
            scheduler = schedulers[name]
            saved_scheduler_fields = component_fields(
                source.state_keys, "schedulers", name
            )
            if "adapter" in saved_scheduler_fields:
                load.states[name] = scheduler_state_for_save(scheduler)
                supported_names.append(name)
            else:
                warn_rank0(
                    f"Scheduler '{name}' has no saved adapter identity; "
                    "keeping it fresh."
                )
        load.names = tuple(supported_names)
        if load.states:
            state["schedulers"] = load.states
        return load

    @staticmethod
    def _apply_scheduler_load(
        schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler],
        load: _SchedulerLoad,
        source: LoadSource,
    ) -> bool:
        restored_scheduler = False
        for name in load.names:
            scheduler_payload = load.states[name]
            source_adapter = scheduler_payload["adapter"]
            current_adapter = scheduler_adapter_name(schedulers[name])
            if source_adapter != current_adapter:
                warn_rank0(
                    f"Scheduler '{name}' source adapter "
                    f"'{source_adapter}' is incompatible with "
                    f"'{current_adapter}'; keeping it fresh."
                )
                continue
            saved_scheduler_fields = component_fields(
                source.state_keys, "schedulers", name
            )
            saved_state_fields = {
                field.removeprefix("state.")
                for field in saved_scheduler_fields
                if field.startswith("state.")
            }
            if current_adapter == LAMBDA_LR_ADAPTER:
                restored_scheduler |= restore_lambda_lr_progress(
                    schedulers[name],  # type: ignore[arg-type]
                    scheduler_payload["state"],
                    saved_state_fields,
                )
            else:
                restored_scheduler |= restore_closed_form_scheduler_progress(
                    name,
                    schedulers[name],
                    scheduler_payload["state"],
                    saved_state_fields,
                )
        return restored_scheduler

    @staticmethod
    def _prepare_trainer_load(
        grad_scaler: torch.amp.GradScaler | None,
        source: LoadSource,
        state: dict[str, Any],
    ) -> _TrainerLoad:
        trainer_state: dict[str, Any] = {"iteration": 0}
        current_scaler_state: dict[str, Any] | None = None
        if grad_scaler is not None:
            current_scaler_state = grad_scaler.state_dict()
            if current_scaler_state:
                trainer_state["grad_scaler"] = dict(current_scaler_state)
        saved_rng_fields = {
            key.removeprefix("trainer.")
            for key in source.state_keys
            if key.startswith("trainer.rng_state")
        }
        if saved_rng_fields:
            trainer_state["rng_state"] = capture_distributed_rng_state()
            _match_rng_blob_shapes(trainer_state["rng_state"], source.metadata)
        state["trainer"] = trainer_state
        return _TrainerLoad(
            state=trainer_state, current_scaler_state=current_scaler_state
        )

    def _apply_trainer_load(
        self,
        grad_scaler: torch.amp.GradScaler | None,
        load: _TrainerLoad,
        source: LoadSource,
        dataloader: Any | None,
    ) -> tuple[int, bool]:
        assert load.state is not None
        iteration = 0
        restored_iteration = False
        trainer_fields = component_fields(source.state_keys, "trainer")
        if "iteration" in trainer_fields:
            iteration = int(load.state["iteration"])
            restored_iteration = True
        if grad_scaler is not None and load.current_scaler_state is not None:
            scaler_fields = {
                field.removeprefix("grad_scaler.")
                for field in trainer_fields
                if field.startswith("grad_scaler.")
            }
            loaded_scaler_state = load.state.get("grad_scaler", {})
            restored_scaler = restore_grad_scaler_progress(
                grad_scaler,
                load.current_scaler_state,
                loaded_scaler_state,
                scaler_fields,
            )
            if not restored_scaler and load.current_scaler_state:
                warn_rank0(
                    "Checkpoint has no compatible GradScaler progress; "
                    "keeping the current scaler state."
                )
        loaded_rng_state = load.state.get("rng_state")
        if loaded_rng_state is not None:
            try:
                validate_distributed_rng_state(loaded_rng_state)
                if dataloader is None:
                    # No worker iterator needs to be constructed first, so
                    # the public load() call can complete the process RNG
                    # restore synchronously.
                    restore_distributed_rng_state(loaded_rng_state)
                else:
                    # Stateful loaders restore their cursor when the first
                    # replacement iterator is created.  Trainer calls the
                    # pending hook immediately after that construction.
                    self._pending_rng_state = loaded_rng_state
            except (RuntimeError, ValueError, TypeError) as error:
                warn_rank0(
                    "Skipping exact RNG restore; continuing with current RNG "
                    f"streams ({type(error).__name__}: {error})."
                )
        return iteration, restored_iteration

    def restore_pending_rng_state(self) -> bool:
        if self._pending_rng_state is None:
            return False
        state = self._pending_rng_state
        self._pending_rng_state = None
        restore_distributed_rng_state(state)
        return True

    def _resolve_path(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None
        if str(path) != "latest":
            return Path(path)
        if not easy_io.exists(self.latest_path):
            raise FileNotFoundError(
                f"No latest checkpoint pointer at {self.latest_path}"
            )
        target = easy_io.get_text(self.latest_path).strip()
        if not target:
            raise FileNotFoundError(
                f"Latest checkpoint pointer is empty: {self.latest_path}"
            )
        target_path = Path(target)
        return target_path if target_path.is_absolute() else self.save_dir / target_path

    @staticmethod
    def _warn_model_difference(
        missing: tuple[str, ...], unexpected: tuple[str, ...]
    ) -> None:
        if not missing and not unexpected:
            return

        def summary(keys: tuple[str, ...]) -> str:
            examples = ", ".join(keys[:3])
            suffix = ", ..." if len(keys) > 3 else ""
            return f"{len(keys)} [{examples}{suffix}]"

        warn_rank0(
            "Non-strict model resume: "
            f"missing={summary(missing)}, unexpected={summary(unexpected)}."
        )

    @staticmethod
    def _warn_named_difference(
        label: str, current: set[str], saved: set[str]
    ) -> tuple[str, ...]:
        fresh = sorted(current - saved)
        ignored = sorted(saved - current)
        if fresh or ignored:

            def summary(names: list[str]) -> str:
                examples = ", ".join(names[:3])
                suffix = ", ..." if len(names) > 3 else ""
                return f"{len(names)} [{examples}{suffix}]"

            parts = []
            if fresh:
                parts.append(f"fresh={summary(fresh)}")
            if ignored:
                parts.append(f"ignored={summary(ignored)}")
            warn_rank0(f"{label} names differ: {', '.join(parts)}.")
        return tuple(sorted(current.intersection(saved)))
