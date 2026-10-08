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
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from wm.infra.config import LazyDict, instantiate

StepOutput: TypeAlias = tuple[dict[str, Any], torch.Tensor]
TrainingClosure: TypeAlias = Callable[[], StepOutput]
TrainingClosureSpec: TypeAlias = tuple[str, TrainingClosure, bool]


@runtime_checkable
class LifecycleCadence(Protocol):
    def is_cadence_due(self, iteration: int, interval: int | None) -> bool: ...


@dataclass(frozen=True, slots=True)
class OptimizerBinding:
    name: str
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


class ContextParallelContext:
    def __init__(self, mesh: DeviceMesh | None = None) -> None:
        self.mesh = mesh
        self.group = None
        self.size = 1
        self.rank = 0
        self.leader_global_rank = 0
        self.artifact_rank = 0
        self.groups: tuple[tuple[int, ...], ...] = ()

        if mesh is None or mesh.size() <= 1:
            if dist.is_available() and dist.is_initialized():
                global_rank = dist.get_rank()
                self.leader_global_rank = global_rank
                self.artifact_rank = global_rank
            return

        self.group = mesh.get_group()
        self.size = int(mesh.size())
        self.rank = int(mesh.get_local_rank())
        group_ranks = tuple(dist.get_process_group_ranks(self.group))
        self.leader_global_rank = group_ranks[0]

        gathered_groups: list[tuple[int, ...] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_groups, group_ranks)
        self.groups = tuple(
            sorted({group for group in gathered_groups if group is not None})
        )
        self.artifact_rank = self.groups.index(group_ranks)

    @property
    def enabled(self) -> bool:
        return self.size > 1

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def global_rank(self, local_rank: int) -> int:
        if not self.enabled:
            return self.leader_global_rank
        return dist.get_global_rank(self.group, int(local_rank))

    def broadcast_objects(
        self,
        values: list[Any],
        *,
        source_rank: int,
    ) -> None:
        if self.enabled:
            dist.broadcast_object_list(
                values,
                src=self.global_rank(source_rank),
                group=self.group,
            )

    def checkpoint_topology(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "size": self.size,
            "groups": [list(group) for group in self.groups],
        }


class Model(torch.nn.Module, ABC):
    def __init__(
        self,
        optimizer_parameter_groups: Sequence[Mapping[str, Any]] | None = None,
        optimizer_split_decay_groups: bool = False,
    ) -> None:
        super().__init__()
        self.optimizer_parameter_groups = optimizer_parameter_groups
        self.optimizer_split_decay_groups = bool(optimizer_split_decay_groups)
        self.optimizer_dict: dict[str, torch.optim.Optimizer] = {}
        self.scheduler_dict: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self._context_parallel_context = ContextParallelContext()

    @property
    def context_parallel_context(self) -> ContextParallelContext:
        return self._context_parallel_context

    def register_context_parallel_mesh(
        self,
        mesh: DeviceMesh | None,
        *role_meshes: DeviceMesh | None,
    ) -> None:
        def signature(value: DeviceMesh | None) -> tuple[int, ...]:
            if value is None or value.size() <= 1:
                if dist.is_available() and dist.is_initialized():
                    return (dist.get_rank(),)
                return (0,)
            return tuple(dist.get_process_group_ranks(value.get_group()))

        expected = signature(mesh)
        if any(signature(role_mesh) != expected for role_mesh in role_meshes):
            raise ValueError(
                "all Networks in one training objective must use the same "
                "context-parallel topology"
            )
        self._context_parallel_context = ContextParallelContext(mesh)

    @staticmethod
    def _group_module_trainable_parameters(
        module: torch.nn.Module,
        declarations: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        trainable = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        if not declarations:
            return [{"params": [parameter for _, parameter in trainable]}]

        names_by_parameter: dict[int, list[str]] = {
            id(parameter): [] for _, parameter in trainable
        }
        for name, parameter in module.named_parameters(remove_duplicate=False):
            if parameter.requires_grad and id(parameter) in names_by_parameter:
                names_by_parameter[id(parameter)].append(name)

        assigned: set[int] = set()
        override_groups: list[dict[str, Any]] = []
        for declaration in declarations:
            patterns = declaration["patterns"]
            if isinstance(patterns, str):
                patterns = (patterns,)
            compiled_patterns = tuple(re.compile(pattern) for pattern in patterns)
            parameters = [
                parameter
                for _, parameter in trainable
                if id(parameter) not in assigned
                and any(
                    pattern.fullmatch(alias)
                    for pattern in compiled_patterns
                    for alias in names_by_parameter[id(parameter)]
                )
            ]
            if not parameters:
                continue
            assigned.update(id(parameter) for parameter in parameters)
            group = {
                key: value for key, value in declaration.items() if key != "patterns"
            }
            group["params"] = parameters
            override_groups.append(group)

        default_parameters = [
            parameter for _, parameter in trainable if id(parameter) not in assigned
        ]
        groups: list[dict[str, Any]] = []
        if default_parameters:
            groups.append({"params": default_parameters})
        groups.extend(override_groups)
        return groups

    def _group_trainable_parameters(self) -> list[dict[str, Any]]:
        groups = self._group_module_trainable_parameters(
            self,
            self.optimizer_parameter_groups,
        )
        if not self.optimizer_split_decay_groups or self.optimizer_parameter_groups:
            return groups
        split: list[dict[str, Any]] = []
        for group in groups:
            parameters = list(group["params"])
            decay = [parameter for parameter in parameters if parameter.ndim >= 2]
            no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
            for selected in (decay, no_decay):
                if selected:
                    split.append(
                        {key: value for key, value in group.items() if key != "params"}
                        | {"params": selected}
                    )
        return split

    def configure_trainable_parameters(
        self,
        patterns: str | Sequence[str] | None,
        *,
        module: torch.nn.Module | None = None,
    ) -> None:
        if patterns is None:
            return

        if isinstance(patterns, str):
            patterns = (patterns,)
        compiled_patterns = tuple(re.compile(pattern) for pattern in patterns)
        target = self if module is None else module
        target.requires_grad_(False)
        # One physical parameter may have several names.  Treat those aliases
        # as an OR: matching any full name makes the shared parameter live.
        for name, parameter in target.named_parameters(remove_duplicate=False):
            if any(pattern.fullmatch(name) for pattern in compiled_patterns):
                parameter.requires_grad_(True)

    def init_optimizer_scheduler(
        self,
        optimizer_config: LazyDict[torch.optim.Optimizer],
        scheduler_config: LazyDict[torch.optim.lr_scheduler.LRScheduler],
    ) -> tuple[
        dict[str, torch.optim.Optimizer],
        dict[str, torch.optim.lr_scheduler.LRScheduler],
    ]:
        optimizer = instantiate(
            optimizer_config,
            params=self._group_trainable_parameters(),
        )
        scheduler = instantiate(scheduler_config, optimizer=optimizer)

        self.optimizer_dict = {"net": optimizer}
        self.scheduler_dict = {"net": scheduler}
        return self.optimizer_dict, self.scheduler_dict

    def get_optimizer_names(self, iteration: int) -> tuple[str, ...]:
        del iteration
        return ("net",)

    def get_optimizer_bindings(self, iteration: int) -> tuple[OptimizerBinding, ...]:
        names = tuple(self.get_optimizer_names(iteration))
        if not names:
            raise RuntimeError("Model selected no optimizer for this iteration")
        if len(set(names)) != len(names):
            raise RuntimeError(f"Optimizer roles are selected more than once: {names}")
        bindings = []
        for name in names:
            optimizer = self.optimizer_dict.get(name)
            scheduler = self.scheduler_dict.get(name)
            if optimizer is None or scheduler is None:
                raise RuntimeError(
                    f"Optimizer role {name!r} is not registered with a scheduler"
                )
            if getattr(scheduler, "optimizer", None) is not optimizer:
                raise RuntimeError(
                    f"Scheduler for role {name!r} owns a different optimizer"
                )
            bindings.append(OptimizerBinding(name, optimizer, scheduler))

        # Multiple active roles form one update boundary. Sharing a Parameter
        # would unscale and step it twice; inactive roles may still share it.
        if len(bindings) > 1:
            parameter_owners: dict[int, str] = {}
            for binding in bindings:
                for group in binding.optimizer.param_groups:
                    for parameter in group["params"]:
                        previous = parameter_owners.setdefault(
                            id(parameter), binding.name
                        )
                        if previous != binding.name:
                            raise RuntimeError(
                                f"Active optimizer roles {previous!r} and {binding.name!r} "
                                "share a parameter; active roles must own disjoint parameters"
                            )
        return tuple(bindings)

    def is_cadence_due(self, iteration: int, interval: int | None) -> bool:
        return (
            interval is not None
            and int(interval) > 0
            and int(iteration) > 0
            and int(iteration) % int(interval) == 0
        )

    def parallelize(self, device: torch.device | str) -> None:
        self.to(device)

    def prepare_data_batch(self, data_batch: dict[str, Any]) -> dict[str, Any]:
        """Encode a raw batch after it reached the training device."""
        return data_batch

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        raise NotImplementedError

    def training_step_closures(
        self, data_batch: dict[str, Any], iteration: int
    ) -> Iterator[TrainingClosureSpec]:
        yield "train", lambda: self.training_step(data_batch, iteration), True

    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def sample(self, data_batch: dict[str, Any], **sampling_options: Any) -> Any:
        raise NotImplementedError

    def checkpoint_model_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        return state_dict


__all__ = [
    "ContextParallelContext",
    "LifecycleCadence",
    "Model",
    "StepOutput",
    "TrainingClosure",
    "TrainingClosureSpec",
]
