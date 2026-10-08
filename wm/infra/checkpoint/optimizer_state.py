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

import copy
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _get_fqns,
    get_optimizer_state_dict,
)

from wm.infra.checkpoint.common import warn_rank0

OPTIMIZER_STATE_OPTIONS = StateDictOptions(flatten_optimizer_state_dict=True)


def option_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return bool(torch.equal(left, right))
    try:
        result = left == right
        return (
            bool(result) if not isinstance(result, torch.Tensor) else bool(result.all())
        )
    except (TypeError, ValueError):
        return left is right


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    return copy.deepcopy(value)


def optimizer_group_options(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {key: _snapshot_value(value) for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    )


def flattened_optimizer_group_options(
    optimizer_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: _snapshot_value(value)
        for key, value in optimizer_state.items()
        if key.startswith("param_groups.")
    }


def _canonical_parameter_names(
    model: torch.nn.Module,
) -> dict[torch.nn.Parameter, tuple[str, ...]]:
    return {
        parameter: tuple(sorted(_get_fqns(model, name)))
        for name, parameter in model.named_parameters()
    }


def _optimizer_parameter_candidates(field: str) -> Iterator[str]:
    if not field.startswith("state."):
        return
    name = field.removeprefix("state.")
    while True:
        name, separator, _slot = name.rpartition(".")
        if not separator:
            return
        yield name


def optimizer_state_parameter_names(
    fields: Iterable[str], parameter_names: set[str]
) -> set[str]:
    return {
        name
        for field in fields
        for name in _optimizer_parameter_candidates(field)
        if name in parameter_names
    }


def remove_optimizer_parameter_states(
    state: dict[str, Any], parameter_names: set[str]
) -> None:
    if not parameter_names:
        return
    for field in tuple(state):
        if any(
            name in parameter_names for name in _optimizer_parameter_candidates(field)
        ):
            del state[field]


def optimizer_state_for_save(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    present_before = set(optimizer.state)
    optimizer_state = get_optimizer_state_dict(
        model, optimizer, options=OPTIMIZER_STATE_OPTIONS
    )
    created = set(optimizer.state).difference(present_before)
    if created:
        parameter_names = _canonical_parameter_names(model)
        created_names = {
            name for parameter in created for name in parameter_names.get(parameter, ())
        }
        remove_optimizer_parameter_states(optimizer_state, created_names)
        for parameter in created:
            optimizer.state.pop(parameter, None)
    return optimizer_state


def optimizer_options_differ(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return True

    for key in left:
        prefix, separator, field = key.rpartition(".")
        initial_lr_key = f"{prefix}.initial_lr"
        if (
            separator
            and field == "lr"
            and initial_lr_key in left
            and initial_lr_key in right
        ):
            # LRScheduler mutates ``lr`` as runtime progress while preserving
            # the configured base recipe in ``initial_lr``. The latter remains
            # part of this comparison, so a real base-LR change still warns.
            continue
        if not option_equal(left[key], right[key]):
            return True
    return False


@dataclass(slots=True)
class UnsavedOptimizerState:
    present: dict[torch.nn.Parameter, dict[str, Any]]
    absent: tuple[torch.nn.Parameter, ...]
    names: dict[torch.nn.Parameter, tuple[str, ...]]


def snapshot_unsaved_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    saved_fields: set[str],
) -> UnsavedOptimizerState:
    parameter_names = _canonical_parameter_names(model)
    saved_parameter_names = optimizer_state_parameter_names(
        saved_fields,
        {name for names in parameter_names.values() for name in names},
    )
    present: dict[torch.nn.Parameter, dict[str, Any]] = {}
    absent: list[torch.nn.Parameter] = []
    unsaved_names: dict[torch.nn.Parameter, tuple[str, ...]] = {}
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            names = parameter_names.get(parameter)
            if names is None or any(name in saved_parameter_names for name in names):
                continue
            unsaved_names[parameter] = names
            if parameter in optimizer.state:
                present[parameter] = _snapshot_value(optimizer.state[parameter])
            else:
                absent.append(parameter)
    return UnsavedOptimizerState(
        present=present,
        absent=tuple(absent),
        names=unsaved_names,
    )


def remove_unsaved_optimizer_template_state(
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
    snapshot: UnsavedOptimizerState,
) -> None:
    remove_optimizer_parameter_states(
        optimizer_state,
        {name for names in snapshot.names.values() for name in names},
    )
    for parameter in snapshot.names:
        optimizer.state.pop(parameter, None)


def restore_unsaved_optimizer_state(
    optimizer: torch.optim.Optimizer,
    snapshot: UnsavedOptimizerState,
) -> None:
    for parameter in snapshot.absent:
        optimizer.state.pop(parameter, None)
    for parameter, state in snapshot.present.items():
        optimizer.state[parameter] = state


def restore_optimizer_group_options(
    name: str,
    optimizer: torch.optim.Optimizer,
    current_options: tuple[dict[str, Any], ...],
    *,
    checkpoint_differed: bool,
) -> None:
    loaded_options = optimizer_group_options(optimizer)
    changed = (
        checkpoint_differed
        or len(loaded_options) != len(current_options)
        or any(
            set(loaded) != set(current)
            or any(not option_equal(loaded[key], current[key]) for key in current)
            for loaded, current in zip(loaded_options, current_options, strict=False)
        )
    )

    # ``set_optimizer_state_dict`` normally preserves the live group layout.
    # If a custom optimizer changes its group count, leave its original DCP
    # error/behavior intact instead of inventing a remapping here.
    for group, options in zip(optimizer.param_groups, current_options, strict=False):
        for key in tuple(group):
            if key != "params":
                del group[key]
        group.update(options)

    if changed:
        warn_rank0(
            f"Optimizer '{name}' resumed its state while keeping current "
            "parameter-group options."
        )
