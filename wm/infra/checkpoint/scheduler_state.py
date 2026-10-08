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

from collections.abc import Mapping
from typing import Any

import torch

from wm.infra.checkpoint.common import warn_rank0
from wm.infra.checkpoint.optimizer_state import option_equal

LAMBDA_LR_ADAPTER = "lambda_lr_v1"


_LR_SCHEDULER_ADAPTER_PREFIX = "lr_scheduler_v1:"


def restore_lambda_lr_progress(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loaded_state: Mapping[str, Any],
    saved_fields: set[str],
) -> bool:
    if "last_epoch" not in saved_fields:
        return False

    last_epoch = int(loaded_state["last_epoch"])
    scheduler.last_epoch = last_epoch
    if "_step_count" in saved_fields:
        scheduler._step_count = int(loaded_state["_step_count"])

    # Re-evaluate the loaded progress using this run's base LR and lambda
    # recipe. In particular, never accept checkpoint ``base_lrs`` or the
    # serialized state of a callable lambda object.
    current_lrs = [
        base_lr * lr_lambda(last_epoch)
        for base_lr, lr_lambda in zip(
            scheduler.base_lrs, scheduler.lr_lambdas, strict=True
        )
    ]
    for parameter_group, learning_rate in zip(
        scheduler.optimizer.param_groups, current_lrs, strict=True
    ):
        parameter_group["lr"] = learning_rate
    scheduler._last_lr = current_lrs
    return True


def scheduler_adapter_name(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> str:
    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        return LAMBDA_LR_ADAPTER
    scheduler_type = type(scheduler)
    name = f"{scheduler_type.__module__}.{scheduler_type.__qualname__}"
    return f"{_LR_SCHEDULER_ADAPTER_PREFIX}{name}"


def restore_closed_form_scheduler_progress(
    name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loaded_state: Mapping[str, Any],
    saved_fields: set[str],
) -> bool:
    if "last_epoch" not in saved_fields:
        return False
    closed_form = getattr(scheduler, "_get_closed_form_lr", None)
    if not callable(closed_form):
        warn_rank0(
            f"Scheduler '{name}' ({type(scheduler).__name__}) cannot recompute "
            "learning rates from restored progress; keeping it fresh."
        )
        return False

    current_state = scheduler.state_dict()
    progress_fields = {
        "last_epoch",
        "_step_count",
        "_last_lr",
        "_get_lr_called_within_step",
    }
    comparable_fields = (set(loaded_state) & set(current_state)) - progress_fields
    if any(
        not option_equal(loaded_state[field], current_state[field])
        for field in comparable_fields
    ):
        warn_rank0(
            f"Scheduler '{name}' resumed progress while keeping current recipe options."
        )

    scheduler.last_epoch = int(loaded_state["last_epoch"])
    if "_step_count" in saved_fields:
        scheduler._step_count = int(loaded_state["_step_count"])
    current_lrs = list(closed_form())
    for parameter_group, learning_rate in zip(
        scheduler.optimizer.param_groups,
        current_lrs,
        strict=True,
    ):
        parameter_group["lr"] = learning_rate
    scheduler._last_lr = current_lrs
    return True


def scheduler_state_for_save(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    return {
        "adapter": scheduler_adapter_name(scheduler),
        "state": scheduler.state_dict(),
    }


def restore_grad_scaler_progress(
    grad_scaler: torch.amp.GradScaler,
    current_state: Mapping[str, Any],
    loaded_state: Mapping[str, Any],
    saved_fields: set[str],
) -> bool:
    dynamic_fields = ("scale", "_growth_tracker")
    available = [field for field in dynamic_fields if field in saved_fields]
    if available and not current_state:
        warn_rank0(
            "Checkpoint has enabled GradScaler progress while the current "
            "GradScaler is disabled; keeping it disabled."
        )
        return False
    if not available or not current_state:
        return False

    recipe_fields = ("growth_factor", "backoff_factor", "growth_interval")
    recipe_changed = any(
        field in saved_fields
        and not option_equal(loaded_state[field], current_state[field])
        for field in recipe_fields
    )

    merged_state = dict(current_state)
    for field in available:
        merged_state[field] = loaded_state[field]
    grad_scaler.load_state_dict(merged_state)
    if recipe_changed:
        warn_rank0(
            "GradScaler resumed scale progress while keeping current "
            "growth/backoff settings."
        )
    return True
