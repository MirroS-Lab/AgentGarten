# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .base import Callback


def _validate_max_norm(max_norm: float) -> float:
    value = float(max_norm)
    if math.isnan(value) or value < 0.0:
        raise ValueError("max_norm must be non-negative and not NaN")
    return value


@torch.no_grad()
def clip_grad_norm_groups_(
    parameter_groups: Mapping[str, Iterable[torch.Tensor]],
    max_norm: float,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    max_norm = _validate_max_norm(max_norm)
    parameters_by_role_and_mesh: dict[
        tuple[str, DeviceMesh | None], list[torch.Tensor]
    ] = defaultdict(list)
    parameters_by_mesh: dict[DeviceMesh | None, list[torch.Tensor]] = defaultdict(list)
    seen: set[int] = set()
    for raw_role, parameters in parameter_groups.items():
        role = str(raw_role)
        for parameter in parameters:
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            if parameter.requires_grad and parameter.grad is not None:
                mesh = parameter.device_mesh if isinstance(parameter, DTensor) else None
                parameters_by_role_and_mesh[(role, mesh)].append(parameter)
                parameters_by_mesh[mesh].append(parameter)

    if not parameters_by_mesh:
        return None, {}

    mesh_norms_by_role: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (role, _mesh), mesh_parameters in parameters_by_role_and_mesh.items():
        mesh_norm = torch.nn.utils.get_total_norm(
            [parameter.grad for parameter in mesh_parameters],
            norm_type=2.0,
            error_if_nonfinite=False,
        )
        if isinstance(mesh_norm, DTensor):
            mesh_norm = mesh_norm.full_tensor()
        mesh_norms_by_role[role].append(mesh_norm.reshape(()))

    role_norms: dict[str, torch.Tensor] = {}
    norm_device = next(iter(mesh_norms_by_role.values()))[0].device
    for role, mesh_norms in mesh_norms_by_role.items():
        role_norms[role] = torch.linalg.vector_norm(
            torch.stack([norm.to(device=norm_device) for norm in mesh_norms]),
            ord=2.0,
        )

    total_norm = torch.linalg.vector_norm(
        torch.stack(tuple(role_norms.values())),
        ord=2.0,
    )
    # An unlimited norm is observation-only, including when the norm itself
    # is non-finite. Multiplying by inf / inf would otherwise create NaNs.
    if math.isinf(max_norm):
        return total_norm, role_norms
    for mesh_parameters in parameters_by_mesh.values():
        torch.nn.utils.clip_grads_with_norm_(
            mesh_parameters,
            max_norm,
            total_norm,
        )
    return total_norm, role_norms


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
) -> torch.Tensor | None:
    total_norm, _role_norms = clip_grad_norm_groups_(
        {"model": parameters},
        max_norm,
    )
    return total_norm


class GradClip(Callback):
    def __init__(self, max_norm: float = 1.0) -> None:
        super().__init__()
        self.max_norm = _validate_max_norm(max_norm)

    def on_before_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler, grad_scaler, iteration
        clip_grad_norm_(model_ddp.parameters(), self.max_norm)


__all__ = ["GradClip", "clip_grad_norm_", "clip_grad_norm_groups_"]
