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

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, ParamSpec, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

from wm.infra import log

_P = ParamSpec("_P")
_R = TypeVar("_R")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", get_local_rank())
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", get_local_rank())
    return resolved


def init(
    *,
    device: torch.device | str | None = None,
    backend: str | None = None,
    timeout: timedelta = timedelta(minutes=30),
) -> torch.device:
    resolved_device = _resolve_device(device)
    if dist.is_available() and dist.is_initialized():
        return resolved_device

    if resolved_device.type == "cuda":
        # Device binding must precede process-group creation and any model
        # allocation so each local rank does not accidentally use cuda:0.
        torch.cuda.set_device(resolved_device)

    if not dist.is_available():
        return resolved_device

    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not launched:
        return resolved_device

    selected_backend = backend
    if selected_backend is None:
        selected_backend = "nccl" if resolved_device.type == "cuda" else "gloo"
    dist.init_process_group(
        backend=selected_backend,
        init_method="env://",
        timeout=timeout,
    )
    log.info(
        "Initialized "
        f"{selected_backend} process group: rank={dist.get_rank()} "
        f"world_size={dist.get_world_size()} local_rank={get_local_rank()}",
        rank0_only=False,
    )
    return resolved_device


def destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank(group: dist.ProcessGroup | None = None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


def get_world_size(group: dist.ProcessGroup | None = None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group)
    return 1


def get_local_world_size() -> int:
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is not None:
        return int(local_world_size)
    return get_world_size()


def is_rank0() -> bool:
    return get_rank() == 0


class _ClosureExecutionRoot(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, closure: Callable[[], _R]) -> _R:
        return closure()


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        execution_root = _ClosureExecutionRoot(module)
        super().__init__(execution_root, *args, **kwargs)
        object.__setattr__(self, "training_model", module)
        self.show_sync_grad_static_graph_warning = True

    def forward_closure(self, closure: Callable[[], _R]) -> _R:
        # Passing the callable as a positional argument also keeps CUDA DDP's
        # input-device path valid without a meaningless sentinel.
        return self(closure)


def wrap_ddp(
    model: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
    process_group: dist.ProcessGroup | None = None,
    device_ids: Sequence[int] | None = None,
    output_device: int | None = None,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
    broadcast_buffers: bool = True,
    gradient_as_bucket_view: bool = True,
) -> torch.nn.Module:
    if not (dist.is_available() and dist.is_initialized()):
        return model
    if device is not None and device_ids is None:
        resolved_device = _resolve_device(device)
        if resolved_device.type == "cuda":
            device_index = resolved_device.index
            if device_index is None:
                device_index = get_local_rank()
            device_ids = (device_index,)
            if output_device is None:
                output_device = device_index
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        process_group=process_group,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        broadcast_buffers=broadcast_buffers,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def _outermost_modules(
    model: torch.nn.Module, module_type: type[torch.nn.Module]
) -> tuple[torch.nn.Module, ...]:
    roots: list[torch.nn.Module] = []

    def visit(module: torch.nn.Module) -> None:
        if isinstance(module, module_type):
            roots.append(module)
            return
        for child in module.children():
            visit(child)

    visit(model)
    return tuple(roots)


def _fsdp_roots(model: torch.nn.Module) -> tuple[FSDPModule, ...]:
    return tuple(_outermost_modules(model, FSDPModule))  # type: ignore[return-value]


def _uses_native_hsdp(root: FSDPModule) -> bool:
    for parameter in root.parameters():
        mesh = getattr(parameter, "device_mesh", None)
        if mesh is not None:
            return int(getattr(mesh, "ndim", 0)) == 2

    # This fallback is useful for lightweight wrappers and tests whose FSDP
    # root owns no parameters. Production roots normally take the DTensor path.
    mesh = getattr(root, "fsdp_device_mesh", None)
    return int(getattr(mesh, "ndim", 0)) == 2


@contextmanager
def ddp_sync_grad(model: torch.nn.Module, enabled: bool) -> Iterator[None]:
    if isinstance(model, DistributedDataParallel):
        previous = model.require_backward_grad_sync
        if model.static_graph and previous != enabled:
            if model.show_sync_grad_static_graph_warning:
                if is_rank0():
                    log.warning(
                        "DDP static_graph=True does not support changing gradient "
                        "synchronization; accumulation remains synchronized."
                    )
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = bool(enabled)
        try:
            yield
        finally:
            model.require_backward_grad_sync = previous
        return

    hsdp_roots = tuple(root for root in _fsdp_roots(model) if _uses_native_hsdp(root))
    try:
        for root in hsdp_roots:
            root.set_requires_all_reduce(bool(enabled), recurse=True)
        yield
    finally:
        # Trainer accumulation contexts are not nested. Restoring the public
        # default on exceptions prevents later optimizer steps becoming local.
        for root in reversed(hsdp_roots):
            root.set_requires_all_reduce(True, recurse=True)


__all__ = [
    "DistributedDataParallel",
    "ddp_sync_grad",
    "destroy_process_group",
    "get_local_rank",
    "get_local_world_size",
    "get_rank",
    "get_world_size",
    "init",
    "is_rank0",
    "wrap_ddp",
]
