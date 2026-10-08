# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm.infra.distributed import get_rank

__all__ = [
    "frozen_parameters",
    "map_tensors",
    "prefetch",
    "preserve_training_mode",
    "set_random_seed",
    "to",
]


def map_tensors(value: Any, transform: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    tensors: dict[int, torch.Tensor] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            identity = id(item)
            if identity not in tensors:
                tensors[identity] = transform(item)
            return tensors[identity]
        if isinstance(item, Mapping):
            mapped = {key: visit(child) for key, child in item.items()}
            if isinstance(item, defaultdict):
                return type(item)(item.default_factory, mapped)
            return type(item)(mapped)
        if isinstance(item, tuple) and hasattr(item, "_fields"):
            return type(item)(*(visit(child) for child in item))
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, range)
        ):
            return type(item)([visit(child) for child in item])
        return item

    return visit(value)


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    def move(tensor: torch.Tensor) -> torch.Tensor:
        tensor_memory_format = memory_format
        if (memory_format == torch.channels_last and tensor.dim() != 4) or (
            memory_format == torch.channels_last_3d and tensor.dim() != 5
        ):
            tensor_memory_format = torch.preserve_format
        target_is_cpu = device is not None and torch.device(device).type == "cpu"
        return tensor.to(
            device=device,
            dtype=dtype,
            memory_format=tensor_memory_format,
            non_blocking=not target_is_cpu,
        )

    return map_tensors(data, move)


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    if by_rank:
        seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def preserve_training_mode(model: torch.nn.Module) -> Iterator[None]:
    states = tuple((module, module.training) for module in model.modules())
    try:
        yield
    finally:
        try:
            model.train(states[0][1])
        finally:
            for module, training in states:
                module.training = training


@contextmanager
def frozen_parameters(module: torch.nn.Module) -> Iterator[None]:
    """Disable gradients of ``module`` inside the block, then restore each flag.

    Unlike ``requires_grad_(False)`` / ``requires_grad_(True)``, parameters that
    were already frozen (for example by ``trainable_patterns``) stay frozen.
    """
    parameters = tuple(module.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)


def prefetch(path: str | Path, *, threads: int = 16, chunk: int = 64 << 20) -> None:
    """Pull a file into the page cache with parallel positional reads.

    A network filesystem serves these many times faster than the single
    stream a deserializer issues, so call this before loading a checkpoint.
    """
    size = os.path.getsize(path)
    descriptor = os.open(path, os.O_RDONLY)

    def read(start: int) -> None:
        scratch = memoryview(bytearray(min(chunk, size - start)))
        done = 0
        while done < len(scratch):
            count = os.preadv(descriptor, [scratch[done:]], start + done)
            if count <= 0:
                break
            done += count

    try:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(read, range(0, size, chunk)))
    finally:
        os.close(descriptor)
