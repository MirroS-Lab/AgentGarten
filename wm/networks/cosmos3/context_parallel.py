# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from .attention import TextContext

__all__ = ["ContextParallel", "ContextParallelAttention"]


def _all_to_all(
    local: torch.Tensor, *, scatter: int, gather: int, mesh: DeviceMesh
) -> torch.Tensor:
    distributed = DTensor.from_local(local, mesh, (Shard(gather),), run_check=False)
    return distributed.redistribute(mesh, (Shard(scatter),)).to_local()


def _pad(value: torch.Tensor, tokens: int) -> torch.Tensor:
    if value.shape[1] == tokens:
        return value
    return torch.nn.functional.pad(
        value, (0, 0) * (value.ndim - 2) + (0, tokens - value.shape[1])
    )


def _repeat_heads(value: torch.Tensor, repeats: int) -> torch.Tensor:
    return value if repeats == 1 else value.repeat_interleave(repeats, dim=2)


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: int) -> torch.Tensor:
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.scale, None


@dataclass(frozen=True)
class ContextParallel:
    mesh: DeviceMesh
    num_attention_heads: int
    num_key_value_heads: int

    def __post_init__(self) -> None:
        size = self.size
        if self.num_attention_heads % size:
            raise ValueError(
                f"{self.num_attention_heads} query heads do not split over CP {size}"
            )
        if (self.num_key_value_heads * self.kv_repeats) % size:
            raise ValueError(
                f"{self.num_key_value_heads} KV heads do not split over CP {size}"
            )

    @property
    def size(self) -> int:
        return self.mesh.size()

    @property
    def rank(self) -> int:
        return dist.get_rank(self.mesh.get_group())

    @property
    def kv_repeats(self) -> int:
        return max(self.size // self.num_key_value_heads, 1)

    def shard(self, value: torch.Tensor) -> torch.Tensor:
        local = math.ceil(value.shape[1] / self.size)
        return _pad(value, local * self.size).narrow(1, self.rank * local, local)

    def gather(self, value: torch.Tensor, tokens: int) -> torch.Tensor:
        full = DTensor.from_local(value, self.mesh, (Shard(1),), run_check=False)
        full = full.redistribute(self.mesh, (Replicate(),)).to_local()
        return _ScaleGradient.apply(full, self.size)[:, :tokens]

    def to_heads(self, value: torch.Tensor, tokens: int, *, kv: bool) -> torch.Tensor:
        if kv:
            value = _repeat_heads(value, self.kv_repeats)
        return _all_to_all(value, scatter=2, gather=1, mesh=self.mesh)[:, :tokens]

    def to_tokens(self, value: torch.Tensor, local_tokens: int) -> torch.Tensor:
        value = _pad(value, local_tokens * self.size)
        return _all_to_all(value, scatter=1, gather=2, mesh=self.mesh)

    def text(self, text: TextContext) -> TextContext:
        heads = self.num_key_value_heads * self.kv_repeats // self.size
        start = self.rank * heads
        return replace(
            text,
            keys=tuple(
                _repeat_heads(k, self.kv_repeats)[:, :, start : start + heads]
                for k in text.keys
            ),
            values=tuple(
                _repeat_heads(v, self.kv_repeats)[:, :, start : start + heads]
                for v in text.values
            ),
        )


@dataclass
class ContextParallelAttention:
    inner: Any
    cp: ContextParallel
    tokens: list[int]  # unsharded token count of every chunk

    def __call__(
        self,
        layer: int,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        cp = self.cp
        heads_q = [
            cp.to_heads(q, n, kv=False)
            for q, n in zip(queries, self.tokens, strict=True)
        ]
        heads_k = [
            cp.to_heads(k, n, kv=True) for k, n in zip(keys, self.tokens, strict=True)
        ]
        heads_v = [
            cp.to_heads(v, n, kv=True) for v, n in zip(values, self.tokens, strict=True)
        ]
        outputs = self.inner(layer, heads_q, heads_k, heads_v)
        return [
            cp.to_tokens(o, q.shape[1]) for o, q in zip(outputs, queries, strict=True)
        ]
