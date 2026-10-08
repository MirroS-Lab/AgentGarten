# SPDX-License-Identifier: Apache-2.0

"""Contracts between family-agnostic models and a network family.

Models live in wm.models, families in wm.networks.<family>. Latents are
[B, C, T, H, W] with frame 0 the clean C0; sigma follows rectified flow
(1 = noise, 0 = data), per sample [B] or per frame [B, T]; networks predict the
velocity noise - x0. Models never look inside a Condition.

BlockARNetwork: forward_ar(clean=x) must reproduce, bitwise, a rollout that
prefills C0 and commits x block by block (Self Gradient Forcing relies on it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

import torch

__all__ = [
    "BlockARNetwork",
    "Condition",
    "Conditioner",
    "DenoisingNetwork",
    "Discriminator",
    "HistoryWindow",
]

T = TypeVar("T")


@runtime_checkable
class Condition(Protocol):
    def repeat(self, times: int) -> Self: ...


class Conditioner(Protocol):
    # Frozen text tower (a ParallelNetwork), parallelized by the model.
    text_encoder: torch.nn.Module

    def encode(self, batch: Mapping[str, Any], *, training: bool) -> dict[str, Any]: ...

    def condition(
        self,
        batch: Mapping[str, Any],
        *,
        negative: bool = False,
        drop: torch.Tensor | None = None,
    ) -> Condition: ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor: ...

    def visuals(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]: ...


class DenoisingNetwork(Protocol):
    def __call__(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        condition: Condition,
        *,
        known_frames: int = 1,
        feature_layers: Sequence[int] = (),
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]: ...


# Generated blocks an AR query attends to: the first ``sink_blocks`` and the
# last ``recent_blocks`` (``None`` keeps all). C0 is always visible on top.
@dataclass(frozen=True)
class HistoryWindow:
    sink_blocks: int = 0
    recent_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.sink_blocks < 0 or (
            self.recent_blocks is not None and self.recent_blocks < 0
        ):
            raise ValueError(f"history window sizes must be non-negative, got {self}")

    def visible(self, blocks: Sequence[T]) -> list[T]:
        if (
            self.recent_blocks is None
            or len(blocks) <= self.sink_blocks + self.recent_blocks
        ):
            return list(blocks)
        tail = len(blocks) - self.recent_blocks
        return [*blocks[: self.sink_blocks], *blocks[tail:]]


class BlockARNetwork(DenoisingNetwork, Protocol):
    def forward_ar(
        self,
        anchor: torch.Tensor,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        condition: Condition,
        *,
        block_frames: int,
        clean: torch.Tensor | None = None,
        history: HistoryWindow | None = None,
    ) -> torch.Tensor: ...

    def prefill(
        self,
        anchor: torch.Tensor,
        condition: Condition,
        *,
        history: HistoryWindow | None = None,
    ) -> Any: ...

    def denoise(
        self, cache: Any, noisy: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor: ...

    def commit(
        self, cache: Any, clean: torch.Tensor, sigma: torch.Tensor | None = None
    ) -> Any: ...


@runtime_checkable
class Discriminator(Protocol):
    feature_layers: tuple[int, ...]

    def __call__(
        self, feature_groups: Sequence[Sequence[torch.Tensor]]
    ) -> torch.Tensor: ...

    def parallelize(self, config: Any, device: torch.device | str) -> None: ...
