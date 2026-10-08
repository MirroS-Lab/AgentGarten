# SPDX-License-Identifier: Apache-2.0

"""Persistent single-sample attention storage for online Cosmos inference."""

from __future__ import annotations

import torch

from .attention import sdpa

__all__ = ["StreamAttentionCache"]


class StreamAttentionCache:
    """Keep text, completed visuals and query scratch in stable storage."""

    def __init__(
        self,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
        frame_tokens: int,
        capacity: int,
        block_frames: int,
    ) -> None:
        self.text_tokens = text_k.shape[1]
        self.frame_tokens = frame_tokens
        self.capacity = capacity
        self.block_frames = block_frames
        shape = (
            1,
            self.text_tokens + (capacity + block_frames) * frame_tokens,
            *text_k.shape[2:],
        )
        self.key = text_k.new_empty(shape)
        self.value = text_v.new_empty(shape)
        self.key[:, : self.text_tokens].copy_(text_k)
        self.value[:, : self.text_tokens].copy_(text_v)
        self.frames = 0
        # Second storage for ``relocate``, allocated on first use.
        self._spare: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def visual_key(self) -> torch.Tensor:
        return self.key[
            :, self.text_tokens : self.text_tokens + self.frames * self.frame_tokens
        ]

    @property
    def visual_value(self) -> torch.Tensor:
        return self.value[
            :, self.text_tokens : self.text_tokens + self.frames * self.frame_tokens
        ]

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Attend to committed history and an uncommitted current query."""
        start = self.text_tokens + self.frames * self.frame_tokens
        stop = start + key.shape[1]
        self.key[:, start:stop].copy_(key)
        self.value[:, start:stop].copy_(value)
        return sdpa(query, self.key[:, :stop], self.value[:, :stop], None)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Publish clean K/V into the retained history."""
        start = self.text_tokens + self.frames * self.frame_tokens
        self.key[:, start : start + key.shape[1]].copy_(key)
        self.value[:, start : start + value.shape[1]].copy_(value)
        self.frames += key.shape[1] // self.frame_tokens

    def replace_visual(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Replace retained history after FIFO eviction and RoPE relocation."""
        self.frames = key.shape[1] // self.frame_tokens
        self.visual_key.copy_(key)
        self.visual_value.copy_(value)

    def relocate(
        self,
        sink_frames: int,
        evicted_frames: int,
        rotation: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Evict the frames after the sink and rotate the kept keys, in one pass.

        Equivalent to ``replace_visual`` on the retained, rotated history, but
        each tensor is read and written once: the result goes to a second
        storage, which then becomes the cache.
        """
        from wm.kernels.stream_cache import relocate

        if self._spare is None:
            self._spare = torch.empty_like(self.key), torch.empty_like(self.value)
        text, kept = self.text_tokens, sink_frames * self.frame_tokens
        shift = evicted_frames * self.frame_tokens
        used = text + self.frames * self.frame_tokens
        key, value = self._spare
        key[:, :text].copy_(self.key[:, :text])
        value[:, :text].copy_(self.value[:, :text])
        relocate(self.key[:, text:used], key[:, text:], kept, shift, rotation)
        relocate(self.value[:, text:used], value[:, text:], kept, shift)
        self._spare = self.key, self.value
        self.key, self.value = key, value
        self.frames -= evicted_frames

    def update_text(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Update the fixed text prefix."""
        self.key[:, : self.text_tokens].copy_(key)
        self.value[:, : self.text_tokens].copy_(value)
