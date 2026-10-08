# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from wm.protocols import HistoryWindow

if TYPE_CHECKING:
    from .network import Cosmos3Condition

__all__ = [
    "BlockCausalAttention",
    "FullAttention",
    "HistoryWindow",
    "KVCache",
    "TextContext",
    "sdpa",
]


@dataclass(frozen=True)
class TextContext:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    valid_lengths: torch.Tensor
    lengths: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    def repeat(self, times: int) -> TextContext:
        if times == 1:
            return self
        return TextContext(
            keys=tuple(torch.cat((key,) * times) for key in self.keys),
            values=tuple(torch.cat((value,) * times) for value in self.values),
            valid_lengths=torch.cat((self.valid_lengths,) * times),
            lengths=self.lengths * times,
        )

    def head_slice(self, start: int, stop: int) -> TextContext:
        return replace(
            self,
            keys=tuple(key[:, :, start:stop] for key in self.keys),
            values=tuple(value[:, :, start:stop] for value in self.values),
        )


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    q, k, v = (x.transpose(1, 2) for x in (query, key, value))
    if any(fwAD.unpack_dual(x).tangent is not None for x in (q, k, v)):
        key_mask = None if mask is None else mask.reshape(mask.shape[0], -1)
        return _AttentionJVP.apply(q, k, v, key_mask).transpose(1, 2)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    return output.transpose(1, 2)


def _reference_attention_jvp(q, k, v, dq, dk, dv, key_mask):
    repeat = q.shape[1] // k.shape[1]
    q, dq = q.double(), dq.double()
    k, v, dk, dv = (x.repeat_interleave(repeat, 1).double() for x in (k, v, dk, dv))
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-1, -2) * scale
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    probabilities = scores.softmax(-1)
    d_scores = (dq @ k.transpose(-1, -2) + q @ dk.transpose(-1, -2)) * scale
    d_probabilities = probabilities * (
        d_scores - (probabilities * d_scores).sum(-1, keepdim=True)
    )
    return d_probabilities @ v + probabilities @ dv


class _AttentionJVP(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, key_mask):
        mask = None if key_mask is None else key_mask[:, None, None, :]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, key_mask = inputs
        ctx.save_for_forward(q, k, v)
        ctx.key_mask = key_mask

    @staticmethod
    def jvp(ctx, dq, dk, dv, _dmask):
        q, k, v = ctx.saved_tensors
        tangents = [
            torch.zeros_like(x) if dx is None else dx
            for x, dx in ((q, dq), (k, dk), (v, dv))
        ]
        if q.is_cuda and q.dtype == torch.bfloat16:
            from wm.kernels.attention_jvp import attention_jvp

            _, tangent = attention_jvp(q, k, v, *tangents, key_mask=ctx.key_mask)
            return tangent.to(q.dtype)
        tangent = _reference_attention_jvp(q, k, v, *tangents, ctx.key_mask)
        return tangent.to(q.dtype)


def _key_mask(text: TextContext, text_tokens: int, visual_tokens: int) -> torch.Tensor:
    device = text.valid_lengths.device
    valid_text = (
        torch.arange(text_tokens, device=device)[None] < text.valid_lengths[:, None]
    )
    visual = torch.ones(
        valid_text.shape[0], visual_tokens, dtype=torch.bool, device=device
    )
    return torch.cat((valid_text, visual), dim=1)[:, None, None]


@dataclass
class FullAttention:
    text: TextContext

    def __call__(
        self,
        layer: int,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        text_k, text_v = self.text.keys[layer], self.text.values[layer]
        key = torch.cat((text_k, *keys), dim=1)
        value = torch.cat((text_v, *values), dim=1)
        visual = key.shape[1] - text_k.shape[1]
        mask = _key_mask(self.text, text_k.shape[1], visual)
        output = sdpa(torch.cat(queries, dim=1), key, value, mask)
        return list(output.split([query.shape[1] for query in queries], dim=1))


@dataclass
class BlockCausalAttention:
    text: TextContext
    teacher_forcing: bool
    history: HistoryWindow = field(default_factory=HistoryWindow)

    def __call__(
        self,
        layer: int,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        stride = 2 if self.teacher_forcing else 1
        if (len(queries) - 1) % stride:
            raise ValueError(
                "teacher forcing needs a publication chunk per query chunk"
            )
        anchor = (keys[0], values[0], [0])
        frame_tokens = keys[0].shape[1]
        outputs = [_attend_block(self.text, layer, queries[0], [], keys[0], values[0])]
        history: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
        next_frame = 1
        for start in range(1, len(queries), stride):
            prefix = [anchor, *self.history.visible(history)]
            for index in range(start, start + stride):
                outputs.append(
                    _attend_block(
                        self.text,
                        layer,
                        queries[index],
                        prefix,
                        keys[index],
                        values[index],
                    )
                )
            # The last chunk of a block (publication for TF, the query itself
            # for DF) becomes visible history for later blocks.
            last = start + stride - 1
            frames = keys[last].shape[1] // frame_tokens
            history.append(
                (keys[last], values[last], list(range(next_frame, next_frame + frames)))
            )
            next_frame += frames
        return outputs


def _attend_block(
    text: TextContext,
    layer: int,
    query: torch.Tensor,
    prefix: list[tuple[torch.Tensor, torch.Tensor, list[int]]],
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    text_k, text_v = text.keys[layer], text.values[layer]
    prefix_k = [k for k, _, _ in prefix]
    prefix_v = [v for _, v, _ in prefix]
    return _attend(text, query, text_k, text_v, [*prefix_k, key], [*prefix_v, value])


def _attend(
    text: TextContext,
    query: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    visual_k: list[torch.Tensor],
    visual_v: list[torch.Tensor],
) -> torch.Tensor:
    key = torch.cat((text_k, *visual_k), dim=1)
    value = torch.cat((text_v, *visual_v), dim=1)
    mask = _key_mask(text, text_k.shape[1], key.shape[1] - text_k.shape[1])
    return sdpa(query, key, value, mask)


@dataclass
class KVCache:
    text: TextContext
    anchor: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    # Only blocks inside the history window are stored, each with its absolute
    # frame indices; positions come from ``frames_committed``, not the storage.
    blocks: tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], ...] = ()
    block_frames: tuple[range, ...] = ()
    history: HistoryWindow = field(default_factory=HistoryWindow)
    frames_committed: int = 0
    # Owned by the network: the condition and RGB patch grid of this rollout.
    condition: Cosmos3Condition | None = None
    grid: tuple[int, int] = (0, 0)

    def prefix(self, layer: int) -> list[tuple[torch.Tensor, torch.Tensor, list[int]]]:
        return [
            (*self.anchor[layer], [0]),
            *(
                (*block[layer], list(frames))
                for block, frames in zip(self.blocks, self.block_frames, strict=True)
            ),
        ]

    def committed(
        self,
        block: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        frames: int,
    ) -> KVCache:
        start = 1 + self.frames_committed
        kept = self.history.visible(
            [
                *zip(self.blocks, self.block_frames, strict=True),
                (block, range(start, start + frames)),
            ]
        )
        return replace(
            self,
            blocks=tuple(kv for kv, _ in kept),
            block_frames=tuple(span for _, span in kept),
            frames_committed=self.frames_committed + frames,
        )


@dataclass
class CachedAttention:
    cache: KVCache
    record: bool = False
    recorded: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def __call__(
        self,
        layer: int,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        (query,), (key,), (value,) = queries, keys, values
        cache = self.cache
        output = _attend_block(
            cache.text, layer, query, cache.prefix(layer), key, value
        )
        if self.record:
            self.recorded.append((key, value))
        return [output]


@dataclass
class AnchorAttention:
    text: TextContext
    recorded: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def __call__(
        self,
        layer: int,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        (query,), (key,), (value,) = queries, keys, values
        self.recorded.append((key, value))
        return [_attend_block(self.text, layer, query, [], key, value)]
