# SPDX-License-Identifier: Apache-2.0

"""One-pass eviction and RoPE relocation of a stream's cached keys and values."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["relocate"]

_BLOCK = 4096


@triton.jit
def _relocate_kernel(
    source_ptr,
    out_ptr,
    cosine_ptr,
    sine_ptr,
    rows,
    kept_rows,
    shift_rows,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
    ROTATE: tl.constexpr,
):
    # One row is one head of one retained token. Rows past the sink come from
    # ``shift_rows`` further on in the source, and their keys are rotated.
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, HALF)
    mask = (row < rows)[:, None] & (column < HALF)[None, :]
    moved = row >= kept_rows
    first = (row + tl.where(moved, shift_rows, 0))[:, None] * (2 * HALF) + column[
        None, :
    ]
    a = tl.load(source_ptr + first, mask=mask, other=0.0)
    b = tl.load(source_ptr + first + HALF, mask=mask, other=0.0)
    if ROTATE:
        token = (row // HEADS)[:, None] * (2 * HALF) + column[None, :]
        cosine_a = tl.load(cosine_ptr + token, mask=mask, other=0.0)
        cosine_b = tl.load(cosine_ptr + token + HALF, mask=mask, other=0.0)
        sine_a = tl.load(sine_ptr + token, mask=mask, other=0.0)
        sine_b = tl.load(sine_ptr + token + HALF, mask=mask, other=0.0)
        x = a.to(tl.float32)
        y = b.to(tl.float32)
        # value * cosine + rotate_half(value) * sine, where rotate_half is (-y, x).
        a = tl.where(moved[:, None], x * cosine_a - y * sine_a, x).to(tl.bfloat16)
        b = tl.where(moved[:, None], y * cosine_b + x * sine_b, y).to(tl.bfloat16)
    target = row[:, None] * (2 * HALF) + column[None, :]
    tl.store(out_ptr + target, a, mask=mask)
    tl.store(out_ptr + target + HALF, b, mask=mask)


def relocate(
    source: torch.Tensor,
    out: torch.Tensor,
    kept_tokens: int,
    shift_tokens: int,
    rotation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Write the retained tokens of ``source`` [1, N, H, D] to the front of ``out``.

    The first ``kept_tokens`` stay in place; the tokens after them are taken
    ``shift_tokens`` further on. With ``rotation`` (FP32 cosine and sine,
    [1, tokens, D]) the moved tokens are rotated on the way, as
    ``apply_rotary(key.float(), cosine, sine)`` does. ``source`` and ``out``
    must not overlap.
    """
    heads, width = source.shape[-2:]
    tokens = source.shape[1] - shift_tokens
    rows = tokens * heads
    per_program = max(1, _BLOCK // width)
    cosine, sine = (source, source) if rotation is None else rotation
    _relocate_kernel[(triton.cdiv(rows, per_program),)](
        source,
        out,
        cosine,
        sine,
        rows,
        kept_tokens * heads,
        shift_tokens * heads,
        HEADS=heads,
        HALF=width // 2,
        ROWS=per_program,
        ROTATE=rotation is not None,
        num_warps=8,
    )
