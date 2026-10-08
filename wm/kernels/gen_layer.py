# SPDX-License-Identifier: Apache-2.0

"""Hand-written serving kernels for the pointwise parts of a GEN layer.

``project`` and ``finish`` compute what ``Cosmos3GenLayer.project`` and
``Cosmos3GenLayer.finish`` compute. The matrix products stay in cuBLAS; each
run of pointwise operations between them becomes one Triton kernel. Every
kernel rounds to BF16 wherever the eager code stores a BF16 tensor, so the
result differs from eager only through the order of the RMS reductions.

These need no graph compilation: Triton builds each kernel once, in seconds,
and caches it on disk.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from wm.networks.cosmos3.layers import Cosmos3GenLayer

__all__ = ["finish", "project"]

# Elements one program handles; a row never spans two programs.
_BLOCK = 4096


@triton.jit
def _bf16(value):
    """Round FP32 to the nearest BF16 value (ties to even), kept in FP32.

    Written on the bits because a ``to(bfloat16).to(float32)`` round trip is
    not always preserved by the compiler.
    """
    bits = value.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    addend_ptr,
    weight_ptr,
    sum_ptr,
    out_ptr,
    rows,
    eps,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    ADD: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, WIDTH)
    offset = row[:, None] * WIDTH + column[None, :]
    mask = (row < rows)[:, None] & (column < WIDTH)[None, :]
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    if ADD:
        # hidden + projection, stored as the layer's BF16 residual.
        addend = tl.load(addend_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        x = _bf16(x + addend)
        tl.store(sum_ptr + offset, x.to(tl.bfloat16), mask=mask)
    scale = tl.rsqrt(tl.sum(x * x, axis=1) / WIDTH + eps)
    normalized = _bf16(x * scale[:, None])
    weight = tl.load(weight_ptr + column).to(tl.float32)
    tl.store(
        out_ptr + offset, (weight[None, :] * normalized).to(tl.bfloat16), mask=mask
    )


@triton.jit
def _norm_rotary_kernel(
    x_ptr,
    weight_ptr,
    cosine_ptr,
    sine_ptr,
    out_ptr,
    rows,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
):
    # One row is one head of one token; cosine and sine are shared by its heads.
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, HALF)
    mask = (row < rows)[:, None] & (column < HALF)[None, :]
    first = row[:, None] * (2 * HALF) + column[None, :]
    second = first + HALF
    a = tl.load(x_ptr + first, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + second, mask=mask, other=0.0).to(tl.float32)
    squares = tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)
    scale = tl.rsqrt(squares / (2 * HALF) + eps)[:, None]
    weight_a = tl.load(weight_ptr + column).to(tl.float32)[None, :]
    weight_b = tl.load(weight_ptr + HALF + column).to(tl.float32)[None, :]
    a = _bf16(weight_a * _bf16(a * scale))
    b = _bf16(weight_b * _bf16(b * scale))

    token = (row // HEADS)[:, None] * (2 * HALF) + column[None, :]
    cosine_a = tl.load(cosine_ptr + token, mask=mask, other=0.0).to(tl.float32)
    cosine_b = tl.load(cosine_ptr + token + HALF, mask=mask, other=0.0).to(tl.float32)
    sine_a = tl.load(sine_ptr + token, mask=mask, other=0.0).to(tl.float32)
    sine_b = tl.load(sine_ptr + token + HALF, mask=mask, other=0.0).to(tl.float32)
    # value * cosine + rotate_half(value) * sine, where rotate_half is (-b, a).
    out_a = _bf16(a * cosine_a) + _bf16(-b * sine_a)
    out_b = _bf16(b * cosine_b) + _bf16(a * sine_b)
    tl.store(out_ptr + first, out_a.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + second, out_b.to(tl.bfloat16), mask=mask)


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    gate = tl.load(gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    activated = _bf16(gate / (1.0 + tl.exp(-gate)))
    tl.store(out_ptr + offset, (activated * up).to(tl.bfloat16), mask=mask)


def _check(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
            raise TypeError("serving kernels take BF16 CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("serving kernels take contiguous tensors")


def _rms_norm(
    value: torch.Tensor,
    norm: torch.nn.Module,
    addend: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """RMSNorm of ``value`` (plus ``addend``): (the BF16 sum or None, the norm)."""
    width = value.shape[-1]
    rows = value.numel() // width
    per_program = max(1, _BLOCK // width)
    out = torch.empty_like(value)
    total = None if addend is None else torch.empty_like(value)
    _rms_norm_kernel[(triton.cdiv(rows, per_program),)](
        value,
        value if addend is None else addend,
        norm.weight,
        out if total is None else total,
        out,
        rows,
        norm.variance_epsilon,
        WIDTH=width,
        ROWS=per_program,
        ADD=addend is not None,
        num_warps=8,
    )
    return total, out


def _norm_rotary(
    value: torch.Tensor,
    norm: torch.nn.Module,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Per-head RMSNorm followed by the rotary rotation, for [B, N, H, D]."""
    heads, width = value.shape[-2:]
    rows = value.numel() // width
    per_program = max(1, _BLOCK // width)
    out = torch.empty_like(value)
    _norm_rotary_kernel[(triton.cdiv(rows, per_program),)](
        value,
        norm.weight,
        cosine,
        sine,
        out,
        rows,
        norm.variance_epsilon,
        HEADS=heads,
        HALF=width // 2,
        ROWS=per_program,
        num_warps=8,
    )
    return out


def project(
    layer: Cosmos3GenLayer,
    hidden: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``layer.project`` with fused pointwise kernels."""
    _check(hidden, cosine, sine)
    batch, tokens, _ = hidden.shape
    attn = layer.self_attn
    _, normalized = _rms_norm(hidden, layer.input_layernorm_moe_gen)
    query = attn.add_q_proj(normalized).view(
        batch, tokens, layer.num_attention_heads, layer.head_dim
    )
    key = attn.add_k_proj(normalized).view(
        batch, tokens, layer.num_key_value_heads, layer.head_dim
    )
    value = attn.add_v_proj(normalized).view(
        batch, tokens, layer.num_key_value_heads, layer.head_dim
    )
    return (
        _norm_rotary(query, attn.norm_added_q, cosine, sine),
        _norm_rotary(key, attn.norm_added_k, cosine, sine),
        value,
    )


def finish(
    layer: Cosmos3GenLayer, hidden: torch.Tensor, attention: torch.Tensor
) -> torch.Tensor:
    """``layer.finish`` with fused pointwise kernels."""
    _check(hidden, attention)
    mlp = layer.mlp_moe_gen
    residual, normalized = _rms_norm(
        hidden,
        layer.post_attention_layernorm_moe_gen,
        layer.self_attn.to_add_out(attention.flatten(-2)),
    )
    gate = mlp.gate_proj(normalized)
    up = mlp.up_proj(normalized)
    activated = torch.empty_like(gate)
    _silu_mul_kernel[(triton.cdiv(gate.numel(), _BLOCK),)](
        gate, up, activated, gate.numel(), BLOCK=_BLOCK, num_warps=8
    )
    return residual + mlp.down_proj(activated)
