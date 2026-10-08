# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from RCM (Kaiwen Zheng), based on OpenAI's Triton attention tutorial:
# https://github.com/NVlabs/rcm (commit ed3cb14),
# rcm/utils/flash_attention_jvp_triton.py; license in LICENSE-RCM.txt.
# Changes (chijw/fa3-jvp, commit 56c790c): combined derivative accumulators,
# native GQA, FP32 output, key validity masks, safe empty rows.

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_jvp(Q, K, V, DQ, DK, DV, LENS, MASK, O, DO, LSE,
               HQ: tl.constexpr, HK: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
               D: tl.constexpr, VD: tl.constexpr, SCALE: tl.constexpr,
               HAS_LENS: tl.constexpr, HAS_MASK: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    b = bh // HQ
    kh = (bh % HQ) // (HQ // HK)
    di = tl.arange(0, D)
    vi = tl.arange(0, VD)
    ni = tl.arange(0, BN)
    qoff = (bh * NQ + mi[:, None]) * D + di[None, :]
    q = tl.load(Q + qoff, mi[:, None] < NQ, 0)
    dq = tl.load(DQ + qoff, mi[:, None] < NQ, 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    r = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, VD), tl.float32)
    t = tl.zeros((BM, VD), tl.float32)
    length = NK
    if HAS_LENS:
        length = tl.minimum(NK, tl.maximum(0, tl.load(LENS + b)))
    for start in range(0, length, BN):
        ns = start + ni
        valid = ns < length
        if HAS_MASK:
            valid = valid & tl.load(MASK + b * NK + ns, ns < NK, 0)
        koff = ((b * HK + kh) * NK + ns[None, :]) * D + di[:, None]
        k = tl.load(K + koff, valid[None, :], 0)
        dk = tl.load(DK + koff, valid[None, :], 0)
        s = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        s = tl.where(valid[None, :], s, -float("inf"))
        new_m = tl.maximum(m, tl.max(s, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        p = tl.exp2(s - safe_m[:, None])
        alpha = tl.exp2(m - safe_m)
        l = l * alpha + tl.sum(p, 1)
        o = o * alpha[:, None]
        t = t * alpha[:, None]
        ds = (tl.dot(dq, k) + tl.dot(q, dk)) * SCALE
        pd = p * ds
        r = r * alpha + tl.sum(pd, 1)
        voff = ((b * HK + kh) * NK + ns[:, None]) * VD + vi[None, :]
        v = tl.load(V + voff, valid[:, None], 0)
        dv = tl.load(DV + voff, valid[:, None], 0)
        pb = p.to(tl.bfloat16)
        pdb = pd.to(tl.bfloat16)
        o = tl.dot(pb, v, o)
        t = tl.dot(pb, dv, t)
        t = tl.dot(pdb, v, t)
        m = new_m
    inv_l = 1.0 / tl.where(l > 0, l, 1.0)
    out = o * inv_l[:, None]
    dout = t * inv_l[:, None] - out * (r * inv_l)[:, None]
    oo = (bh * NQ + mi[:, None]) * VD + vi[None, :]
    tl.store(O + oo, out, mi[:, None] < NQ)
    tl.store(DO + oo, dout, mi[:, None] < NQ)
    tl.store(LSE + bh * NQ + mi, (m + tl.log2(l)) * 0.6931471805599453, mi < NQ)


@torch.no_grad()
def attention_jvp(q, k, v, dq, dk, dv, *, key_lengths=None, key_mask=None,
                  key_bias=None, scale=None, config=None):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected [B,H,N,D] tensors")
    b, hq, nq, d = q.shape
    bk, hk, nk, kd = k.shape
    if min(b, hq, nq, d, bk, hk, nk, kd) <= 0:
        raise ValueError("Dimensions must be positive")
    if b != bk or d != kd or hq % hk or v.shape[:3] != k.shape[:3]:
        raise ValueError("Incompatible GQA shapes")
    vd = v.shape[-1]
    if d not in (16, 32, 64, 128, 256) or vd not in (16, 32, 64, 128, 256):
        raise ValueError("Head dimensions must be 16,32,64,128,256")
    for x, dx in ((q, dq), (k, dk), (v, dv)):
        if x.dtype != torch.bfloat16 or dx.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Requires BF16 primals and FP32/FP16/BF16 tangents")
        if x.shape != dx.shape:
            raise ValueError("Tangent shape mismatch")
        for a in (x, dx):
            if not a.is_cuda or a.device != q.device or not a.is_contiguous():
                raise ValueError("Inputs must be contiguous on the same CUDA device")
            if a.requires_grad:
                raise ValueError("Forward JVP only; detach inputs explicitly (no autograd backward)")
    if key_bias is not None:
        raise NotImplementedError("Additive bias is not implemented")
    for a, shape, dtypes in ((key_lengths, (b,), (torch.int32, torch.int64)),
                              (key_mask, (b, nk), (torch.bool,))):
        if a is not None and (a.shape != shape or a.dtype not in dtypes or a.device != q.device or not a.is_contiguous()):
            raise ValueError("Invalid key_lengths or key_mask")
    scale = d ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    if config is None:
        if max(d, vd) == 256:
            config = (32, 32, 8, 1)
        elif d == 128 and vd == 128 and nq >= 128:
            config = (128, 64, 8, 2)
        elif max(d, vd) <= 64 and b * hq * triton.cdiv(nq, 64) < 32:
            config = (16, 128, 4, 3)
        else:
            config = (64, 64, 4, 3)
    bm, bn, warps, stages = config
    if bm not in (16, 32, 64, 128) or bn not in (16, 32, 64, 128) or warps not in (4, 8) or stages not in (1, 2, 3, 4, 5, 7):
        raise ValueError("Unsupported launch configuration")
    dq, dk, dv = [x.to(torch.bfloat16) for x in (dq, dk, dv)]
    o = torch.empty((b, hq, nq, vd), device=q.device, dtype=torch.float32)
    do = torch.empty_like(o)
    lse = torch.empty((b, hq, nq), device=q.device, dtype=torch.float32)
    _fused_jvp[(triton.cdiv(nq, bm), b * hq)](
        q, k, v, dq, dk, dv, key_lengths, key_mask, o, do, lse,
        hq, hk, nq, nk, d, vd, scale, key_lengths is not None, key_mask is not None,
        bm, bn, num_warps=warps, num_stages=stages)
    return o, do, lse
