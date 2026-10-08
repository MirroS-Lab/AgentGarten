# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Forward-mode attention JVP for Hopper (chijw/fa3-jvp, commit 56c790c),
# adapting the TMA/WGMMA patterns and layout helpers of NVIDIA CUTLASS
# examples/python/CuTeDSL/hopper/fmha.py.

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
import torch
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack


class MmaLayoutHelpers:
    @staticmethod
    def convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )


    @staticmethod
    def layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())

        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])

        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r


    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
        )

        V_M = separated[0]
        V_N = separated[1]
        V_M1 = None
        V_N1 = None
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])

        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        r = None
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r



@cute.jit
def _logical_view(x: cute.Tensor, value: cutlass.Constexpr = False):
    b, h, n, d = x.shape
    if const_expr(value):
        return cute.make_tensor(x.iterator, cute.make_layout((d, n, b * h), stride=(1, d, n * d)))
    else:
        return cute.make_tensor(x.iterator, cute.make_layout((n, d, b * h), stride=(d, 1, n * d)))


class CuteJVP(MmaLayoutHelpers):
    def __init__(self, d, bn=64, prefetch=False, has_lens=False, has_mask=False):
        self.d, self.bm, self.bn = d, 64, bn
        self.prefetch = prefetch
        self.has_lens, self.has_mask = has_lens, has_mask

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
                 dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor,
                 o: cute.Tensor, do: cute.Tensor, lse: cute.Tensor,
                 lengths: Optional[cute.Tensor], mask: Optional[cute.Tensor],
                 hq: cutlass.Constexpr, hk: cutlass.Constexpr,
                 scale: Float32, stream: cuda.CUstream):
        q = _logical_view(q)
        k = _logical_view(k)
        v = _logical_view(v, True)
        if const_expr(dq.element_type == Float32):
            # dQ occupies the first BF16 half of each FP32 output row.
            # This CTA loads its entire dQ tile before writing any output row.
            b, h, n, d = dq.shape
            dq = cute.make_tensor(cute.recast_ptr(dq.iterator, dtype=cutlass.BFloat16),
                cute.make_layout((n, d, b * h), stride=(2 * d, 1, 2 * n * d)))
        else:
            dq = _logical_view(dq)
        dk = _logical_view(dk)
        dv = _logical_view(dv, True)
        o = _logical_view(o)
        do = _logical_view(do)
        lse = cute.make_tensor(lse.iterator, cute.make_layout(
            (lse.shape[2], lse.shape[0] * lse.shape[1]), stride=(1, lse.shape[2])))
        d, bm, bn = self.d, self.bm, self.bn
        qmma = sm90.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32, (1, 1, 1), (bm, bn))
        pmma = sm90.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            Float32, (1, 1, 1), (bm, d), warpgroup.OperandSource.RMEM)
        qlayout = cute.slice_(sm90.make_smem_layout_a(utils.LayoutEnum.ROW_MAJOR,
                            (bm, bn, d), cutlass.BFloat16, 1), (None, None, 0))
        klayout = cute.slice_(sm90.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR,
                            (bm, bn, d), cutlass.BFloat16, 1), (None, None, 0))
        vlayout = cute.slice_(sm90.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR,
                            (bm, d, bn), cutlass.BFloat16, 1), (None, None, 0))
        qa, qt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), q, qlayout, (bm, d))
        dqa, dqt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), dq, qlayout, (bm, d))
        ka, kt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), k, klayout, (bn, d))
        dka, dkt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), dk, klayout, (bn, d))
        va, vt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), v, vlayout, (d, bn))
        dva, dvt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), dv, vlayout, (d, bn))
        self.kernel(qmma, pmma, qlayout, klayout, vlayout,
                    qa, qt, dqa, dqt, ka, kt, dka, dkt, va, vt, dva, dvt,
                    o, do, lse, lengths, mask, hq, hk, scale).launch(
                        grid=(cute.ceil_div(q.shape[0], bm), q.shape[2], 1),
                        block=(128, 1, 1), stream=stream)

    @cute.jit
    def parts(self, atom, smem, gmem):
        return cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                    cute.group_modes(smem, 0, 2), cute.group_modes(gmem, 0, 2))

    @cute.jit
    def mma(self, atom, a, b, c, zero: cutlass.Constexpr):
        for idx in cutlass.range_constexpr(cute.size(a, mode=[2])):
            atom.set(warpgroup.Field.ACCUMULATE, not zero or idx != 0)
            cute.gemm(atom, c, a[None, None, idx], b[None, None, idx], c)

    @cute.jit
    def operand(self, acc, mma):
        op = cute.make_rmem_tensor(
            self.convert_c_layout_to_a_layout(acc.layout, mma.tv_layout_A.shape[1]),
            cutlass.BFloat16)
        as_acc = cute.make_tensor(op.iterator, acc.layout)
        as_acc.store(acc.load().to(cutlass.BFloat16))
        return op

    @cute.jit
    def mn(self, acc, mma):
        return cute.make_tensor(acc.iterator, self.layout_acc_mn(mma, acc.layout))

    @cute.kernel
    def kernel(self, qmma: cute.TiledMma, pmma: cute.TiledMma,
               qlayout: cute.ComposedLayout, klayout: cute.ComposedLayout, vlayout: cute.ComposedLayout,
               qa: cute.CopyAtom, q: cute.Tensor, dqa: cute.CopyAtom, dq: cute.Tensor,
               ka: cute.CopyAtom, k: cute.Tensor, dka: cute.CopyAtom, dk: cute.Tensor,
               va: cute.CopyAtom, v: cute.Tensor, dva: cute.CopyAtom, dv: cute.Tensor,
               o: cute.Tensor, do: cute.Tensor, lse: cute.Tensor,
               lengths: Optional[cute.Tensor], mask: Optional[cute.Tensor],
               hq: cutlass.Constexpr, hk: cutlass.Constexpr, scale: Float32):
        tid, _, _ = cute.arch.thread_idx()
        block_m, bh, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        kh = bh // hq * hk + (bh % hq) // (hq // hk)
        d, bm, bn = self.d, self.bm, self.bn
        nq, nk = q.shape[0], k.shape[0]
        length = nk
        if const_expr(self.has_lens):
            length = cutlass.min(nk, cutlass.max(0, lengths[bh // hq]))
        blocks = cutlass.max(1, cute.ceil_div(length, bn))
        smem = utils.SmemAllocator()
        sq = smem.allocate_tensor(cutlass.BFloat16, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sdq = smem.allocate_tensor(cutlass.BFloat16, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sk = smem.allocate_tensor(cutlass.BFloat16, klayout.outer, byte_alignment=128, swizzle=klayout.inner)
        sdk = smem.allocate_tensor(cutlass.BFloat16, klayout.outer, byte_alignment=128, swizzle=klayout.inner)
        sv = smem.allocate_tensor(cutlass.BFloat16, vlayout.outer, byte_alignment=128, swizzle=vlayout.inner)
        sdv = smem.allocate_tensor(cutlass.BFloat16, vlayout.outer, byte_alignment=128, swizzle=vlayout.inner)
        barriers = smem.allocate_array(cutlass.Int64, 3)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barriers, 1)
                cute.arch.mbarrier_init(barriers + 1, 1)
                cute.arch.mbarrier_init(barriers + 2, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        qs, qg = self.parts(qa, sq, cute.local_tile(q, (bm, d), (None, 0, bh)))
        dqs, dqg = self.parts(dqa, sdq, cute.local_tile(dq, (bm, d), (None, 0, bh)))
        ks, kg = self.parts(ka, sk, cute.local_tile(k, (bn, d), (None, 0, kh)))
        dks, dkg = self.parts(dka, sdk, cute.local_tile(dk, (bn, d), (None, 0, kh)))
        vs, vg = self.parts(va, sv, cute.local_tile(v, (d, bn), (0, None, kh)))
        dvs, dvg = self.parts(dva, sdv, cute.local_tile(dv, (d, bn), (0, None, kh)))
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barriers, 2 * bm * d * 2)
            cute.copy(qa, qg[None, block_m], qs, tma_bar_ptr=barriers)
            cute.copy(dqa, dqg[None, block_m], dqs, tma_bar_ptr=barriers)
            if const_expr(self.prefetch):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(barriers + 1, 2 * bn * d * 2)
                    cute.arch.mbarrier_arrive_and_expect_tx(barriers + 2, 2 * bn * d * 2)
                cute.copy(ka, kg[None, 0], ks, tma_bar_ptr=barriers + 1)
                cute.copy(dka, dkg[None, 0], dks, tma_bar_ptr=barriers + 1)
                cute.copy(va, vg[None, 0], vs, tma_bar_ptr=barriers + 2)
                cute.copy(dva, dvg[None, 0], dvs, tma_bar_ptr=barriers + 2)
        qthr = qmma.get_slice(tid)
        pthr = pmma.get_slice(tid)
        aq = qmma.make_fragment_A(qthr.partition_A(sq))
        adq = qmma.make_fragment_A(qthr.partition_A(sdq))
        bk = qmma.make_fragment_B(qthr.partition_B(sk))
        bdk = qmma.make_fragment_B(qthr.partition_B(sdk))
        bv = pmma.make_fragment_B(pthr.partition_B(sv))
        bdv = pmma.make_fragment_B(pthr.partition_B(sdv))
        score = cute.make_rmem_tensor(qthr.partition_C(cute.make_identity_tensor((bm, bn))).shape, Float32)
        dscore = cute.make_rmem_tensor_like(score)
        out = cute.make_rmem_tensor(pthr.partition_C(cute.make_identity_tensor((bm, d))).shape, Float32)
        dout = cute.make_rmem_tensor_like(out)
        out.fill(0.0)
        dout.fill(0.0)
        smn, dsmn = self.mn(score, qmma), self.mn(dscore, qmma)
        omn, domn = self.mn(out, pmma), self.mn(dout, pmma)
        rowmax = cute.make_rmem_tensor(cute.size(smn, mode=[0]), Float32)
        rowsum = cute.make_rmem_tensor_like(rowmax)
        rowds = cute.make_rmem_tensor_like(rowmax)
        rowmax.fill(-Float32.inf)
        rowsum.fill(0.0)
        rowds.fill(0.0)
        scoord = self.mn(qthr.partition_C(cute.make_identity_tensor((bm, bn))), qmma)
        oc = self.mn(pthr.partition_C(cute.make_identity_tensor((bm, d))), pmma)
        og = self.mn(pthr.partition_C(cute.local_tile(o, (bm, d), (block_m, 0, bh))), pmma)
        dog = self.mn(pthr.partition_C(cute.local_tile(do, (bm, d), (block_m, 0, bh))), pmma)
        cute.arch.mbarrier_wait(barriers, 0)
        for nb in range(blocks):
            if const_expr(not self.prefetch):
                if warp == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers + 1, 4 * bn * d * 2)
                    cute.copy(ka, kg[None, nb], ks, tma_bar_ptr=barriers + 1)
                    cute.copy(dka, dkg[None, nb], dks, tma_bar_ptr=barriers + 1)
                    cute.copy(va, vg[None, nb], vs, tma_bar_ptr=barriers + 1)
                    cute.copy(dva, dvg[None, nb], dvs, tma_bar_ptr=barriers + 1)
            cute.arch.mbarrier_wait(barriers + 1, nb % 2)
            warpgroup.fence()
            self.mma(qmma, aq, bk, score, True)
            warpgroup.commit_group()
            self.mma(qmma, adq, bk, dscore, True)
            self.mma(qmma, aq, bdk, dscore, False)
            warpgroup.commit_group()
            # Primal score is ready; tangent score may still be in flight.
            warpgroup.wait_group(1)
            for i in cutlass.range_constexpr(cute.size(smn, mode=[0])):
                local_max = rowmax[i]
                for j in cutlass.range_constexpr(cute.size(smn, mode=[1])):
                    s = smn[i, j] * (scale * 1.4426950408889634)
                    column = scoord[i, j][1] + nb * bn
                    valid = column < length
                    if const_expr(self.has_mask):
                        if column < nk:
                            valid = valid & mask[bh // hq, column]
                    if not valid:
                        s = -Float32.inf
                    smn[i, j] = s
                    local_max = cute.arch.fmax(local_max, s)
                newmax = cute.arch.warp_reduction_max(local_max, threads_in_group=4)
                safe_max = newmax
                if newmax == -Float32.inf:
                    safe_max = 0.0
                alpha = cute.math.exp2(rowmax[i] - safe_max, fastmath=True)
                rowmax[i] = newmax
                rowsum[i] *= alpha
                rowds[i] *= alpha
                for j in cutlass.range_constexpr(cute.size(omn, mode=[1])):
                    omn[i, j] *= alpha
                    domn[i, j] *= alpha
                for j in cutlass.range_constexpr(cute.size(smn, mode=[1])):
                    p = cute.math.exp2(smn[i, j] - safe_max, fastmath=True)
                    smn[i, j] = p
                    rowsum[i] += p
            # Both score groups must finish before dscore use and K/dK overwrite.
            warpgroup.wait_group(0)
            if const_expr(self.prefetch):
                cute.arch.sync_threads()
                if warp == 0 and nb + 1 < blocks:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers + 1, 2 * bn * d * 2)
                    cute.copy(ka, kg[None, nb + 1], ks, tma_bar_ptr=barriers + 1)
                    cute.copy(dka, dkg[None, nb + 1], dks, tma_bar_ptr=barriers + 1)
            for i in cutlass.range_constexpr(cute.size(smn, mode=[0])):
                for j in cutlass.range_constexpr(cute.size(smn, mode=[1])):
                    column = scoord[i, j][1] + nb * bn
                    valid = column < length
                    if const_expr(self.has_mask):
                        if column < nk:
                            valid = valid & mask[bh // hq, column]
                    pd = Float32(0.0)
                    if valid:
                        pd = smn[i, j] * dsmn[i, j] * scale
                    dsmn[i, j] = pd
                    rowds[i] += pd
            p = self.operand(score, pmma)
            pd = self.operand(dscore, pmma)
            if const_expr(self.prefetch):
                cute.arch.mbarrier_wait(barriers + 2, nb % 2)
            warpgroup.fence()
            self.mma(pmma, p, bv, out, False)
            self.mma(pmma, p, bdv, dout, False)
            self.mma(pmma, pd, bv, dout, False)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            cute.arch.sync_threads()
            if const_expr(self.prefetch):
                if warp == 0 and nb + 1 < blocks:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers + 2, 2 * bn * d * 2)
                    cute.copy(va, vg[None, nb + 1], vs, tma_bar_ptr=barriers + 2)
                    cute.copy(dva, dvg[None, nb + 1], dvs, tma_bar_ptr=barriers + 2)
        for i in cutlass.range_constexpr(cute.size(omn, mode=[0])):
            norm = cute.arch.warp_reduction_sum(rowsum[i], threads_in_group=4)
            inv = Float32(1.0)
            if norm > 0:
                inv = 1.0 / norm
            mean = cute.arch.warp_reduction_sum(rowds[i], threads_in_group=4) * inv
            for j in cutlass.range_constexpr(cute.size(omn, mode=[1])):
                val = omn[i, j] * inv
                omn[i, j] = val
                domn[i, j] = domn[i, j] * inv - val * mean
            if oc[i, 0][0] + block_m * bm < nq:
                cute.autovec_copy(omn[i, None], og[i, None])
                cute.autovec_copy(domn[i, None], dog[i, None])
            if oc[i, 0][1] == 0 and oc[i, 0][0] + block_m * bm < nq:
                lse[oc[i, 0][0] + block_m * bm, bh] = (rowmax[i] + cute.math.log2(norm, fastmath=True)) * 0.6931471805599453


class _CastTangents:
    def __init__(self, chunks=1):
        self.chunks = chunks

    @cute.jit
    def __call__(self, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor,
                 oq: cute.Tensor, ok: cute.Tensor, ov: cute.Tensor, stream: cuda.CUstream):
        nq = cute.ceil_div(cute.size(dq), 1024 * self.chunks)
        nk = cute.ceil_div(cute.size(dk), 1024 * self.chunks)
        nv = cute.ceil_div(cute.size(dv), 1024 * self.chunks)
        self.kernel(dq, dk, dv, oq, ok, ov, nq, nk).launch(
            grid=(nq + nk + nv, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def convert(self, src: cute.Tensor, dst: cute.Tensor, block, tid):
        if const_expr(src.element_type != cutlass.BFloat16):
            for chunk in cutlass.range_constexpr(self.chunks):
                index = cute.assume((block * 128 * self.chunks + tid + chunk * 128) * 8, divby=8)
                if index < cute.size(src):
                    # Every supported head dimension is divisible by eight.
                    r = cute.make_rmem_tensor(8, src.element_type)
                    result = cute.make_rmem_tensor(8, cutlass.BFloat16)
                    s = cute.make_tensor(src.iterator + index, cute.make_layout(8))
                    if const_expr(dst.element_type == Float32):
                        width = src.shape[-1]
                        offset = (index // width) * (2 * width) + index % width
                        ptr = cute.recast_ptr(dst.iterator, dtype=cutlass.BFloat16)
                        d = cute.make_tensor(ptr + offset, cute.make_layout(8))
                    else:
                        d = cute.make_tensor(dst.iterator + index, cute.make_layout(8))
                    cute.autovec_copy(s, r)
                    result.store(r.load().to(cutlass.BFloat16))
                    cute.autovec_copy(result, d)

    @cute.kernel
    def kernel(self, dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor,
               oq: cute.Tensor, ok: cute.Tensor, ov: cute.Tensor,
               nq: cutlass.Constexpr, nk: cutlass.Constexpr):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if block < nq:
            self.convert(dq, oq, block, tid)
        elif block < nq + nk:
            self.convert(dk, ok, block - nq, tid)
        else:
            self.convert(dv, ov, block - nq - nk, tid)


_compiled = {}
_cast_compiled = {}


def _cast_tangents(xs, chunks=None, out_q=None):
    if all(x.dtype == torch.bfloat16 for x in xs):
        return xs
    if chunks is None:
        # More work per CTA helps medium sizes; large casts are bandwidth bound.
        total = sum(x.numel() for x in xs if x.dtype != torch.bfloat16)
        chunks = 4 if 2**21 <= total <= 2**23 else 1
    outputs = tuple(out_q if i == 0 and out_q is not None else
                    x if x.dtype == torch.bfloat16 else torch.empty_like(x, dtype=torch.bfloat16)
                    for i, x in enumerate(xs))
    tensors = (*xs, *outputs)
    stream = cuda.CUstream(torch.cuda.current_stream(xs[0].device).cuda_stream)
    key = (tuple((tuple(x.shape), x.dtype) for x in xs), xs[0].device.index, chunks, out_q is not None)
    if key not in _cast_compiled:
        cs = tuple(from_dlpack(x, assumed_align=16, enable_tvm_ffi=True) for x in tensors)
        _cast_compiled[key] = cute.compile(_CastTangents(chunks), *cs, stream, options="--enable-tvm-ffi")
    _cast_compiled[key](*tensors, stream)
    return outputs


class _CastThenJVP:
    def __init__(self, chunks, attention):
        self.cast = _CastTangents(chunks)
        self.attention = attention

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
                 dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor,
                 cq: cute.Tensor, ck: cute.Tensor, cv: cute.Tensor,
                 o: cute.Tensor, do: cute.Tensor, lse: cute.Tensor,
                 lengths: Optional[cute.Tensor], mask: Optional[cute.Tensor],
                 hq: cutlass.Constexpr, hk: cutlass.Constexpr,
                 scale: Float32, stream: cuda.CUstream):
        self.cast(dq, dk, dv, cq, ck, cv, stream)
        self.attention(q, k, v, cq, ck, cv, o, do, lse, lengths, mask, hq, hk, scale, stream)


_paired_compiled = {}


def _launch_pair(q, k, v, dq, dk, dv, o, do, lse, scale, lengths, mask,
                 block_n, prefetch, reuse_q, pipeline, cluster_size):
    xs = (dq, dk, dv)
    total = sum(x.numel() for x in xs if x.dtype != torch.bfloat16)
    chunks = 4 if 2**21 <= total <= 2**23 else 1
    converted = tuple(o if i == 0 and reuse_q else
                      x if x.dtype == torch.bfloat16 else torch.empty_like(x, dtype=torch.bfloat16)
                      for i, x in enumerate(xs))
    tensors = (q, k, v, dq, dk, dv, *converted, o, do, lse, lengths, mask)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (tuple(q.shape), tuple(k.shape), tuple(x.dtype for x in xs), q.device.index,
           block_n, prefetch, reuse_q, pipeline, cluster_size, chunks,
           lengths.dtype if lengths is not None else None, mask is not None)
    if key not in _paired_compiled:
        cs = tuple(from_dlpack(x, assumed_align=16, enable_tvm_ffi=True) if x is not None else None
                   for x in tensors)
        kernel_type = CuteJVP
        if pipeline:
            from .cute_pipeline import PipelinedJVP
            kernel_type = PipelinedJVP
        opts = {"cluster_size": cluster_size} if pipeline else {}
        op = kernel_type(q.shape[-1], block_n, prefetch, lengths is not None, mask is not None, **opts)
        _paired_compiled[key] = cute.compile(_CastThenJVP(chunks, op), *cs,
            q.shape[1], k.shape[1], Float32(scale), stream, options="--enable-tvm-ffi")
    _paired_compiled[key](*tensors, scale, stream)
    return o, do, lse


@torch.no_grad()
def attention_jvp_cute(q, k, v, dq, dk, dv, *, scale=None, key_mask=None,
                       key_lengths=None, key_bias=None, block_n=None, prefetch=True,
                       fused_cast=True, reuse_output=True, pipeline=None,
                       fused_launch=True, cluster_size=None):
    if key_bias is not None:
        raise NotImplementedError("Additive bias is not implemented")
    if any(x.requires_grad for x in (q, k, v, dq, dk, dv)):
        raise ValueError("Forward JVP only; detach inputs explicitly")
    if q.ndim != 4 or k.ndim != 4 or min(*q.shape, *k.shape) <= 0:
        raise ValueError("Expected positive [B,H,N,D] dimensions")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid GQA shapes")
    if block_n is None:
        block_n = 128 if q.shape[-1] == 64 else 64
    if q.shape[-1] not in (32, 64, 128) or block_n not in (32, 64, 128):
        raise ValueError("Initial CuTe supports D32/64/128, BN32/64/128")
    for x, dx in ((q, dq), (k, dk), (v, dv)):
        if x.dtype != torch.bfloat16 or dx.dtype not in (torch.bfloat16, torch.float16, torch.float32) or x.shape != dx.shape:
            raise ValueError("BF16 primals and matching tangents required")
        if any(not a.is_cuda or a.device != q.device or not a.is_contiguous() for a in (x, dx)):
            raise ValueError("Contiguous tensors on one CUDA device required")
    for x, shape, dtypes in ((key_lengths, (q.shape[0],), (torch.int32, torch.int64)),
                              (key_mask, (q.shape[0], k.shape[-2]), (torch.bool,))):
        if x is not None and (x.shape != shape or x.dtype not in dtypes or x.device != q.device or not x.is_contiguous()):
            raise ValueError("Invalid key_lengths or key_mask")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("CuTe backend currently targets Hopper SM90")
    if pipeline is None:
        pipeline = (prefetch and q.shape[-1] == 128 and block_n == 64
                    and k.shape[-2] >= 1024 and key_lengths is None
                    and q.shape[0] * q.shape[1] * ((q.shape[-2] + 127) // 128) >= 128)
    if pipeline and (q.shape[-1] != 128 or block_n != 64 or not prefetch):
        raise ValueError("Pipelined schedule requires D128, BN64 and prefetch")
    if cluster_size is None:
        cluster_size = (2 if pipeline and k.shape[-2] >= 8192 and q.shape[-2] >= 4096
                        and q.shape[0] * q.shape[1] * ((q.shape[-2] + 127) // 128) >= 512
                        and key_lengths is None and key_mask is None else 1)
    if cluster_size not in (1, 2) or (cluster_size > 1 and not pipeline):
        raise ValueError("Cluster size must be 1, or 2 with the pipelined schedule")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    o = torch.empty_like(q, dtype=torch.float32)
    do = torch.empty_like(o)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    reuse_q = fused_cast and reuse_output and dq.dtype != torch.bfloat16
    if fused_launch and fused_cast and any(x.dtype != torch.bfloat16 for x in (dq, dk, dv)):
        return _launch_pair(q, k, v, dq, dk, dv, o, do, lse, scale, key_lengths, key_mask,
                            block_n, prefetch, reuse_q, pipeline, cluster_size)
    dq, dk, dv = (_cast_tangents((dq, dk, dv), out_q=o if reuse_q else None) if fused_cast
                  else tuple(x.to(torch.bfloat16) for x in (dq, dk, dv)))
    ts = (q, k, v, dq, dk, dv, o, do)
    ls = lse
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (tuple(q.shape), tuple(k.shape), block_n, prefetch, reuse_q, pipeline, cluster_size, q.device.index,
           key_lengths.dtype if key_lengths is not None else None, key_mask is not None)
    if key not in _compiled:
        cs = tuple(from_dlpack(x, assumed_align=16, enable_tvm_ffi=True) if x is not None else None
                   for x in (*ts, ls, key_lengths, key_mask))
        kernel_type = CuteJVP
        if pipeline:
            from .cute_pipeline import PipelinedJVP
            kernel_type = PipelinedJVP
        opts = {"cluster_size": cluster_size} if pipeline else {}
        _compiled[key] = cute.compile(kernel_type(q.shape[-1], block_n, prefetch, key_lengths is not None, key_mask is not None, **opts), *cs,
                                      q.shape[1], k.shape[1], Float32(scale), stream,
                                      options="--enable-tvm-ffi")
    _compiled[key](*ts, ls, key_lengths, key_mask, scale, stream)
    return o, do, lse
