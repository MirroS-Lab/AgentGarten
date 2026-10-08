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

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup

from .cute_backend import CuteJVP, _logical_view


class PipelinedJVP(CuteJVP):
    def __init__(self, d, bn=64, prefetch=True, has_lens=False, has_mask=False,
                 cluster_size=1, splits=1, consumer_groups=2):
        super().__init__(d, bn, prefetch, has_lens, has_mask)
        self.bm = consumer_groups * 64
        self.cluster_size = cluster_size
        self.splits = splits
        self.consumer_groups = consumer_groups

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
                 dq: cute.Tensor, dk: cute.Tensor, dv: cute.Tensor,
                 o: cute.Tensor, do: cute.Tensor, lse: cute.Tensor,
                 lengths: Optional[cute.Tensor], mask: Optional[cute.Tensor],
                 hq: cutlass.Constexpr, hk: cutlass.Constexpr,
                 scale: Float32, stream: cuda.CUstream,
                 means: Optional[cute.Tensor] = None):
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
        if const_expr(self.splits > 1):
            means = cute.make_tensor(means.iterator, cute.make_layout(
                (means.shape[2], means.shape[0] * means.shape[1]), stride=(1, means.shape[2])))
        d, bm, bn = self.d, self.bm, self.bn
        qmma = sm90.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32, (self.consumer_groups, 1, 1), (64, bn))
        pmma = sm90.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            Float32, (self.consumer_groups, 1, 1), (64, d), warpgroup.OperandSource.RMEM)
        qlayout = cute.slice_(sm90.make_smem_layout_a(utils.LayoutEnum.ROW_MAJOR,
                            (bm, bn, d), cutlass.BFloat16, 1), (None, None, 0))
        klayout = sm90.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR,
                            (bm, bn, d), cutlass.BFloat16, 2)
        vlayout = sm90.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR,
                            (bm, d, bn), cutlass.BFloat16, 2)
        qa, qt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), q, qlayout, (bm, d))
        dqa, dqt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), dq, qlayout, (bm, d))
        kv_op = cpasync.CopyBulkTensorTileG2SOp()
        if const_expr(self.cluster_size > 1):
            # A single leader partitions the whole tile and broadcasts it.
            kv_op = cpasync.CopyBulkTensorTileG2SMulticastOp()
        ka, kt = cpasync.make_tiled_tma_atom(kv_op, k, cute.slice_(klayout, (None, None, 0)), (bn, d))
        dka, dkt = cpasync.make_tiled_tma_atom(kv_op, dk, cute.slice_(klayout, (None, None, 0)), (bn, d))
        va, vt = cpasync.make_tiled_tma_atom(kv_op, v, cute.slice_(vlayout, (None, None, 0)), (d, bn))
        dva, dvt = cpasync.make_tiled_tma_atom(kv_op, dv, cute.slice_(vlayout, (None, None, 0)), (d, bn))
        if const_expr(self.cluster_size > 1):
            self.kernel(qmma, pmma, qlayout, klayout, vlayout,
                        qa, qt, dqa, dqt, ka, kt, dka, dkt, va, vt, dva, dvt,
                        o, do, lse, means, lengths, mask, hq, hk, scale).launch(
                            grid=(cute.ceil_div(q.shape[0], bm * self.cluster_size) * self.cluster_size, q.shape[2], self.splits),
                            block=((self.consumer_groups + 1) * 128, 1, 1), cluster=(self.cluster_size, 1, 1), stream=stream)
        else:
            self.kernel(qmma, pmma, qlayout, klayout, vlayout,
                        qa, qt, dqa, dqt, ka, kt, dka, dkt, va, vt, dva, dvt,
                        o, do, lse, means, lengths, mask, hq, hk, scale).launch(
                            grid=(cute.ceil_div(q.shape[0], bm), q.shape[2], self.splits),
                            block=((self.consumer_groups + 1) * 128, 1, 1), stream=stream)

    @cute.jit
    def load_kv(self, atom, src, dst, barrier):
        if const_expr(self.cluster_size > 1):
            cute.copy(atom, src, dst, tma_bar_ptr=barrier,
                      mcast_mask=cutlass.Int16((1 << self.cluster_size) - 1))
        else:
            cute.copy(atom, src, dst, tma_bar_ptr=barrier)

    @cute.kernel
    def kernel(self, qmma: cute.TiledMma, pmma: cute.TiledMma,
               qlayout: cute.ComposedLayout, klayout: cute.ComposedLayout, vlayout: cute.ComposedLayout,
               qa: cute.CopyAtom, q: cute.Tensor, dqa: cute.CopyAtom, dq: cute.Tensor,
               ka: cute.CopyAtom, k: cute.Tensor, dka: cute.CopyAtom, dk: cute.Tensor,
               va: cute.CopyAtom, v: cute.Tensor, dva: cute.CopyAtom, dv: cute.Tensor,
               o: cute.Tensor, do: cute.Tensor, lse: cute.Tensor, means: Optional[cute.Tensor],
               lengths: Optional[cute.Tensor], mask: Optional[cute.Tensor],
               hq: cutlass.Constexpr, hk: cutlass.Constexpr, scale: Float32):
        tid, _, _ = cute.arch.thread_idx()
        block_m, bh, part = cute.arch.block_idx()
        output_bh = bh
        if const_expr(self.splits > 1):
            output_bh = bh + part * q.shape[2]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        kh = bh // hq * hk + (bh % hq) // (hq // hk)
        d, bm, bn = self.d, self.bm, self.bn
        nq, nk = q.shape[0], k.shape[0]
        length = nk
        if const_expr(self.has_lens):
            length = cutlass.min(nk, cutlass.max(0, lengths[bh // hq]))
        blocks = cutlass.max(1, cute.ceil_div(length, bn))
        first_block = 0
        if const_expr(self.splits > 1):
            part_blocks = cute.ceil_div(cute.ceil_div(nk, bn), self.splits)
            first_block = part * part_blocks
            last_block = cutlass.min(cute.ceil_div(length, bn), first_block + part_blocks)
            blocks = cutlass.max(1, last_block - first_block)
        smem = utils.SmemAllocator()
        sq = smem.allocate_tensor(cutlass.BFloat16, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sdq = smem.allocate_tensor(cutlass.BFloat16, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sk = smem.allocate_tensor(cutlass.BFloat16, klayout.outer, byte_alignment=128, swizzle=klayout.inner)
        sdk = smem.allocate_tensor(cutlass.BFloat16, klayout.outer, byte_alignment=128, swizzle=klayout.inner)
        sv = smem.allocate_tensor(cutlass.BFloat16, vlayout.outer, byte_alignment=128, swizzle=vlayout.inner)
        sdv = smem.allocate_tensor(cutlass.BFloat16, vlayout.outer, byte_alignment=128, swizzle=vlayout.inner)
        barriers = smem.allocate_array(cutlass.Int64, 5)
        if warp == 0:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(barriers + stage, 1)
                    cute.arch.mbarrier_init(barriers + 2 + stage, 128 * self.consumer_groups * self.cluster_size)
                cute.arch.mbarrier_init(barriers + 4, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if const_expr(self.cluster_size > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        qs, qg = self.parts(qa, sq, cute.local_tile(q, (bm, d), (None, 0, bh)))
        dqs, dqg = self.parts(dqa, sdq, cute.local_tile(dq, (bm, d), (None, 0, bh)))
        ks, kg = self.parts(ka, sk, cute.local_tile(k, (bn, d), (None, 0, kh)))
        dks, dkg = self.parts(dka, sdk, cute.local_tile(dk, (bn, d), (None, 0, kh)))
        vs, vg = self.parts(va, sv, cute.local_tile(v, (d, bn), (0, None, kh)))
        dvs, dvg = self.parts(dva, sdv, cute.local_tile(dv, (d, bn), (0, None, kh)))
        if tid < 128 * self.consumer_groups:
            cute.arch.warpgroup_reg_alloc(240)
            qthr = qmma.get_slice(tid)
            pthr = pmma.get_slice(tid)
            aq = qmma.make_fragment_A(qthr.partition_A(sq))
            adq = qmma.make_fragment_A(qthr.partition_A(sdq))
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
            og = self.mn(pthr.partition_C(cute.local_tile(o, (bm, d), (block_m, 0, output_bh))), pmma)
            dog = self.mn(pthr.partition_C(cute.local_tile(do, (bm, d), (block_m, 0, output_bh))), pmma)
            cute.arch.mbarrier_wait(barriers + 4, 0)
            for nb in range(blocks):
                stage = nb % 2
                cute.arch.mbarrier_wait(barriers + stage, (nb // 2) % 2)
                bk = qmma.make_fragment_B(qthr.partition_B(sk[None, None, stage]))
                bdk = qmma.make_fragment_B(qthr.partition_B(sdk[None, None, stage]))
                bv = pmma.make_fragment_B(pthr.partition_B(sv[None, None, stage]))
                bdv = pmma.make_fragment_B(pthr.partition_B(sdv[None, None, stage]))
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
                        column = scoord[i, j][1] + (first_block + nb) * bn
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
                    if rowmax[i] != newmax:
                        alpha = cute.math.exp2(rowmax[i] - safe_max, fastmath=True)
                        rowsum[i] *= alpha
                        rowds[i] *= alpha
                        for j in cutlass.range_constexpr(cute.size(omn, mode=[1])):
                            omn[i, j] *= alpha
                            domn[i, j] *= alpha
                    rowmax[i] = newmax
                    for j in cutlass.range_constexpr(cute.size(smn, mode=[1])):
                        p = cute.math.exp2(smn[i, j] - safe_max, fastmath=True)
                        smn[i, j] = p
                        rowsum[i] += p
                # Both score groups must finish before dscore use and K/dK overwrite.
                warpgroup.wait_group(0)
                for i in cutlass.range_constexpr(cute.size(smn, mode=[0])):
                    for j in cutlass.range_constexpr(cute.size(smn, mode=[1])):
                        column = scoord[i, j][1] + (first_block + nb) * bn
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
                warpgroup.fence()
                self.mma(pmma, p, bv, out, False)
                self.mma(pmma, p, bdv, dout, False)
                self.mma(pmma, pd, bv, dout, False)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                if const_expr(self.cluster_size > 1):
                    # The leader reuses a stage only after every CTA consumes it.
                    cute.arch.mbarrier_arrive(barriers + 2 + stage, peer_cta_rank_in_cluster=0)
                else:
                    cute.arch.mbarrier_arrive(barriers + 2 + stage)
            for i in cutlass.range_constexpr(cute.size(omn, mode=[0])):
                norm = cute.arch.warp_reduction_sum(rowsum[i], threads_in_group=4)
                inv = Float32(1.0)
                if norm > 0:
                    inv = 1.0 / norm
                mean = cute.arch.warp_reduction_sum(rowds[i], threads_in_group=4) * inv
                for j in cutlass.range_constexpr(cute.size(omn, mode=[1])):
                    val = omn[i, j] * inv
                    omn[i, j] = val
                    if const_expr(self.splits > 1):
                        domn[i, j] = domn[i, j] * inv
                    else:
                        domn[i, j] = domn[i, j] * inv - val * mean
                if oc[i, 0][0] + block_m * bm < nq:
                    cute.autovec_copy(omn[i, None], og[i, None])
                    cute.autovec_copy(domn[i, None], dog[i, None])
                if oc[i, 0][1] == 0 and oc[i, 0][0] + block_m * bm < nq:
                    if const_expr(self.splits > 1):
                        means[oc[i, 0][0] + block_m * bm, output_bh] = mean
                    lse[oc[i, 0][0] + block_m * bm, output_bh] = (rowmax[i] + cute.math.log2(norm, fastmath=True)) * 0.6931471805599453
        else:
            cute.arch.warpgroup_reg_dealloc(24)
            if warp == self.consumer_groups * 4:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(barriers + 4, 2 * bm * d * 2)
                cute.copy(qa, qg[None, block_m], qs, tma_bar_ptr=barriers + 4)
                cute.copy(dqa, dqg[None, block_m], dqs, tma_bar_ptr=barriers + 4)
                if block_m % self.cluster_size == 0:
                    for nb in range(blocks):
                        stage = nb % 2
                        if nb >= 2:
                            cute.arch.mbarrier_wait(barriers + 2 + stage, ((nb // 2) - 1) % 2)
                        with cute.arch.elect_one():
                            if const_expr(self.cluster_size > 1):
                                for peer in cutlass.range_constexpr(self.cluster_size):
                                    cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, 4 * bn * d * 2,
                                                                          peer_cta_rank_in_cluster=peer)
                            else:
                                cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, 4 * bn * d * 2)
                        self.load_kv(ka, kg[None, first_block + nb], ks[None, stage], barriers + stage)
                        self.load_kv(dka, dkg[None, first_block + nb], dks[None, stage], barriers + stage)
                        self.load_kv(va, vg[None, first_block + nb], vs[None, stage], barriers + stage)
                        self.load_kv(dva, dvg[None, first_block + nb], dvs[None, stage], barriers + stage)
        if const_expr(self.cluster_size > 1):
            # A peer can still reference this CTA's barriers until all finish.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
