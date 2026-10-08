/*
 * Copyright (c) 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused LoRA apply, split-kernel edition. Replaces the single
 * per-token kernel: at small batch (decode) one-token-per-block left most of
 * the 40 AIVs idle; this split fills the machine at any B and degrades to
 * the v1 mapping (one token per block) once B >= core count.
 *
 *   z1 kernel: z1[s][t][r] = scale * sum_h x[t,h] * A_s[slot,r,h]
 *     grid: (slice, token, rank-group) units; each block streams its rank
 *     group's A rows over H1 tiles with the x tile cast once per tile.
 *     z1 is staged to a GM fp32 workspace [S][B][R] (32B-aligned writes).
 *   z2 kernel: y[t, off+j] (+)= sum_r z1[s][t][r] * B_s[slot,j,r]
 *     grid: (token, chunk) units over the CONCATENATED output range of up to
 *     4 slices (qkv / gate_up / o_proj / qkvz); chunks are multiples of the
 *     W-tile quantum (8192/R) so tiles never split across blocks; a chunk may
 *     straddle slice boundaries (handled by per-slice intersection). z2 ports
 *     bgmv_expand's repeat-mask structure (dup z1 across a 256B repeat, Mul
 *     with src0RepStride=0, Block/PairReduceSum tree).
 *
 * Constraints (enforced by add_lora_eligible in torch_binding.cpp):
 *   - R in {16, 32, 64}: must divide 64 (z2 replicates z1 into a
 *     64-element repeat; also selects the reduce-tree shape below) and be a
 *     multiple of 16 (GM copy alignment)
 *   - H1 / every H2_s / y width are multiples of 16; <= 4 slices;
 *     offset_start == 0 (the kernels address y from column 0)
 *   - indices are int64; -1 skips the token (overwrite mode zeroes it)
 * Native fp16/bf16 (template T), fp32 accumulate, deterministic.
 * UB budget: z1 ~72KB, z2 ~112KB (192KB per AIV on Ascend910B1-4).
 *
 * EDITING NOTE: these kernels are parsed by the build's auto_gen tool —
 * pointer parameters MUST be written `__gm__ void* name` (star on the type,
 * not the name) or the generated launchers dereference them; and any kernel
 * signature change requires wiping the generated chain (build/.../auto_gen,
 * include/vllm_ascend_kernels, the *_precompile/preprocess/aic/aiv
 * device-prefix dirs) or stale launchers poison the rebuild.
 */

#include "kernel_operator.h"
#include "types.h"

namespace {
constexpr uint32_t TILE_H = 4096;               // x / A-row tile (z1 phase)
constexpr uint32_t Z1_RANK_SUB = 4;             // batch size for MTE2 load
constexpr uint32_t Z1_RANK_SUB_W = 8;           // EXP-29: batchEw W sub-block rows
constexpr uint32_t W_IN_TILE = 8192;            // B elements per z2 compute tile
constexpr uint32_t W_IN_TILE_BIG = 15360;       // EXP-35: G2 W tile, 240 reps
                                                // (2201 repeat field is uint8_t: max 255)
constexpr uint32_t Y_OUT_TILE = 4096;           // max outputs per block (tmpY_ size)
constexpr uint32_t NUM_BYTES_PER_REPEAT = 256;  // vector unit read granularity
constexpr uint32_t NUM_BLOCKS_PER_REPEAT = 8;
constexpr uint32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);
constexpr uint32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE / NUM_ELEMENTS_PER_REPEAT;
constexpr uint32_t PAIR_REDUCE_NUM_REPEATS_16 =
    (BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1)
    / NUM_ELEMENTS_PER_REPEAT;
constexpr uint32_t PAIR_REDUCE_NUM_REPEATS_32 = (PAIR_REDUCE_NUM_REPEATS_16 + 1) / 2;
constexpr uint32_t R_MAX = 64;
// reduce-tree strides (values mirror bgmv_expand's reduceSumParams):
// dst advances one 32B block per repeat; src advances 8 blocks (= 256B repeat)
constexpr uint16_t REDUCE_DST_REP_STRIDE = 1;
constexpr uint16_t REDUCE_SRC_BLK_STRIDE = 1;
constexpr uint16_t REDUCE_SRC_REP_STRIDE = 8;
}  // namespace

// ============================ kernel 1: z1 =================================
template <typename scalar_t>
class AddLoraZ1 {
public:
    using T = scalar_t;

    __aicore__ inline AddLoraZ1(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *x, __gm__ void *wa0, __gm__ void *wa1,
                                __gm__ void *wa2, __gm__ void *wa3, __gm__ void *indices,
                                __gm__ void* z1out, uint32_t batch, uint32_t units,
                                uint32_t unitsPerCore, uint32_t H1, uint32_t R,
                                uint32_t rankBlock, float scale)
    {
        batch_ = batch;
        units_ = units;
        unitsPerCore_ = unitsPerCore;
        H1_ = H1;
        R_ = R;
        rankBlock_ = rankBlock;
        scale_ = scale;
        wa_[0].SetGlobalBuffer((__gm__ T *)wa0);
        wa_[1].SetGlobalBuffer((__gm__ T *)wa1);
        wa_[2].SetGlobalBuffer((__gm__ T *)wa2);
        wa_[3].SetGlobalBuffer((__gm__ T *)wa3);
        groups_ = (R_ + rankBlock_ - 1) / rankBlock_;

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, batch);
        z1Gm_.SetGlobalBuffer((__gm__ float *)z1out);

        // EXP-29: the batchEw path carries W sub-blocks of 8 rows of width
        // H1_ <= 1024, so its queue/TBuf sizing is runtime-computed and
        // SMALLER than the old worst-case (G2: 4KB nodes vs 32KB, 8KB tmpWA_
        // vs 16KB). Non-batchEw sizing is byte-identical to before.
        const bool batchEwInit = (H1_ <= 1024);
        pipe_->InitBuffer(inQueueX_, 1, TILE_H * sizeof(T));
        // EXP-8: depth 2 so sub-block k+1 can be in flight while k computes
        pipe_->InitBuffer(inQueueWA_, 2,
                          (batchEwInit ? Z1_RANK_SUB_W * H1_
                                       : Z1_RANK_SUB * TILE_H) * sizeof(T));
        pipe_->InitBuffer(tmpX_, TILE_H * sizeof(float));
        pipe_->InitBuffer(tmpWA_,
                          (batchEwInit ? Z1_RANK_SUB_W * H1_ : TILE_H) * sizeof(float));
        // EXP-3: stage is (r1-r0) 32B-strided slots (element 8r valid) so the
        // per-row dot result can be accumulated entirely in the V pipe.
        pipe_->InitBuffer(z1Stage_, R_MAX * 8 * sizeof(float));
        pipe_->InitBuffer(z1Out_, R_MAX * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        // EXP-25: for small-H1 geoms (arena G2: H1=256) the elementwise
        // Cast/Mul prologue of each 4-row A block runs as ONE Cast + ONE Mul
        // over 4n elements instead of 4+4 per-row instructions -- the scalar
        // pipe pays per issued V instruction (EXP-7: the ALU itself is free).
        const bool batchEw = (H1_ <= 1024);
        // EXP-29: batchEw stacks 8 W rows per queue round trip (halves the
        // scalar Alloc/EnQue/DeQue/Free bookkeeping and the W copy count);
        // non-batchEw keeps 4 so the G1 code path is identical to before.
        const uint32_t rankSubW = batchEw ? Z1_RANK_SUB_W : Z1_RANK_SUB;
        int64_t end = (int64_t)(blockIdx + 1) * unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }
        for (int64_t u = (int64_t)blockIdx * unitsPerCore_; u < end; ++u) {
            // unit -> (slice, token, rank group); units are planned only for
            // ACTIVE slices (h2 != 0), so s < nSlices <= 4 always holds
            uint32_t s = (uint32_t)(u / ((int64_t)batch_ * groups_));
            int64_t rem = u % ((int64_t)batch_ * groups_);
            int64_t t = rem / groups_;
            uint32_t g = (uint32_t)(rem % groups_);
            int64_t slot = indicesGm_.GetValue(t);
            if (slot < 0) {
                continue;  // no-lora token: z2 skips it too
            }
            uint32_t r0 = g * rankBlock_;
            uint32_t r1 = r0 + rankBlock_;
            if (r1 > R_) {
                r1 = R_;
            }
            AscendC::LocalTensor<float> stage = z1Stage_.Get<float>();
            if (!batchEw) {
                Duplicate(stage, 0.0f, (r1 - r0) * 8);  // EXP-3: one V op, was a
                                                        // scalar SetValue loop
            }
            // EXP-26: batchEw skips the seed -- H1_ <= 1024 <= TILE_H means
            // a single h iteration, and ReduceSum writes each row's sum
            // straight into its slot (nothing to accumulate onto).
            const int64_t aBase = slot * (int64_t)R_ * H1_;
            for (uint32_t h0 = 0; h0 < H1_; h0 += TILE_H) {
                uint32_t n = (H1_ - h0 < TILE_H) ? (H1_ - h0) : TILE_H;
                AscendC::LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
                if (batchEw) {
                    // EXP-25: tile the x row Z1_RANK_SUB times into one node
                    // (same GM row re-read; MTE2 is the idle pipe on G2) so
                    // the block-wide Mul below has a per-row-matching operand
                    for (uint32_t j = 0; j < Z1_RANK_SUB; ++j) {
                        DataCopy(xLocal[j * n], xGm_[(int64_t)t * H1_ + h0], n);
                    }
                } else {
                    DataCopy(xLocal, xGm_[(int64_t)t * H1_ + h0], n);
                }
                inQueueX_.EnQue(xLocal);
                xLocal = inQueueX_.DeQue<T>();
                AscendC::LocalTensor<float> xF = tmpX_.Get<float>();
                AscendC::LocalTensor<float> wF = tmpWA_.Get<float>();
                Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE,
                     batchEw ? Z1_RANK_SUB * n : n);
                inQueueX_.FreeTensor(xLocal);
                // EXP-8: 2-deep pipeline over the A sub-blocks. The load of
                // sub-block k+1 is issued BEFORE the blocking DeQue of block
                // k, so its MTE2 read is in flight during block k's V chain
                // (EXP-4 tree: depth-1 queue + adjacent EnQue/DeQue left
                // every A load fully exposed on the serial chain).
                const uint32_t nSub = (r1 - r0 + rankSubW - 1) / rankSubW;
                {  // seed: issue block 0
                    uint32_t rbI = r0;
                    uint32_t subI = (r1 - rbI < rankSubW) ? (r1 - rbI) : rankSubW;
                    AscendC::LocalTensor<T> wIssued = inQueueWA_.AllocTensor<T>();
                    if (H1_ <= TILE_H) {
                        DataCopy(wIssued, wa_[s][aBase + (int64_t)rbI * H1_], subI * n);
                    } else {
                        // we are copying sub rows, but each row is longer than TILE_H
                        for (uint32_t i = 0; i < subI; ++i)
                            DataCopy(wIssued[i * n], wa_[s][aBase + (int64_t)(rbI + i) * H1_ + h0], n);
                    }
                    inQueueWA_.EnQue(wIssued);
                }
                for (uint32_t k = 0; k < nSub; ++k) {
                    if (k + 1 < nSub) {  // issue block k+1 ahead of the wait
                        uint32_t rbI = r0 + (k + 1) * rankSubW;
                        uint32_t subI = (r1 - rbI < rankSubW) ? (r1 - rbI) : rankSubW;
                        AscendC::LocalTensor<T> wIssued = inQueueWA_.AllocTensor<T>();
                        if (H1_ <= TILE_H) {
                            DataCopy(wIssued, wa_[s][aBase + (int64_t)rbI * H1_], subI * n);
                        } else {
                            for (uint32_t i = 0; i < subI; ++i)
                                DataCopy(wIssued[i * n], wa_[s][aBase + (int64_t)(rbI + i) * H1_ + h0], n);
                        }
                        inQueueWA_.EnQue(wIssued);
                    }
                    AscendC::LocalTensor<T> wLocal = inQueueWA_.DeQue<T>();
                    const uint32_t rbUse = r0 + k * rankSubW;
                    const uint32_t subUse =
                        (r1 - rbUse < rankSubW) ? (r1 - rbUse) : rankSubW;
                    if (batchEw) {
                        // EXP-25: one Cast + one Mul across the whole row
                        // block. Elementwise, so every product is bit-exact
                        // what the per-row pair below computes; only the
                        // instruction count (4+4 -> 2) drops.
                        // EXP-29: the block is 8 rows now, so ONE Cast spans
                        // all 8n products; the block-wide Mul cannot use the
                        // 4-tiled xF as a single 8n operand, so it runs as
                        // two 4n halves on the SAME tiles -- every row
                        // multiplies the same x row, so element k of rows
                        // 4..7 pairs with xF tile k%n exactly as rows 0..3
                        // do. Bit-exact vs the EXP-25 pair-of-4 by the same
                        // elementwise argument.
                        Cast(wF, wLocal, AscendC::RoundMode::CAST_NONE, subUse * n);
                        const uint32_t nH0 = (subUse < 4) ? subUse : 4;
                        Mul(wF, xF, wF, nH0 * n);
                        if (subUse > 4) {
                            Mul(wF[nH0 * n], xF, wF[nH0 * n], (subUse - nH0) * n);
                        }
                    }
                    for (uint32_t i = 0; i < subUse; ++i) {
                        // EXP-2: the Cast/Mul/ReduceSum chain is all V-pipe
                        // with the compiler inserting same-pipe sync (0016:
                        // "PIPE_V由编译器自动完成同步插入"); the manual
                        // drains between them were removed. EXP-3: the
                        // accumulate is now a V-pipe Add too, so no barrier
                        // remains inside the row loop at all.
                        AscendC::LocalTensor<float> wRow =
                            batchEw ? wF[i * n] : wF;
                        if (batchEw) {
                            // EXP-27: element-stride dst (EXP-26's direct-to-
                            // slot reduced further by dropping the *8): dot j
                            // lands at stage[j], its 32B dst block spanning
                            // [j, j+8). Blocks overlap by 7; same-pipe in-order
                            // issue makes block j's own dst[0] write the LAST
                            // writer of element j (block i touches only
                            // [i, i+8)), so stage[0..r1-r0-1] ends up holding
                            // the dots CONTIGUOUS -- no drain, no compaction.
                            ReduceSum<float>(stage[rbUse + i - r0], wRow, wRow,
                                             n);
                        } else {
                            AscendC::LocalTensor<float> slot =
                                stage[(rbUse + i - r0) * 8];
                            Cast(wF, wLocal[i * n], AscendC::RoundMode::CAST_NONE, n);
                            Mul(wF, xF, wF, n);
                            ReduceSum<float>(wRow, wRow, wRow, n);
                            // EXP-3: accumulate wF[0] into this row's 32B-
                            // aligned slot in the V pipe (one block; elements
                            // 1..7 of both srcs are don't-care padding). No
                            // barrier, no V->S->UB round trip.
                            Add(slot, slot, wRow, 8);
                        }
                    }
                    inQueueWA_.FreeTensor(wLocal);
                }
            }
            if (batchEw) {
                // EXP-27: dots already contiguous in stage[0..r1-r0) (see
                // row loop). Fold scale as ONE vector Muls at the 32B-
                // aligned stage base (fp32 mult == the scalar fp32 mult the
                // EXP-3 drain used, bit-exact) and write GM straight from
                // stage. The only barrier left orders V -> MTE2; the pre-
                // drain barrier is gone because Muls is same-pipe as the
                // ReduceSum chain.
                Muls(stage, stage, scale_, r1 - r0);
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(z1Gm_[((int64_t)s * batch_ + t) * R_ + r0], stage,
                         r1 - r0);
            } else {
                // EXP-3: one V->S drain, then a single scalar pass folds
                // scale (fp32, same rounding as the v1 vector Muls) and
                // compacts the 8-spaced slots into the contiguous GM write
                // (32B-aligned: rankBlock is a multiple of 8 and R % 16 == 0)
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<float> outb = z1Out_.Get<float>();
                for (uint32_t r = 0; r < (r1 - r0); ++r) {
                    outb.SetValue(r, stage.GetValue(r * 8) * scale_);
                }
                AscendC::PipeBarrier<PIPE_V>();  // baseline-proven tail:
                                                // scalar stores -> UB->GM
                                                // DataCopy
                DataCopy(z1Gm_[((int64_t)s * batch_ + t) * R_ + r0], outb,
                         r1 - r0);
            }
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueWA_;  // EXP-8: 2-deep
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpX_, tmpWA_, z1Stage_, z1Out_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> wa_[4];
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<float> z1Gm_;
    uint32_t batch_, units_, unitsPerCore_, H1_, R_, rankBlock_, groups_;
    float scale_;
};

// ============================ kernel 2: z2 =================================
template <typename scalar_t>
class AddLoraZ2 {
public:
    using T = scalar_t;

    __aicore__ inline AddLoraZ2(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *z1in, __gm__ void *wb0, __gm__ void *wb1,
                                __gm__ void *wb2, __gm__ void *wb3, __gm__ void *indices,
                                __gm__ void *y, uint32_t batch, uint32_t units,
                                uint32_t unitsPerCore, uint32_t R, uint32_t h2_0,
                                uint32_t h2_1, uint32_t h2_2, uint32_t h2_3, uint32_t chunk,
                                uint32_t chunksPerToken, uint32_t yWidth, uint32_t addInputs,
                                uint32_t wTile)
    {
        batch_ = batch;
        units_ = units;
        unitsPerCore_ = unitsPerCore;
        R_ = R;
        chunk_ = chunk;
        chunksPerToken_ = chunksPerToken;
        yWidth_ = yWidth;
        addInputs_ = addInputs;
        // EXP-35: host selects the W-tile cap (15360 for the single-slice G2
        // shape, 8192 everywhere else); 8192 reproduces today's constants.
        wTile_ = wTile;
        wReps_ = wTile / NUM_ELEMENTS_PER_REPEAT;
        h2_[0] = h2_0;
        h2_[1] = h2_1;
        h2_[2] = h2_2;
        h2_[3] = h2_3;
        offs_[0] = 0;
        offs_[1] = h2_0;
        offs_[2] = h2_0 + h2_1;
        offs_[3] = h2_0 + h2_1 + h2_2;
        totalH2_ = offs_[3] + h2_[3];
        wb_[0].SetGlobalBuffer((__gm__ T *)wb0);
        wb_[1].SetGlobalBuffer((__gm__ T *)wb1);
        wb_[2].SetGlobalBuffer((__gm__ T *)wb2);
        wb_[3].SetGlobalBuffer((__gm__ T *)wb3);

        z1Gm_.SetGlobalBuffer((__gm__ float *)z1in);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, batch);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        // DEBUG EXP-B: restore the 256B slab that the old z1Buf_ occupied, so
        // every downstream UB buffer keeps its baseline address.
        dupSel_ = 0;
        pipe_->InitBuffer(debugPad_, R_MAX * sizeof(float));
        for (uint32_t k = 0; k < 4; ++k) {
            pipe_->InitBuffer(dupBufs_[k], NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        }
        // EXP-18: depth 2 (was 1) - DoRange issues the next tile's copy
        // before DeQue-ing the current one, compute(i) covers copy(i+1).
        pipe_->InitBuffer(inQueueW_, 2, wTile_ * sizeof(T));
        pipe_->InitBuffer(tmpW_, wTile_ * sizeof(float));
        pipe_->InitBuffer(inQueueY_, 1, Y_OUT_TILE * sizeof(T));
        pipe_->InitBuffer(outQueueY_, 1, Y_OUT_TILE * sizeof(T));
        // +32 floats slack: a remainder PairReduce writes in 32-output
        // granularity; at dstOff > 0 that round-up can pass the union end.
        pipe_->InitBuffer(tmpY_, (Y_OUT_TILE + 32) * sizeof(float));
        pipe_->InitBuffer(yInF_, Y_OUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t end = (int64_t)(blockIdx + 1) * unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }
        for (int64_t u = (int64_t)blockIdx * unitsPerCore_; u < end; ++u) {
            int64_t t = u / chunksPerToken_;
            uint32_t c = (uint32_t)(u % chunksPerToken_);
            // EXP-37: reuse the index read if the previous unit's prefetch
            // already did it -- the cross-unit seed MOVES this GM read
            // earlier, it never adds one.
            int64_t slot = (u == pfUnit_) ? pfSlot_ : indicesGm_.GetValue(t);
            uint32_t gs = c * chunk_;
            uint32_t ge = gs + chunk_;
            if (ge > totalH2_) {
                ge = totalH2_;
            }
            const uint64_t yRow = (uint64_t)t * yWidth_;
            if (slot < 0) {
                // no-lora token: accumulate leaves y untouched; overwrite
                // zeroes this block's chunk (EXP-15: the per-slice
                // intersections partition [gs, ge), so one ZeroRange over
                // the clipped chunk writes exactly what they did).
                if (!addInputs_ && ge > gs) {
                    ZeroRange(yRow + gs, ge - gs);
                }
                continue;
            }
            if (ge <= gs) {
                continue;
            }
            // EXP-15: ONE chain per token instead of one per slice
            // intersection. Each slice's dot writes its disjoint half of
            // tmpY (dstOff = a - gs); the y copy/tail runs once over the
            // union [gs, ge) instead of once per intersection.
            if (addInputs_) {
                CopyInY(yRow + gs, ge - gs);
            }
            for (uint32_t s = 0; s < 4; ++s) {
                uint32_t a, b;
                if (IntersectSlice(s, gs, ge, a, b)) {
                    DoRange(t, s, slot, a - offs_[s], b - offs_[s], a - gs);
                }
            }
            // EXP-37: cross-unit seed -- issue unit u+1's dup fill and first
            // W tile BEFORE this unit's tail so the 30,720 B G2 seed arrives
            // under ScaleOutput's V work + CopyOut's MTE3 push instead of
            // exposing itself at u+1's head drain.
            if (u + 1 < end) {
                PrefetchNext(u + 1);
            }
            ScaleOutput(ge - gs);
            CopyOut(yRow + gs, ge - gs);
        }
    }

private:
    // intersect the chunk [gs, ge) with slice s's range; empty for padding
    // slices (h2 == 0 gives b == offs_[s] <= a) — no separate skip needed
    __aicore__ inline bool IntersectSlice(uint32_t s, uint32_t gs, uint32_t ge, uint32_t &a,
                                           uint32_t &b)
    {
        a = gs > offs_[s] ? gs : offs_[s];
        b = ge < (offs_[s] + h2_[s]) ? ge : (offs_[s] + h2_[s]);
        return a < b;
    }

    // EXP-37: seed unit u2's dup slot + first W tile early. Coordinate and
    // seed-size arithmetic mirror Process/DoRange exactly, so DoRange(u2)'s
    // first intersection consumes the pair via (pfT_, pfS_) instead of
    // issuing it: per-pipe instruction SEQUENCE and data are unchanged, only
    // issue timing moves ahead of the previous unit's tail. The dup ring
    // cadence SEQUENCE is preserved (rotation order identical); the slot this
    // overwrites was last read 3 PrepareZ1s ago, past that ComputeTile's
    // PIPE_V barrier (L2/EXP-1 invariant). inQueueW_ is empty here (last
    // tile freed inside DoRange(u)'s loop), so the depth-2 EXP-18 invariant
    // holds on entry to DoRange(u2).
    __aicore__ inline void PrefetchNext(int64_t u2)
    {
        pfUnit_ = -1;
        pfT_ = -1;
        int64_t t2 = u2 / chunksPerToken_;
        uint32_t c2 = (uint32_t)(u2 % chunksPerToken_);
        uint32_t gs2 = c2 * chunk_;
        uint32_t ge2 = gs2 + chunk_;
        if (ge2 > totalH2_) {
            ge2 = totalH2_;
        }
        if (ge2 <= gs2) {
            return;
        }
        int64_t slot2 = indicesGm_.GetValue(t2);
        pfUnit_ = u2;
        pfSlot_ = slot2;
        if (slot2 < 0) {
            return;  // no-lora unit: cache the read, seed nothing
        }
        for (uint32_t s2 = 0; s2 < 4; ++s2) {
            uint32_t a2, b2;
            if (!IntersectSlice(s2, gs2, ge2, a2, b2)) {
                continue;
            }
            uint32_t la2 = a2 - offs_[s2];
            uint32_t lb2 = b2 - offs_[s2];
            PrepareZ1(t2, s2);
            const uint32_t outPerTile = wTile_ / R_;
            const uint32_t len2 = lb2 - la2;
            uint32_t nFull2 = len2 / outPerTile;
            uint32_t rem2 = len2 - nFull2 * outPerTile;
            CopyInW(slot2, s2, la2 * R_,
                    (nFull2 > 0 ? (int32_t)wTile_ : (int32_t)(rem2 * R_)));
            pfT_ = t2;
            pfS_ = s2;
            break;
        }
    }

    // z1 workspace is [S][B][R]: each slice intersection reads its own row,
    // then replicates it into one 256B repeat for the repeat-mask dot.
    // The replication is (256B/rowBytes) DataCopys of the SAME GM row into
    // successive slots of the dup buffer -- the row is L2-hot after the first
    // copy, and this moves the 64*R scalar UB accesses onto MTE2, which has
    // headroom.
    //
    // ADDRESS ROTATION IS LOAD-BEARING (measured, LOG EXP-1): the dot's Mul
    // reads dup as a broadcast operand (src0RepStride=0, the same 256B read
    // every repeat), and the V operand cache is NOT invalidated when MTE2
    // rewrites the SAME UB address -- a second PrepareZ1 on a core then
    // multiplies stale z1 data. Scalar stores do snoop that cache, which is
    // why the original scalar dup loop was safe. Each PrepareZ1 hands out the
    // next slot of a 4-buffer ring so rewritten dup data is always at a fresh
    // address; single-slot or depth-2 aliasing reproduced the corruption.
    __aicore__ inline void PrepareZ1(int64_t t, uint32_t s)
    {
        dupSel_ = (dupSel_ + 1u) & 3u;
        AscendC::LocalTensor<float> dup = dupBufs_[dupSel_].Get<float>();
        AscendC::GlobalTensor<float> row = z1Gm_[((int64_t)s * batch_ + t) * R_];
        for (uint32_t i = 0; i < NUM_ELEMENTS_PER_REPEAT; i += R_) {
            DataCopy(dup[i], row, R_);
        }
        // EXP-1b: the PIPE_MTE2 drain moved to DoRange, after the first
        // W-tile copy is issued, so the exposed latency is max(dup, W)
        // instead of dup + W. ComputeTile still runs after the drain.
    }

    __aicore__ inline void CopyInW(int64_t slot, uint32_t slice, uint32_t wElemOff,
                                   int32_t numElements)
    {
        AscendC::LocalTensor<T> wLocal = inQueueW_.AllocTensor<T>();
        const uint64_t slotOff = (uint64_t)slot * h2_[slice] * R_;
        DataCopy(wLocal, wb_[slice][slotOff + wElemOff], numElements);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void CopyInY(uint32_t yAddr, int32_t numElements)
    {
        AscendC::LocalTensor<T> yLocal = inQueueY_.AllocTensor<T>();
        DataCopy(yLocal, yGm_[yAddr], numElements);
        inQueueY_.EnQue(yLocal);
    }

    // dot(dup vs one B tile) -> yTile[dstOff + progress...]; bgmv_expand Compute
    __aicore__ inline void ComputeTile(int32_t progress, uint32_t dstOff,
                                       int32_t blockReduceRepeatCount = BLOCK_REDUCE_NUM_REPEATS,
                                       int32_t pairReduceRepeat16 = PAIR_REDUCE_NUM_REPEATS_16,
                                       int32_t pairReduceRepeat32 = PAIR_REDUCE_NUM_REPEATS_32)
    {
        AscendC::LocalTensor<float> yLocal = tmpY_.Get<float>();
        AscendC::LocalTensor<float> dup = dupBufs_[dupSel_].Get<float>();
        AscendC::LocalTensor<T> wLocal = inQueueW_.DeQue<T>();
        AscendC::LocalTensor<float> wTmp = tmpW_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, NUM_ELEMENTS_PER_REPEAT, blockReduceRepeatCount,
             castParams_);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueW_.FreeTensor(wLocal);

        Mul(wTmp, dup, wTmp, NUM_ELEMENTS_PER_REPEAT, blockReduceRepeatCount, dotProductParams_);
        AscendC::PipeBarrier<PIPE_V>();

        if (R_ == 16) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[dstOff + progress], wTmp, pairReduceRepeat16,
                          NUM_ELEMENTS_PER_REPEAT, REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (R_ == 32) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(wTmp, wTmp, pairReduceRepeat16, NUM_ELEMENTS_PER_REPEAT,
                          REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[dstOff + progress], wTmp, pairReduceRepeat32,
                          NUM_ELEMENTS_PER_REPEAT, REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        } else {  // R_ == 64
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            BlockReduceSum(yLocal[dstOff + progress], wTmp, pairReduceRepeat16,
                           NUM_ELEMENTS_PER_REPEAT, REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                           REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // y[t, yAddr..yAddr+len) (+)= tmpY (cast fp32 -> T)
    __aicore__ inline void ScaleOutput(int32_t numElements)
    {
        AscendC::LocalTensor<float> yLocal = tmpY_.Get<float>();
        if (addInputs_) {
            AscendC::LocalTensor<T> yInLocal = inQueueY_.DeQue<T>();
            AscendC::LocalTensor<float> yInF = yInF_.Get<float>();
            Cast(yInF, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueY_.FreeTensor(yInLocal);
            Add(yLocal, yLocal, yInF, numElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::LocalTensor<T> yOutLocal = outQueueY_.AllocTensor<T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue(yOutLocal);
    }

    __aicore__ inline void CopyOut(uint32_t yAddr, int32_t numElements)
    {
        AscendC::LocalTensor<T> yOutLocal = outQueueY_.DeQue<T>();
        DataCopy(yGm_[yAddr], yOutLocal, numElements);
        outQueueY_.FreeTensor(yOutLocal);
    }

    __aicore__ inline void ZeroRange(uint32_t yAddr, uint32_t len)
    {
        AscendC::LocalTensor<T> yOut = outQueueY_.AllocTensor<T>();
        Duplicate(yOut, (T)0, len);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue(yOut);
        yOut = outQueueY_.DeQue<T>();
        DataCopy(yGm_[yAddr], yOut, len);
        outQueueY_.FreeTensor(yOut);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // Process outputs [la, lb) of slice `s` into tmpY[dstOff...], where
    // dstOff = la's position in the token's chunk union (EXP-15: the y
    // copy and the fp32->T tail moved to Process, once per token).
    // Unified full-tile + remainder path (merges v1's main loop and
    // ComputeLastIteration); start offsets are multiples of 16 (chunk and
    // slice boundaries), W reads at la*R elements are 32B-aligned, and the
    // remainder W count rem*R is a multiple of 64 (rem % 16, R % 16).
    __aicore__ inline void DoRange(int64_t t, uint32_t s, int64_t slot, uint32_t la,
                                   uint32_t lb, uint32_t dstOff)
    {
        // EXP-37: skip the pair when PrefetchNext already issued it; the
        // drain below still covers the prefetched arrival at the identical
        // dataflow point.
        const bool skipSeed = (t == pfT_ && s == pfS_);
        pfT_ = -1;
        if (!skipSeed) {
            PrepareZ1(t, s);
        }
        const uint32_t len = lb - la;
        const uint32_t outPerTile = wTile_ / R_;
        uint32_t nFull = len / outPerTile;
        uint32_t rem = len - nFull * outPerTile;
        if (nFull == 0 && rem == 0) {
            return;  // empty intersection; production chunks are >= one tile
        }
        // EXP-18 (depth-2 inQueueW_): seed tile 0, drain for the dup rows +
        // copy 0 (EXP-1b placement kept, but BEFORE the next copy is issued
        // so the barrier cannot absorb it), then each iteration issues copy
        // i+1 into the slot ComputeTile(i-1) freed, ahead of that tile's
        // DeQue - the per-tensor queue-event wait - so compute(i) covers
        // copy(i+1)'s latency. Same pattern as z1's EXP-8 A queue.
        if (!skipSeed) {
            CopyInW(slot, s, la * R_, (nFull > 0 ? (int32_t)wTile_ : (int32_t)(rem * R_)));
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < nFull; ++i) {
            if (i + 1 < nFull) {
                CopyInW(slot, s, (la + (i + 1) * outPerTile) * R_, (int32_t)wTile_);
            } else if (rem != 0) {
                CopyInW(slot, s, (la + nFull * outPerTile) * R_, rem * R_);
            }
            // EXP-35: explicit full-tile repeat counts from wReps_ (identical
            // to the old compile-time defaults at wTile_ == 8192: 128/16/8).
            const int32_t pair16f = (int32_t)((wReps_ * NUM_BLOCKS_PER_REPEAT
                                               + NUM_ELEMENTS_PER_REPEAT - 1)
                                              / NUM_ELEMENTS_PER_REPEAT);
            ComputeTile(i * outPerTile, dstOff, (int32_t)wReps_, pair16f, (pair16f + 1) / 2);
        }
        if (rem != 0) {
            uint32_t remW = rem * R_;
            int32_t lastRepeatCount = remW / NUM_ELEMENTS_PER_REPEAT;
            int32_t pair16 = (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1)
                             / NUM_ELEMENTS_PER_REPEAT;
            int32_t pair32 = (pair16 + 1) / 2;
            ComputeTile(nFull * outPerTile, dstOff, lastRepeatCount, pair16, pair32);
        }
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueW_;  // EXP-18: 2-deep
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> debugPad_, tmpW_, tmpY_, yInF_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dupBufs_[4];
    uint32_t dupSel_ = 0;
    // EXP-37 cross-unit seed state (PrefetchNext/DoRange): pfUnit_ = the
    // unit whose index read was prefetched (loop top reuses pfSlot_);
    // pfT_/pfS_ = pending seeded intersection, pfT_ == -1 means none.
    int64_t pfUnit_ = -1;
    int64_t pfT_ = -1;
    uint32_t pfS_ = 0;
    int64_t pfSlot_ = 0;
    AscendC::GlobalTensor<T> wb_[4];
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<float> z1Gm_;
    uint32_t batch_, units_, unitsPerCore_, R_, chunk_, chunksPerToken_, yWidth_, addInputs_;
    uint32_t wTile_ = W_IN_TILE, wReps_ = BLOCK_REDUCE_NUM_REPEATS;  // EXP-35
    uint32_t h2_[4];
    uint32_t offs_[4];
    uint32_t totalH2_;

    // repeat layouts copied from bgmv_expand (block=32B, 8 blocks per
    // repeat); reduce strides live in the named constants above
    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};
};

#define ADD_LORA_Z1_DECLARE(TYPE)                                                                        \
    extern "C" __global__ __aicore__ void add_lora_z1_##TYPE(                                            \
        __gm__ void* x, __gm__ void* wa0, __gm__ void* wa1, __gm__ void* wa2, __gm__ void* wa3,          \
        __gm__ void* indices, __gm__ void* z1out, uint32_t batch, uint32_t units, uint32_t unitsPerCore, \
        uint32_t H1, uint32_t R, uint32_t rankBlock, float scale)                                        \
    {                                                                                                    \
        AscendC::TPipe pipe;                                                                             \
        AddLoraZ1<TYPE> op(&pipe);                                                                       \
        op.Init(x, wa0, wa1, wa2, wa3, indices, z1out, batch, units, unitsPerCore, H1, R, rankBlock,     \
                scale);                                                                                  \
        op.Process();                                                                                    \
    }

#define ADD_LORA_Z2_DECLARE(TYPE)                                                                        \
    extern "C" __global__ __aicore__ void add_lora_z2_##TYPE(                                            \
        __gm__ void* z1in, __gm__ void* wb0, __gm__ void* wb1, __gm__ void* wb2, __gm__ void* wb3,       \
        __gm__ void* indices, __gm__ void* y, uint32_t batch, uint32_t units, uint32_t unitsPerCore,     \
        uint32_t R, uint32_t h2_0, uint32_t h2_1, uint32_t h2_2, uint32_t h2_3, uint32_t chunk,          \
        uint32_t chunksPerToken, uint32_t yWidth, uint32_t addInputs, uint32_t wTile)                    \
    {                                                                                                    \
        AscendC::TPipe pipe;                                                                             \
        AddLoraZ2<TYPE> op(&pipe);                                                                       \
        op.Init(z1in, wb0, wb1, wb2, wb3, indices, y, batch, units, unitsPerCore, R, h2_0, h2_1, h2_2,    \
                h2_3, chunk, chunksPerToken, yWidth, addInputs, wTile);                                  \
        op.Process();                                                                                    \
    }

ADD_LORA_Z1_DECLARE(half)
ADD_LORA_Z2_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ADD_LORA_Z1_DECLARE(bfloat16_t)
ADD_LORA_Z2_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

// Host-side grid planning shared by both launches.
struct AddLoraGrid {
    uint32_t units;
    uint32_t unitsPerCore;
    uint32_t gridDim;
};

static inline AddLoraGrid PlanGrid(uint32_t units, uint32_t aivNum)
{
    AddLoraGrid g;
    g.units = units;
    g.unitsPerCore = (units + aivNum - 1) / aivNum;
    if (g.unitsPerCore == 0) {
        g.unitsPerCore = 1;
    }
    g.gridDim = (units + g.unitsPerCore - 1) / g.unitsPerCore;
    return g;
}

// Host-side entry. Slice descriptors arrive as plain arrays (wa/wb/h2, first
// nSlices entries valid, nSlices <= 4); the fixed-signature AscendC launchers
// are fed from them at the single unrolled launch site below.
extern void add_lora_fused_impl(AscendType type, void *stream, void *x,
                                void *const *wa, void *const *wb, void *indices, void *y,
                                void *z1ws, uint32_t batch, uint32_t H1, uint32_t R,
                                const uint32_t *h2, uint32_t nSlices, uint32_t yWidth,
                                float scale, uint32_t addInputs, uint32_t aivNum)
{
    // ---- z1: (slice, token, rank-group) units; rank groups sized to fill
    // the machine at small B (rankBlock multiple of 8 keeps GM writes
    // 32B-aligned; R % 16 == 0 keeps the tail group aligned too)
    //
    // EXP-4: pick rankBlock by an explicit core-balance cost model instead
    // of the old targetGroups heuristic. Per-core serial time is ~
    // ceil(units(rb)/aivNum) * rb: the unitsPerCore units the busiest AIV
    // runs, each covering rb rank rows. The ceil() quantisation is exactly
    // what left 8 of 40 cores idle at B=48 with the old rule (96 units ->
    // 32 cores x 3 units). Scan multiples of 8 up to R, prefer the coarser
    // rb, and take a finer one only if it is >8% cheaper on the model --
    // the margin stands in for the per-unit fixed cost (GM indices read,
    // tail drain, compaction) the model ignores.
    uint32_t rankBlock = R;
    uint64_t unitsBest = (uint64_t)nSlices * batch;
    uint64_t bestCost = ((unitsBest + aivNum - 1) / aivNum) * rankBlock;
    for (uint32_t rb = 8; rb < R; rb += 8) {
        uint64_t unitsRb = (uint64_t)nSlices * batch * ((R + rb - 1) / rb);
        uint64_t upc = (unitsRb + aivNum - 1) / aivNum;
        if (upc == 0) {
            upc = 1;
        }
        uint64_t cost = upc * rb;
        if (cost * 25 < bestCost * 23) {  // strictly >8% cheaper
            bestCost = cost;
            rankBlock = rb;
        }
    }
    uint32_t groups = (R + rankBlock - 1) / rankBlock;
    AddLoraGrid g1 = PlanGrid(nSlices * batch * groups, aivNum);

    // ---- z2: (token, chunk) units over the concatenated output range;
    // chunk is a multiple of the W-tile quantum (wTile/R) and capped at
    // Y_OUT_TILE (tmpY_ size)
    const uint32_t totalH2 = h2[0] + h2[1] + h2[2] + h2[3];
    // EXP-35: single-slice shapes (G2) widen the W tile (15360 = 240 reps,
    // uint8-tile ceiling 255) to cut per-token tile-loop/MTE2 issues; G2
    // chunks 4096 run 4 full + one 256-output rem tile (the G1-proven shape),
    // small-B chunks retune to the 960 quantum.
    const uint32_t wTile = (nSlices == 1) ? W_IN_TILE_BIG : W_IN_TILE;
    const uint32_t outPerTile = wTile / R;
    uint32_t wantChunks = (aivNum + batch - 1) / batch;
    if (wantChunks < 1) {
        wantChunks = 1;
    }
    // outPerTile in {128, 256, 512} (R in {16,32,64}) divides Y_OUT_TILE,
    // so a plain cap preserves the W-tile quantum — no alignment fixups needed
    uint32_t chunk = outPerTile * ((totalH2 + wantChunks * outPerTile - 1) / (wantChunks * outPerTile));
    if (chunk > Y_OUT_TILE) {
        chunk = Y_OUT_TILE;
    }
    const uint32_t chunksPerToken = (totalH2 + chunk - 1) / chunk;
    AddLoraGrid g2 = PlanGrid(batch * chunksPerToken, aivNum);

    // pad descriptors to the fixed 4-slot launcher signature: unused slots
    // repeat slot 0 (harmless — z2 skips h2 == 0 slices)
    void *waf[4];
    void *wbf[4];
    uint32_t h2f[4];
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t src = i < nSlices ? i : 0;
        waf[i] = wa[src];
        wbf[i] = wb[src];
        h2f[i] = i < nSlices ? h2[i] : 0;
    }

    if (type == AscendType::FP16) {
        add_lora_z1_half<<<g1.gridDim, nullptr, stream>>>(
            x, waf[0], waf[1], waf[2], waf[3], indices, z1ws, batch, g1.units, g1.unitsPerCore,
            H1, R, rankBlock, scale);
        add_lora_z2_half<<<g2.gridDim, nullptr, stream>>>(
            z1ws, wbf[0], wbf[1], wbf[2], wbf[3], indices, y, batch, g2.units, g2.unitsPerCore,
            R, h2f[0], h2f[1], h2f[2], h2f[3], chunk, chunksPerToken, yWidth, addInputs, wTile);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        add_lora_z1_bfloat16_t<<<g1.gridDim, nullptr, stream>>>(
            x, waf[0], waf[1], waf[2], waf[3], indices, z1ws, batch, g1.units, g1.unitsPerCore,
            H1, R, rankBlock, scale);
        add_lora_z2_bfloat16_t<<<g2.gridDim, nullptr, stream>>>(
            z1ws, wbf[0], wbf[1], wbf[2], wbf[3], indices, y, batch, g2.units, g2.unitsPerCore,
            R, h2f[0], h2f[1], h2f[2], h2f[3], chunk, chunksPerToken, yWidth, addInputs, wTile);
#endif
    }
}
}  // namespace vllm_ascend
