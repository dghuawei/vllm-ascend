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
constexpr uint32_t W_IN_TILE = 8192;            // B elements per z2 compute tile
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
// rank accumulation batch: vector instructions on this ISA address UB in 32B
// blocks; a destination at a 4-float (16B) rank offset device-faults (P4-P9
// bisection, see LOG). 8 ranks = 32B keeps every sums[] destination aligned.
// A-row LOADS stay batched at Z1_RANK_SUB=4, so UB is unchanged.
constexpr uint32_t RANK_BATCH = 8;
// EXP-11: (slice, slot, rank-group) bucketing of a core's unit run.  The
// arena shapes need at most 2 slices * 2 groups * 3 slots = 12 keys; 16
// covers them with slack, overflow units run solo through an ephemeral
// slot.  Header is (s, slot, g, count) per bucket, plus the ephemeral's.
constexpr int64_t Z1_BUCKET_MAX = 16;
constexpr int64_t Z1_BKT_HDR = 4 * (Z1_BUCKET_MAX + 1);
// EXP-35: z2 indices GM->UB prefetch window, in int64 tokens.  A block spans
// ceil(unitsPerCore/cPT) + 1 <= 29 tokens for the arena shapes (worst case
// G1 b1024: unitsPerCore 26, cPT 1); 64 carries the slack.  512B of UB.
constexpr uint32_t IDX_WINDOW_TOKENS = 64;
// EXP-11 V4: stage W once per bucket into wStaging_.  The staging bank is
// T-wide (bf16, 32KB max): the fp32 64KB variant device-faulted (507035)
// because the real UB budget is far under the nominal 192KB (V1/V2
// bisection, see LOG).  bf16 staging kills the per-token MTE2 W read;
// the per-token Cast (UB->fp32) stays, keeping results bit-identical.
// EXP-19: the V1' bisect leg (EXP-10 per-token W queue data flow) that
// this flag once selected is removed -- it was compiled-never-executed
// while its inQueueWA_ queue still reserved 32KB of z1 UB.
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
        // EXP-24: true 16-row merge for short rows (H1_ < TILE_H = G2).
        // At batch >= aivNum the host already sends rankBlock=16 (groups_=1);
        // the rb loop still walked it as TWO 8-row passes, paying x leg,
        // chain final, store and every scalar site twice per token.  With
        // H1_ < TILE_H the rankBlock's A rows are contiguous in GM and 16
        // rows fit the EXISTING wStaging_ bank (16*H1_*2B = 8KB of 64KB),
        // so one rg_-wide pass per token suffices.  G1 (H1_ == TILE_H)
        // keeps RANK_BATCH: a 16-row bank there is 128KB, over the UB
        // budget (EXP-11 bisect).  rankBlock_ is host-rounded to a
        // multiple of 8, so rg_ stays 32B-store-aligned.
        rg_ = (H1_ < TILE_H) ? ((rankBlock_ < 16u) ? rankBlock_ : 16u)
                             : RANK_BATCH;

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, batch);
        z1Gm_.SetGlobalBuffer((__gm__ float *)z1out);

        pipe_->InitBuffer(inQueueX_, 1, TILE_H * sizeof(T));
        pipe_->InitBuffer(tmpX_, TILE_H * sizeof(float));
        pipe_->InitBuffer(tmpWA_, TILE_H * sizeof(float));
        // EXP-20: fp32 W pair staging (2 rows), on the 32KB VECIN freed
        // by EXP-19's dead-leg removal.  z1 UB total ~104KB, under the
        // 116KB device-fault boundary booked in EXP-11's bisect.
        pipe_->InitBuffer(tmpW2_, 2 * TILE_H * sizeof(float));
        // level-1 partial rows for the hardware reduce.  vcgadd emits 8
        // outputs per repeat (one per 32B datablock -- P11 + the manual's
        // dstRepStride row), so a row footprint is 8*reps floats; worst
        // case G1 = 8 rows x 64 repeats x 8 floats = 16KB.
        pipe_->InitBuffer(redScratch_,
                          RANK_BATCH * (TILE_H / NUM_ELEMENTS_PER_REPEAT) *
                              NUM_BLOCKS_PER_REPEAT * sizeof(float));
        // P13c: second reduce bank.  The in-place (dst==src) collapse
        // chain wrote zeros on this chip (P13a, 14/14 rms~1.0); out-of-
        // place alternating passes isolate source-vs-dest ordering as
        // the single bisected variable (see LOG).
        pipe_->InitBuffer(redScratchB_,
                          RANK_BATCH * (TILE_H / NUM_ELEMENTS_PER_REPEAT) *
                              NUM_BLOCKS_PER_REPEAT * sizeof(float));
        // EXP-11 V4: bf16 W staging bank -- the (rb, h0) rank-group of A
        // rows staged ONCE per same-(s,slot) token bucket (see
        // ProcessBucket), replacing the per-token GM->queue leg.  The
        // fp32 (2x-wide) variant faulted at 507035: the V3 probe proved
        // exactly this 32KB T-wide footprint boots, the 64KB fp32 one
        // does not (real UB budget << nominal 192KB, see LOG bisect).
        pipe_->InitBuffer(wStaging_, RANK_BATCH * TILE_H * sizeof(T));
        // EXP-11 bucket scratch: (s,slot,g,count) header per bucket, one
        // fixed runLen<=unitsPerCore_ token slab per bucket, plus the
        // overflow header and its slab slot.  The +4 int64s keep the total
        // a 32B multiple (576 + 128*unitsPerCore) -- UB allocations are
        // 32B granular here; every other InitBuffer in this file is.
        pipe_->InitBuffer(bktBuf_,
                          (Z1_BKT_HDR + Z1_BUCKET_MAX * (int64_t)unitsPerCore_ +
                           4) *
                              sizeof(int64_t));
        // EXP-10: framework-allocated event IDs for the store-fence pair
        // (V->MTE3 gates the copy-out read of wr; MTE3->V gates the next
        // iteration's bank writes vs the store's source read).  IDs MUST
        // come from AllocEventID -- hand-picked IDs are documented to
        // collide with framework sync and hang the core (0016:56584;
        // EXP-5/P13 hang suspect).  Held for kernel lifetime; no other
        // event user here, so no ReleaseEventID needed.
        evVtoM3_ = static_cast<int32_t>(
            pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
        evM3toV_ = static_cast<int32_t>(
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>());
        // EXP-11 V4: staging handoff -- the bucket's GM->wStaging copies
        // (MTE2) must retire before the token loop's Casts (V) read the
        // bank.  One Set + one Wait per (h0, rb) group, issued
        // back-to-back, so no bootstrap and no final drain needed.
        evM2toV_ = static_cast<int32_t>(
            pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        // EXP-11 V5: W-staging re-arm chain (see ProcessBucket): each
        // staging group Waits on V_MTE2 before its copies and Sets it
        // after its token loop.  Bootstrap feeds group #1 exactly like
        // the MTE3_V one below; Process()'s tail Wait drains the last.
        evVtoM2_ = static_cast<int32_t>(
            pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoM2_);
        // Bootstrap: the loop-top/chain-head wait always consumes one
        // pending MTE3_V event; this first SetFlag (no pending MTE3 work,
        // fires immediately) feeds iteration #1 so every Set->Wait pairs
        // one-to-one without a scalar "armed" guard.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evM3toV_);
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t start = (int64_t)blockIdx * unitsPerCore_;
        int64_t end = (int64_t)(blockIdx + 1) * unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }
        // EXP-11: amortize the W leg over same-(s,slot,g) tokens.  Units
        // are slice-major and the A rows depend on (slot, s, rank, h) but
        // NEVER on t -- so a core's consecutive run was reloading (MTE2)
        // and re-Casting (V) byte-identical W rows once per token.  Bucket
        // the run by (s,slot,g); ProcessBucket stages each (h0, rb) W
        // group ONCE into wStaging_ (bf16, the fp32 bank exceeded the
        // UB budget -- see LOG bisect) for its whole bucket; the
        // per-token Cast now reads UB.  Every per-token op, address,
        // barrier and event in ProcessBucket is the EXP-10 code -- same
        // fp32 ops on the same values, bit-identical.
        // Header = (s, slot, g, count) int64s; token lists live in fixed
        // runLen slabs; keys beyond Z1_BUCKET_MAX process the unit solo.
        AscendC::LocalTensor<int64_t> bkt = bktBuf_.Get<int64_t>();
        const int64_t runLen = end - start;
        int64_t nB = 0;
        for (int64_t u = start; u < end; ++u) {
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
            int64_t b = 0;
            while (b < nB && !(bkt.GetValue(4 * b) == (int64_t)s &&
                               bkt.GetValue(4 * b + 1) == slot &&
                               bkt.GetValue(4 * b + 2) == (int64_t)g)) {
                ++b;
            }
            int64_t hbase, lbase;
            bool solo = false;
            if (b < nB) {
                hbase = 4 * b;
                lbase = Z1_BKT_HDR + b * runLen;
            } else if (nB < Z1_BUCKET_MAX) {
                hbase = 4 * nB;
                lbase = Z1_BKT_HDR + nB * runLen;
                bkt.SetValue(hbase, (int64_t)s);
                bkt.SetValue(hbase + 1, slot);
                bkt.SetValue(hbase + 2, (int64_t)g);
                bkt.SetValue(hbase + 3, 0);
                ++nB;
            } else {
                // defensive: keys > Z1_BUCKET_MAX (arena shapes need <=
                // 2 slices * 2 groups * 3 slots).  Ephemeral single-token
                // bucket -- its slab is re-written per overflow unit, but
                // ProcessBucket runs to completion before the next one.
                hbase = 4 * Z1_BUCKET_MAX;
                lbase = Z1_BKT_HDR + Z1_BUCKET_MAX * runLen;
                bkt.SetValue(hbase, (int64_t)s);
                bkt.SetValue(hbase + 1, slot);
                bkt.SetValue(hbase + 2, (int64_t)g);
                bkt.SetValue(hbase + 3, 0);
                solo = true;
            }
            bkt.SetValue(lbase + bkt.GetValue(hbase + 3), t);
            bkt.SetValue(hbase + 3, bkt.GetValue(hbase + 3) + 1);
            if (solo) {
                ProcessBucket(bkt, hbase, lbase);
            }
        }
        for (int64_t b = 0; b < nB; ++b) {
            ProcessBucket(bkt, 4 * b, Z1_BKT_HDR + b * runLen);
        }
        // EXP-10: consume the last iteration's SetFlag<MTE3_V> so every
        // z1 store has landed by kernel exit (bootstrap set in Init makes
        // this wait legal even on cores with zero units).
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evM3toV_);
        // V5: consume the last staging group's V_MTE2 Set (event balance;
        // matches the Init bootstrap on cores with zero units, where this
        // wait consumes the bootstrap itself).
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoM2_);
    }

    // EXP-11: one (slice, slot, rank-group) bucket.  Same per-token
    // pipeline as the EXP-10 loop -- x load+Cast, level-1 reduce,
    // collapse chain, scale, event-fenced store -- with the (h0, rb) W
    // GM load hoisted ahead of the token loop into wStaging_ (bf16);
    // the per-token Cast+Mul stay, reading the staged bank.  Handoff
    // across the MTE2/V boundary: PIPE_V before the copies (previous
    // group's Cast readers retired) + MTE2_V Set/Wait after them (the
    // copies landed before any Cast reads).  x is now re-read per
    // (t, h0, rb-group) instead of per (t, h0) -- 4KB extra per rb
    // group, the accepted trade for removing 32KB of W load per token.
    __aicore__ inline void ProcessBucket(const AscendC::LocalTensor<int64_t> &bkt,
                                         int64_t hbase, int64_t lbase)
    {
        const int64_t s = bkt.GetValue(hbase);
        const int64_t slot = bkt.GetValue(hbase + 1);
        const uint32_t g = (uint32_t)bkt.GetValue(hbase + 2);
        const int64_t cnt = bkt.GetValue(hbase + 3);
        uint32_t r0 = g * rankBlock_;
        uint32_t r1 = r0 + rankBlock_;
        if (r1 > R_) {
            r1 = R_;
        }
        AscendC::LocalTensor<float> scratch = redScratch_.Get<float>();
        AscendC::LocalTensor<T> wSt = wStaging_.Get<T>();
        const int64_t aBase = slot * (int64_t)R_ * H1_;
        for (uint32_t h0 = 0; h0 < H1_; h0 += TILE_H) {
            uint32_t n = (H1_ - h0 < TILE_H) ? (H1_ - h0) : TILE_H;
            const uint32_t reps = (n + 63u) >> 6;  // 64-float repeats
            const uint32_t pad = reps * NUM_ELEMENTS_PER_REPEAT - n;
            // level-1 output footprint per row: 8 outputs per repeat
            // (block-granular vcgadd, P11) -> 8*reps floats
            // scratch row PITCH, in floats.  Level 1 emits 8 outputs per
            // 64-float repeat (block-granular vcgadd, P11), so pitch =
            // 8*reps.  The collapse chain is a chain of 64-float
            // (one-repeat) reductions, so the pitch also has to be a
            // multiple of 64 -- otherwise level 2 mixes two rows inside
            // one repeat, which sums two ranks together.  G1: reps=64 ->
            // pitch 512 (8/64/8 = 1/8/1 = whole repeats per row).  G2:
            // reps=4 -> pitch 32, which is half a repeat, so pad up to
            // 64; the chain then reads one row per repeat and the
            // untouched half repeats stay zero (they contribute 0).
            const uint32_t pitch =
                ((reps * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) /
                 NUM_ELEMENTS_PER_REPEAT) *
                NUM_ELEMENTS_PER_REPEAT;
            // Rank accumulation advances in groups of RANK_BATCH=8.
            // rankBlock is a multiple of 8 and r1 = min(r0+rankBlock, R)
            // with R in {16,32,64}, so sub8 == 8 always and the GM
            // store offset (rb*4B) and count (32B) are block-aligned --
            // mandatory, a vector destination at a 16B offset device-
            // faults (P4-P9 bisection, see LOG).  A-row LOADS still
            // batch at Z1_RANK_SUB=4 inside the group (EXP-19: the
            // inQueueWA_ queue those loads fed with the removed V1'
            // leg is gone; 32KB VECIN freed).
            // EXP-24: batch width is rg_ (16 in the merged short-row pass,
            // RANK_BATCH elsewhere) -- today's two 8-row passes collapse
            // into one 16-row pass exactly when the merge gate is open.
            for (uint32_t rb = r0; rb < r1; rb += rg_) {
                uint32_t sub8 = (r1 - rb < rg_) ? (r1 - rb) : rg_;
                // EXP-11 V4: stage the bucket's ONE copy of these 8 A
                // rows in bf16 (row-major in wStaging_, 32KB bank) before
                // any token reads them -- direct GM->UB DataCopy, no
                // queue (the bank is persistent, depth-1 queue buys
                // nothing here).  Both handoff directions use HARDWARE
                // EVENTS (V5 -- V4's leading PipeBarrier<PIPE_V> FAILED
                // 10/14, rms ~0.2: it gates only the V queue and did not
                // stop the next group's MTE2 copy from overtaking this
                // group's Casts; cf. the z2 comment that raw TBuf + pipe
                // barriers don't order MTE2):
                //   V_MTE2 Wait here  = don't overwrite wSt until the
                //       previous group's Casts retired (bootstrap Set in
                //       Init feeds the first group of the kernel);
                //   MTE2_V Set/Wait   = copies landed before any Cast
                //       reads wSt;
                //   V_MTE2 Set after the token loop = arms the next
                //       group's Wait (all groups chain across buckets;
                //       Process()'s tail Wait drains the last one).
                // Rows are contiguous in GM when H1_ <= TILE_H (G2):
                // one 4KB copy instead of 8.
                {   // EXP-19: was if (Z1_STAGE_W); flag retired with V1'
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoM2_);
                    if (H1_ <= TILE_H) {
                        DataCopy(wSt, wa_[s][aBase + (int64_t)rb * H1_],
                                 sub8 * n);
                    } else {
                        for (uint32_t i = 0; i < sub8; ++i)
                            DataCopy(wSt[i * n],
                                     wa_[s][aBase +
                                            (int64_t)(rb + i) * H1_ + h0],
                                     n);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2toV_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2toV_);
                    // EXP-20: keep an fp32 copy of the group's FIRST TWO
                    // rows (32KB bank) so their per-token Mul skips the
                    // widening Cast (128 reps/token, bucket-amortized
                    // staging).  Same Cast params as the row loop -> the
                    // staged bytes equal today's post-Cast wF bytes, so
                    // Mul(wF, xF, wStF_i) is bit-identical.  The WAR on
                    // wStF vs the previous bucket's token Muls is the
                    // same-pipe order F18a proved hardware-ordered, and
                    // the drain below fences the bucket's own readers.
                    AscendC::LocalTensor<float> wStF = tmpW2_.Get<float>();
                    for (uint32_t i = 0; i < sub8 && i < 2; ++i)
                        AscendC::Cast(wStF[i * n], wSt[i * n],
                                      AscendC::RoundMode::CAST_NONE, n);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (int64_t j = 0; j < cnt; ++j) {
                    const int64_t t = bkt.GetValue(lbase + j);
                    AscendC::LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
                    DataCopy(xLocal, xGm_[(int64_t)t * H1_ + h0], n);
                    inQueueX_.EnQue(xLocal);
                    xLocal = inQueueX_.DeQue<T>();
                    AscendC::LocalTensor<float> xF = tmpX_.Get<float>();
                    AscendC::LocalTensor<float> wF = tmpWA_.Get<float>();
                    Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, n);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueueX_.FreeTensor(xLocal);
                    // P13a: MTE2_V WaitFlag removed (bisect, see LOG)
                    // with mask=64 level 1 writes ALL 8 outputs of every
                    // repeat, so the 8*reps written slots per row are
                    // complete; only when the pitch was rounded up to a
                    // whole 64-float repeat (G2: 32 -> 64) do the pad
                    // blocks need zeroing -- the chain sums over them too
                    // EXP-10: the store drain gate (MTE3_V wait) moves
                    // here iff pad-zero can fire (G2): the pad Muls writes
                    // bank A and G2's store source is bank A (odd chain
                    // pass count), so it must land after the previous
                    // store's read.  G1 has no pad-zero and its store
                    // source is bank B, first rewritten by the chain --
                    // the wait defers to the chain head there (exactly
                    // one wait per iteration, chosen by shape).  In the
                    // bucketed form "previous" is often the PREVIOUS
                    // TOKEN's store; the gate stays one per (token, rb).
                    const bool earlyWait =
                        (pitch > reps * NUM_BLOCKS_PER_REPEAT);
                    if (earlyWait) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evM3toV_);
                        AscendC::Muls(scratch, scratch, 0.0f, sub8 * pitch);
                    }
                    {   // EXP-19: was if (Z1_STAGE_W) with a dead V1'
                        // else-branch; the branch is now unconditional.
                        // V4 staged leg: per-row Cast from the bucket's
                        // UB copy, then the EXP-10 in-place Mul -- the
                        // same V ops on the same bytes as the per-token
                        // queue leg below, just with the bf16 source in
                        // UB instead of freshly DMA'd from GM -> bit-
                        // identical results, and no per-token MTE2 read.
                        AscendC::LocalTensor<float> wStF = tmpW2_.Get<float>();
                        for (uint32_t i = 0; i < sub8; ++i) {
                            if (i < 2) {
                                // EXP-20: rows 0-1 read the bucket's fp32
                                // copy -- no widening Cast this token.
                                Mul(wF, xF, wStF[i * n], n);
                            } else {
                                Cast(wF, wSt[i * n],
                                     AscendC::RoundMode::CAST_NONE, n);
                                AscendC::PipeBarrier<PIPE_V>();
                                Mul(wF, xF, wF, n);
                            }
                            if (pad > 0) {
                                // last repeat must not sum uninitialised lanes
                                AscendC::Muls(wF[n], wF[n], 0.0f, pad);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // level 1: per-row vcgadd block reduce (see
                            // P11 pitch notes on the h0 loop).
                            BlockReduceSum(scratch[i * pitch],
                                           wF, (int32_t)reps,
                                           (int32_t)NUM_ELEMENTS_PER_REPEAT, 1, 1, 8);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                    }
                    // EXP-10: deferred store-drain gate for no-pad shapes
                    // (G1): everything above -- x load (MTE2) and all
                    // level-1 vcgadds (writing bank A only) -- overlaps
                    // the previous store's drain; the chain head is the
                    // first site that can write bank B, the G1 store
                    // source.  Exactly one WaitFlag<MTE3_V> executes per
                    // (token, rb) across the two sites.
                    if (!earlyWait) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evM3toV_);
                    }
                    // P13c hardware collapse chain: OUT-OF-PLACE vcgadd
                    // passes ping-ponging between the two scratch banks.
                    // Each pass turns every 64-float repeat into 8 outputs
                    // (dst 32B), i.e. shrinks the span 8x; rows compact
                    // with it.  The final rt=1 pass over 64 floats reduces
                    // each 8-float row slot to one value -> the sub8 ROW
                    // SUMS land densely packed in the final bank[0..sub8).
                    // P13a's in-place (dst==src) form wrote zeros; P13c's
                    // zero-aliasing form FAILED TOO (rms 1.0 -> sqrt2 =
                    // equal-power stale garbage) -> aliasing refuted.
                    // EXP-7 fence bisection (P13d/P13e put PIPE_ALL on
                    // all four tail barriers -- correct but a ~16% fence
                    // tax at G1): the inter-pass vcgadd->vcgadd and
                    // vcgadd->Muls handoffs are SAME pipe (V), exactly
                    // what PipeBarrier<PIPE_V> covers (0016:56633), so
                    // they drop back to PIPE_V.  EXP-10 replaced the two
                    // STORE-crossing PIPE_ALL hammers with narrow
                    // HardEvent pairs; the copy-out pipe is MTE3 (mte3
                    // busy scales with store count across batches; z2
                    // ZeroRange fences copy-out with PIPE_MTE3).
                    AscendC::LocalTensor<float> rd = scratch;
                    AscendC::LocalTensor<float> wr = redScratchB_.Get<float>();
                    uint32_t span = sub8 * pitch;
                    while (span > NUM_ELEMENTS_PER_REPEAT) {
                        BlockReduceSum(wr, rd,
                                       (int32_t)(span / NUM_ELEMENTS_PER_REPEAT),
                                       (int32_t)NUM_ELEMENTS_PER_REPEAT, 1, 1, 8);
                        AscendC::PipeBarrier<PIPE_V>();
                        span /= NUM_BLOCKS_PER_REPEAT;
                        AscendC::LocalTensor<float> tmp = rd;
                        rd = wr;
                        wr = tmp;
                    }
                    // EXP-24: the while-loop lands at span == 64 for every
                    // legacy shape (the 64-float final pass below is then
                    // exact).  The merged 16-row pass lands at span == sub8
                    // < 64: the LAST vcgadd pass already emitted sub8 dense
                    // per-rank sums into rd (each 64-float repeat reduces
                    // its eight 8-float row slots independently -- same
                    // per-rank reduction shape as the final pass, so bit-
                    // identical values).  A 64-float final there would read
                    // past them into stale non-zero tail data, so it is
                    // skipped and rd is the final bank.  rd is bank A, whose
                    // WAR chain (next token's earlyWait zero + level 1) is
                    // already gated on the MTE3_V event below.
                    AscendC::LocalTensor<float> fin;
                    uint32_t finCnt;
                    if (span == NUM_ELEMENTS_PER_REPEAT) {
                        BlockReduceSum(wr, rd, 1,
                                       (int32_t)NUM_ELEMENTS_PER_REPEAT, 1, 1, 8);
                        AscendC::PipeBarrier<PIPE_V>();
                        Muls(wr, wr, scale_, sub8);
                        fin = wr;
                        finCnt = sub8;
                    } else {
                        Muls(rd, rd, scale_, span);
                        fin = rd;
                        finCnt = span;
                    }
                    // EXP-10 pre-store: V -> MTE3 gate BEFORE the copy is
                    // issued (WaitFlag blocks the MTE3 queue's later
                    // instructions, 0016:56315 -- so DataCopy must come
                    // AFTER the wait).  Muls's writeback lands before MTE3
                    // reads it; MTE2 needs no drain here (unlike the
                    // PIPE_ALL hammer it replaces).
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoM3_);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoM3_);
                    DataCopy(z1Gm_[((int64_t)s * batch_ + t) * R_ + rb], fin,
                             finCnt);
                    // EXP-10 post-store: MTE3 -> V gate, SetFlag only --
                    // fires when this store completes; the paired WaitFlag
                    // runs at the next bank write that can alias this
                    // store's source (loop top for pad-zero shapes, chain
                    // head otherwise) -- in the bucketed form that is
                    // often the NEXT TOKEN's write, which is exactly the
                    // overlap EXP-10 was built for.  P13d's rms 0.693
                    // race stays gated.
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evM3toV_);
                }
                // V5: arm the next staging group's V_MTE2 wait -- fires
                // when ALL V issued for this group (every staged-row Cast
                // included) has retired, so the next group's MTE2 copy
                // cannot overwrite wSt under an in-flight reader.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoM2_);
            }
        }
    }

private:
    AscendC::TPipe *pipe_;
    int32_t evVtoM3_, evM3toV_, evM2toV_, evVtoM2_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpX_, tmpWA_, tmpW2_,
        redScratch_, redScratchB_;
    // EXP-11: bf16 W staging (one rb-group x TILE_H) + bucket scratch
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wStaging_, bktBuf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> wa_[4];
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<float> z1Gm_;
    uint32_t batch_, units_, unitsPerCore_, H1_, R_, rankBlock_, groups_;
    // EXP-24 merged-pass rb width (see Init)
    uint32_t rg_;
    float scale_;
    int32_t evMte2V_;
};

// ============================ kernel 2: z2 =================================
// EXP-29b: WQD = inQueueW_ depth, selected by GEOMETRY in the launcher
// (F31: depth 2 wins ~17% at few-tile G2 shapes, costs +4-9% wait at
// G1's many-token stream -> h2_1==0 (G2) gets 2, G1 keeps 1).
template <typename scalar_t, int WQD, int IDXP = 0>
class AddLoraZ2 {
public:
    using T = scalar_t;

    __aicore__ inline AddLoraZ2(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *z1in, __gm__ void *wb0, __gm__ void *wb1,
                                __gm__ void *wb2, __gm__ void *wb3, __gm__ void *indices,
                                __gm__ void *y, uint32_t batch, uint32_t units,
                                uint32_t unitsPerCore, uint32_t R, uint32_t h2_0,
                                uint32_t h2_1, uint32_t h2_2, uint32_t h2_3, uint32_t chunk,
                                uint32_t chunksPerToken, uint32_t yWidth, uint32_t addInputs)
    {
        batch_ = batch;
        units_ = units;
        unitsPerCore_ = unitsPerCore;
        R_ = R;
        chunk_ = chunk;
        chunksPerToken_ = chunksPerToken;
        yWidth_ = yWidth;
        addInputs_ = addInputs;
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

        // EXP-8 (P16): the dup lane now comes straight from GM via MTE2; a
        // depth-1 VECIN queue carries the MTE2->V handoff (the proven z1
        // pattern -- raw TBuf + PipeBarrier<PIPE_MTE2> does NOT make MTE2
        // writes visible to V, see the EXP-5 root cause).
        pipe_->InitBuffer(inQueueZ1_, 1, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(inQueueW_, WQD, W_IN_TILE * sizeof(T));  // EXP-29b: gated depth
        pipe_->InitBuffer(tmpW_, W_IN_TILE * sizeof(float));
        pipe_->InitBuffer(inQueueY_, 1, Y_OUT_TILE * sizeof(T));
        pipe_->InitBuffer(outQueueY_, 1, Y_OUT_TILE * sizeof(T));
        pipe_->InitBuffer(tmpY_, Y_OUT_TILE * sizeof(float));
        pipe_->InitBuffer(yInF_, Y_OUT_TILE * sizeof(float));
        // EXP-35: eager indices window cache (EXP-33 code, geometry-gated;
        // IDXP=0 instantiations stay byte-exact EXP-30 -- no alloc at all).
        if (IDXP) {
            pipe_->InitBuffer(idxPool_, IDX_WINDOW_TOKENS * sizeof(int64_t));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t begin = (int64_t)blockIdx * unitsPerCore_;
        int64_t end = (int64_t)(blockIdx + 1) * unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }
        // EXP-35 = EXP-33 eager prefetch, reachable ONLY through the
        // unitsPerCore>=4 G1 gate (F33/F34: a blocking-read replacement must
        // land in block START-UP slack; mid-loop stalls have no slack).
        uint32_t idxLo4 = 0, idxHi = 0;
        AscendC::LocalTensor<int64_t> idxView;
        if (IDXP) {
            idxView = idxPool_.Get<int64_t>(IDX_WINDOW_TOKENS);
        }
        if (IDXP && begin < end) {
            const uint32_t beginT = (uint32_t)(begin / chunksPerToken_);
            const uint32_t endT = (uint32_t)((end - 1) / chunksPerToken_) + 1;
            idxLo4 = beginT & ~3U;
            uint32_t hi4 = idxLo4 + (((endT - idxLo4) + 3U) & ~3U);
            if (hi4 > batch_) {
                hi4 = batch_ & ~3U;  // clamp; [hi4, endT) falls back to GM reads
            }
            if (hi4 > idxLo4) {
                idxHi = hi4;
                DataCopy(idxView, indicesGm_[(int64_t)idxLo4], (int32_t)(idxHi - idxLo4));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            }
        }
        for (int64_t u = begin; u < end; ++u) {
            int64_t t = u / chunksPerToken_;
            uint32_t c = (uint32_t)(u % chunksPerToken_);
            int64_t slot;
            if (IDXP && (uint32_t)t >= idxLo4 && (uint32_t)t < idxHi) {
                slot = idxView.GetValue((uint32_t)t - idxLo4);
            } else {
                slot = indicesGm_.GetValue(t);
            }
            uint32_t gs = c * chunk_;
            uint32_t ge = gs + chunk_;
            if (ge > totalH2_) {
                ge = totalH2_;
            }
            const uint64_t yRow = (uint64_t)t * yWidth_;
            if (slot < 0) {
                // no-lora token: accumulate leaves y untouched; overwrite
                // zeroes this block's chunk across all slices (chunk-local
                // ZeroRow; the union over chunks covers the full row).
                if (!addInputs_) {
                    for (uint32_t s = 0; s < 4; ++s) {
                        uint32_t a, b;
                        if (IntersectSlice(s, gs, ge, a, b)) {
                            ZeroRange(yRow + a, b - a);
                        }
                    }
                }
                continue;
            }
            // process the chunk's intersection with each slice (z1 is
            // per-slice: A_s differs, so each intersection loads its own
            // z1 row from the [S][B][R] workspace)
            for (uint32_t s = 0; s < 4; ++s) {
                uint32_t a, b;
                if (IntersectSlice(s, gs, ge, a, b)) {
                    RangeSlice(t, s, slot, a - offs_[s], b - offs_[s], yRow + a);
                }
            }
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

    // z1 workspace is [S][B][R]: each slice intersection reads its own row,
    // then replicates it into one 256B repeat for the repeat-mask dot.
    // EXP-8 (P16): the replication is 64/R aligned GM->UB DataCopies (the
    // row base is a >=64B multiple, R*4B >= 64B), replacing 64 scalar
    // SetValue/GetValue pairs -- z2's scalar pipe was 0.88 busy at G1.
    __aicore__ inline void PrepareZ1(int64_t t, uint32_t s)
    {
        AscendC::LocalTensor<float> dup = inQueueZ1_.AllocTensor<float>();
        const int64_t base = ((int64_t)s * batch_ + t) * R_;
        for (uint32_t k = 0; k < NUM_ELEMENTS_PER_REPEAT; k += R_) {
            DataCopy(dup[k], z1Gm_[base], R_);
        }
        inQueueZ1_.EnQue(dup);
    }

    __aicore__ inline void CopyInW(int64_t slot, uint32_t slice, uint32_t wElemOff,
                                   int32_t numElements)
    {
        AscendC::LocalTensor<T> wLocal = inQueueW_.template AllocTensor<T>();
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

    // dot(dup vs one B tile) -> yTile[progress...]; bgmv_expand Compute
    __aicore__ inline void ComputeTile(AscendC::LocalTensor<float> dup, int32_t progress,
                                       int32_t blockReduceRepeatCount = BLOCK_REDUCE_NUM_REPEATS,
                                       int32_t pairReduceRepeat16 = PAIR_REDUCE_NUM_REPEATS_16,
                                       int32_t pairReduceRepeat32 = PAIR_REDUCE_NUM_REPEATS_32)
    {
        AscendC::LocalTensor<float> yLocal = tmpY_.Get<float>();
        AscendC::LocalTensor<T> wLocal = inQueueW_.template DeQue<T>();
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
            PairReduceSum(yLocal[progress], wTmp, pairReduceRepeat16, NUM_ELEMENTS_PER_REPEAT,
                          REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
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
            PairReduceSum(yLocal[progress], wTmp, pairReduceRepeat32, NUM_ELEMENTS_PER_REPEAT,
                          REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        } else {  // R_ == 64
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            BlockReduceSum(yLocal[progress], wTmp, pairReduceRepeat16, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
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

    // Process outputs [la, lb) of slice `s`, writing y at yAddr (= row + la).
    // Unified full-tile + remainder path (merges v1's main loop and
    // ComputeLastIteration); start offsets are multiples of 16 (chunk and
    // slice boundaries), W reads at la*R elements are 32B-aligned, and the
    // remainder W count rem*R is a multiple of 64 (rem % 16, R % 16).
    __aicore__ inline void RangeSlice(int64_t t, uint32_t s, int64_t slot, uint32_t la,
                                      uint32_t lb, uint32_t yAddr)
    {
        PrepareZ1(t, s);
        // EXP-8: one dup lane per (token,slice); DeQue carries the MTE2->V
        // wait for the tiled copies.  Freed after the last tile -- every
        // ComputeTile ends with PIPE_V after its last dup read, so the
        // last one drains V before the block returns to the queue pool.
        AscendC::LocalTensor<float> dup = inQueueZ1_.DeQue<float>();
        const uint32_t len = lb - la;
        const uint32_t outPerTile = W_IN_TILE / R_;
        if (addInputs_) {
            CopyInY(yAddr, len);
        }
        uint32_t nFull = len / outPerTile;
        uint32_t rem = len - nFull * outPerTile;
        for (uint32_t i = 0; i < nFull; ++i) {
            CopyInW(slot, s, (la + i * outPerTile) * R_, W_IN_TILE);
            ComputeTile(dup, i * outPerTile);
        }
        if (rem != 0) {
            uint32_t remW = rem * R_;
            CopyInW(slot, s, (la + nFull * outPerTile) * R_, remW);
            int32_t lastRepeatCount = remW / NUM_ELEMENTS_PER_REPEAT;
            int32_t pair16 = (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1)
                             / NUM_ELEMENTS_PER_REPEAT;
            int32_t pair32 = (pair16 + 1) / 2;
            ComputeTile(dup, nFull * outPerTile, lastRepeatCount, pair16, pair32);
        }
        inQueueZ1_.FreeTensor(dup);
        ScaleOutput(len);
        CopyOut(yAddr, len);
    }

    AscendC::TPipe *pipe_;
    // EXP-29/29b: inQueueW_ depth WQD -- CopyInW of tile i+1 may issue
    // while tile i's V chain still holds its slot (depth-1 made the
    // single 16KB buffer the per-tile lock; see prediction + F31).
    AscendC::TQue<AscendC::QuePosition::VECIN, WQD> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY_, inQueueZ1_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpW_, tmpY_, yInF_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> idxPool_;  // EXP-35 (IDXP)
    AscendC::GlobalTensor<T> wb_[4];
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<float> z1Gm_;
    uint32_t batch_, units_, unitsPerCore_, R_, chunk_, chunksPerToken_, yWidth_, addInputs_;
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
        uint32_t chunksPerToken, uint32_t yWidth, uint32_t addInputs)                                    \
    {                                                                                                    \
        AscendC::TPipe pipe;                                                                             \
        /* EXP-29b: single-slice G2 shape (h2_1==0) -> depth-3 W queue;   \
           G1 (h2_1!=0) keeps the EXP-24 depth-1 path (F31).              \
           EXP-35: G1 AND unitsPerCore>=4 (b128..b1024) additionally arms  \
           the eager indices prefetch (IDXP=1, F33/F34 startup-slack); the  \
           remaining G1 rows (upc<=2) stay byte-exact EXP-30. */           \
        if (h2_1 == 0) {                                                                                 \
            AddLoraZ2<TYPE, 3> op(&pipe);                                                                \
            op.Init(z1in, wb0, wb1, wb2, wb3, indices, y, batch, units, unitsPerCore, R, h2_0,           \
                    h2_1, h2_2, h2_3, chunk, chunksPerToken, yWidth, addInputs);                         \
            op.Process();                                                                                \
        } else if (unitsPerCore >= 4) {                                                                  \
            AddLoraZ2<TYPE, 1, 1> op(&pipe);                                                             \
            op.Init(z1in, wb0, wb1, wb2, wb3, indices, y, batch, units, unitsPerCore, R, h2_0,           \
                    h2_1, h2_2, h2_3, chunk, chunksPerToken, yWidth, addInputs);                         \
            op.Process();                                                                                \
        } else {                                                                                         \
            AddLoraZ2<TYPE, 1, 0> op(&pipe);                                                             \
            op.Init(z1in, wb0, wb1, wb2, wb3, indices, y, batch, units, unitsPerCore, R, h2_0,           \
                    h2_1, h2_2, h2_3, chunk, chunksPerToken, yWidth, addInputs);                         \
            op.Process();                                                                                \
        }                                                                                                \
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
    uint32_t targetGroups = (aivNum + batch - 1) / batch;
    if (targetGroups < 1) {
        targetGroups = 1;
    }
    if (targetGroups > R) {
        targetGroups = R;
    }
    uint32_t rankBlock = (R + targetGroups - 1) / targetGroups;
    rankBlock = (rankBlock + 7) / 8 * 8;  // round up to multiple of 8
    if (rankBlock > R) {
        rankBlock = R;
    }
    uint32_t groups = (R + rankBlock - 1) / rankBlock;
    AddLoraGrid g1 = PlanGrid(nSlices * batch * groups, aivNum);

    // ---- z2: (token, chunk) units over the concatenated output range;
    // chunk is a multiple of the W-tile quantum (8192/R) and capped at
    // Y_OUT_TILE (tmpY_ size)
    const uint32_t totalH2 = h2[0] + h2[1] + h2[2] + h2[3];
    const uint32_t outPerTile = W_IN_TILE / R;
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
            R, h2f[0], h2f[1], h2f[2], h2f[3], chunk, chunksPerToken, yWidth, addInputs);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        add_lora_z1_bfloat16_t<<<g1.gridDim, nullptr, stream>>>(
            x, waf[0], waf[1], waf[2], waf[3], indices, z1ws, batch, g1.units, g1.unitsPerCore,
            H1, R, rankBlock, scale);
        add_lora_z2_bfloat16_t<<<g2.gridDim, nullptr, stream>>>(
            z1ws, wbf[0], wbf[1], wbf[2], wbf[3], indices, y, batch, g2.units, g2.unitsPerCore,
            R, h2f[0], h2f[1], h2f[2], h2f[3], chunk, chunksPerToken, yWidth, addInputs);
#endif
    }
}
}  // namespace vllm_ascend
