/*
 * Copyright (c) 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused MoE-LoRA w13 epilogue: delta add + SwiGLU + dynamic per-token int8
 * quant, in one pass over the gate_up buffer.
 *
 * Replaces this sequence in vllm_ascend/lora/quant_moe.py:
 *     gate_up + delta             (a materialized elementwise add)
 *     act = npu_swiglu(gate_up)   (read [T, 2W], write [T, W])
 *     y, s = npu_dynamic_quant(act)
 * Traffic per token drops from 21 * W bytes to 11 * W: the sum is never
 * written back, and `act` is produced as a side output rather than re-read.
 *
 * PERFORMANCE NOTE (measured on 910B4, 2026-09-29): traffic is not the whole
 * story. The chain this replaces runs at ~95% of HBM peak because the vendor
 * ops are deeply pipelined, so a kernel that halves traffic but stalls wins
 * nothing. The first version of this kernel did exactly that -- depth-1
 * queues, a per-half DataCopy, and a 4-byte scale store per token -- and
 * measured 46-60% of peak, making the fused path a wash end to end. Hence:
 *   - every queue is BUFFER_NUM deep, and Compute() frees its input tensors
 *     as soon as they are consumed, so MTE2 fetches row t+1 while the vector
 *     pipe works on row t;
 *   - a row of gate_up is contiguous ([T, 2W], gate then up), so it moves in
 *     ONE DataCopy of 2W elements rather than two of W -- half the
 *     instructions at twice the granularity;
 *   - scales accumulate in UB and flush once per SCALE_TILE rows instead of
 *     one 4-byte DataCopyPad per token.
 * UB budget is 34 * W bytes (two 2W input rows and two 2W delta rows double
 * buffered, W-wide act/y outputs double buffered, three fp32 W scratch
 * buffers), which is what caps W at W_MAX below.
 *
 * `act` (the bf16 SwiGLU result) is NOT optional: moe_lora_apply_w2 consumes it
 * as the lora_a input of the w2 shrink, which must stay in floating point.
 *
 *   out[t, j] = silu(gate[t, j] + dGate[t, j]) * (up[t, j] + dUp[t, j])
 * with dGate = delta[t, 0:W], dUp = delta[t, W:2W] -- the delta shares gate_up's
 * [T, 2W] layout, which is exactly what add_lora_expand's two slices write.
 *   act[t, j] = out[t, j]                              (scalar_t)
 *   scale[t]  = max_j |out[t, j]| / 127                (fp32)
 *   y[t, j]   = rint(out[t, j] / scale[t])             (int8)
 * where gate = gateUp[t, 0:W] and up = gateUp[t, W:2W] (npu_swiglu's
 * activate_left=True convention), and the delta inputs may be null.
 *
 * NUMERICS: the quant convention (absmax/127, multiply by the reciprocal, RINT
 * via the int16 -> half -> int8 cast chain) mirrors DequantSwigluQuantDynamicBase,
 * but the fused path is NOT bit-identical to the ops it replaces: it accumulates
 * the delta and evaluates silu in fp32, where the eager chain rounds to the
 * input dtype after the add and again after the activation. Measured divergence
 * against the eager chain is ~0.06 absolute on act and up to 2 int8 levels on y.
 * Tests must compare with a tolerance, not bitwise.
 *
 * One token per unit: a whole row lives in UB, because the per-token absmax
 * needs the full row before anything can be scaled.
 *
 * EDITING NOTE: these kernels are parsed by the build's auto_gen tool --
 * pointer parameters MUST be written `__gm__ void* name` (star on the type,
 * not the name) or the generated launchers dereference them; and any kernel
 * signature change requires wiping the generated chain (build/.../auto_gen,
 * include/vllm_ascend_kernels, the *_precompile/preprocess/aic/aiv
 * device-prefix dirs) or stale launchers poison the rebuild.
 */

#include "kernel_operator.h"
#include "types.h"

namespace {
// UB is 192 KB/core and this kernel needs 34 B per column (see the note above);
// 5120 columns = 170 KB, leaving headroom for TPipe's own bookkeeping. Wider
// rows fall back to the eager chain host-side, which costs nothing: above this
// width the fused path was never ahead of it.
constexpr uint32_t W_MAX = 5120;
constexpr int32_t BUFFER_NUM = 2;
// one DataCopyPad per this many rows instead of one 4-byte store per row
constexpr uint32_t SCALE_TILE = 64;
// Rows are processed in tiles so that the costs that are per-tile rather than
// per-element -- the V_S sync around the absmax readback, and the ~15 pipe
// barriers of the silu/quant chain -- amortize over several rows. Narrow rows
// were the whole gap to roofline: at W=1408 one row is only 5.6 KB, so the
// fixed cost dominated (measured 69% of peak vs 93% at W=4096). A tile is
// contiguous in every buffer it touches, so it still moves in single copies.
constexpr uint32_t TILE_ELEMS = 5120;  // the UB budget, in columns
constexpr uint32_t R_MAX = 8;

template <typename scalar_t>
class AddLoraSwigluQuant {
public:
    using T = scalar_t;

    __aicore__ inline AddLoraSwigluQuant(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *gateUp, __gm__ void *delta, __gm__ void *act,
                                __gm__ void *y, __gm__ void *scale, uint32_t batch,
                                uint32_t units, uint32_t unitsPerCore, uint32_t width,
                                uint32_t hasDelta)
    {
        batch_ = batch;
        units_ = units;
        unitsPerCore_ = unitsPerCore;
        W_ = width;
        hasDelta_ = hasDelta;
        R_ = TILE_ELEMS / W_;
        if (R_ == 0) {
            R_ = 1;
        } else if (R_ > R_MAX) {
            R_ = R_MAX;
        }
        tileElems_ = R_ * W_;

        gateUpGm_.SetGlobalBuffer((__gm__ T *)gateUp);
        // null when the caller has no delta for this layer; never dereferenced
        // in that case (guarded by hasDelta_). Same [T, 2W] layout as gateUp,
        // which is exactly what add_lora_expand's two slices write.
        deltaGm_.SetGlobalBuffer((__gm__ T *)delta);
        actGm_.SetGlobalBuffer((__gm__ T *)act);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale, batch);

        // sized to the actual row, not the cap: W_ is a multiple of 32 (checked
        // host-side), so every allocation stays 32B-aligned
        pipe_->InitBuffer(inQueue_, BUFFER_NUM, tileElems_ * 2 * sizeof(T));
        if (hasDelta_ != 0) {
            pipe_->InitBuffer(deltaQueue_, BUFFER_NUM, tileElems_ * 2 * sizeof(T));
        }
        pipe_->InitBuffer(outActQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
        pipe_->InitBuffer(outYQueue_, BUFFER_NUM, tileElems_ * sizeof(int8_t));
        pipe_->InitBuffer(scaleBuf_, SCALE_TILE * sizeof(float));
        pipe_->InitBuffer(maxBuf_, R_MAX * sizeof(float));
        pipe_->InitBuffer(fGate_, tileElems_ * sizeof(float));
        pipe_->InitBuffer(fUp_, tileElems_ * sizeof(float));
        pipe_->InitBuffer(fTmp_, tileElems_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t start = (int64_t)blockIdx * unitsPerCore_;
        int64_t end = start + unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }

        AscendC::LocalTensor<float> scales = scaleBuf_.Get<float>();
        int64_t scaleBase = start;
        uint32_t nScale = 0;

        for (int64_t t = start; t < end; t += R_) {
            uint32_t r = (uint32_t)(end - t);
            if (r > R_) {
                r = R_;
            }
            CopyIn(t, r);
            Compute(r);
            CopyOut(t, r);
            for (uint32_t i = 0; i < r; ++i) {
                scales.SetValue(nScale, scaleVals_[i]);
                ++nScale;
                if (nScale == SCALE_TILE) {
                    FlushScales(scales, scaleBase, nScale);
                    scaleBase = t + (int64_t)i + 1;
                    nScale = 0;
                }
            }
        }
        if (nScale != 0) {
            FlushScales(scales, scaleBase, nScale);
        }
    }

private:
    // A row of gate_up is [gate(W) | up(W)] and contiguous, so one copy moves
    // both halves. EnQue only -- Compute() does the matching DeQue, which is
    // what lets MTE2 run ahead into the next iteration's buffer.
    __aicore__ inline void CopyIn(int64_t t, uint32_t r)
    {
        const int64_t rowOff = t * (int64_t)W_ * 2;
        const uint32_t n = r * W_ * 2;  // rows are adjacent, so the tile is one run
        AscendC::LocalTensor<T> in = inQueue_.template AllocTensor<T>();
        DataCopy(in, gateUpGm_[rowOff], n);
        inQueue_.EnQue(in);
        if (hasDelta_ != 0) {
            AscendC::LocalTensor<T> d = deltaQueue_.template AllocTensor<T>();
            DataCopy(d, deltaGm_[rowOff], n);
            deltaQueue_.EnQue(d);
        }
    }

    // Fills scaleVals_[0..r) and leaves act and y enqueued for CopyOut.
    // The tile's rows are laid out [gate0|up0|gate1|up1|...] on the way in;
    // the 2r casts below de-interleave them into two contiguous r*W fp32
    // buffers, after which every step but the per-row absmax and rescale runs
    // once over the whole tile instead of once per row.
    __aicore__ inline void Compute(uint32_t r)
    {
        const uint32_t n = r * W_;
        AscendC::LocalTensor<float> g = fGate_.Get<float>();
        AscendC::LocalTensor<float> u = fUp_.Get<float>();
        AscendC::LocalTensor<float> tmp = fTmp_.Get<float>();
        AscendC::LocalTensor<float> maxs = maxBuf_.Get<float>();

        AscendC::LocalTensor<T> in = inQueue_.template DeQue<T>();
        for (uint32_t i = 0; i < r; ++i) {
            Cast(g[i * W_], in[i * 2 * W_], AscendC::RoundMode::CAST_NONE, W_);
            Cast(u[i * W_], in[i * 2 * W_ + W_], AscendC::RoundMode::CAST_NONE, W_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // freed before the long vector chain below, so the next tile's copy can
        // claim this buffer while we are still computing this one
        inQueue_.FreeTensor(in);

        if (hasDelta_ != 0) {
            AscendC::LocalTensor<T> d = deltaQueue_.template DeQue<T>();
            for (uint32_t i = 0; i < r; ++i) {
                Cast(tmp[i * W_], d[i * 2 * W_], AscendC::RoundMode::CAST_NONE, W_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(g, g, tmp, n);
            for (uint32_t i = 0; i < r; ++i) {
                Cast(tmp[i * W_], d[i * 2 * W_ + W_], AscendC::RoundMode::CAST_NONE, W_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(u, u, tmp, n);
            AscendC::PipeBarrier<PIPE_V>();
            deltaQueue_.FreeTensor(d);
        }

        // silu(g) = g / (1 + exp(-g)); written out rather than using the
        // SwiGLU high-level API so the activated half is unambiguous
        Muls(tmp, g, -1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, n);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, 1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        Div(g, g, tmp, n);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(g, g, u, n);  // g now holds the SwiGLU result
        AscendC::PipeBarrier<PIPE_V>();

        // side output: the fp activation the w2 LoRA shrink consumes
        AscendC::LocalTensor<T> actLocal = outActQueue_.template AllocTensor<T>();
        Cast(actLocal, g, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        outActQueue_.EnQue(actLocal);

        // per-token dynamic int8 quant: the absmax is per row, so this is the
        // one reduction that cannot span the tile. One V_S sync covers all r.
        Abs(tmp, g, n);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < r; ++i) {
            ReduceMax<float>(maxs[i], tmp[i * W_], tmp[i * W_], W_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        event_t evtV2S = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evtV2S);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtV2S);
        for (uint32_t i = 0; i < r; ++i) {
            float v = maxs.GetValue(i) / 127.0f;
            scaleVals_[i] = v;
            // an all-zero row would divide by zero; any non-zero scale gives
            // the same all-zero int8 output
            Muls(g[i * W_], g[i * W_], (v > 0.0f) ? (1.0f / v) : 0.0f, W_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // float -> int16 (RINT) -> half -> int8, as in the op this replaces.
        // tmp's storage is recycled here: the reduced maxima live in maxBuf_.
        AscendC::LocalTensor<int16_t> i16 = tmp.template ReinterpretCast<int16_t>();
        Cast(i16, g, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> h = i16.ReinterpretCast<half>();
        Cast(h, i16, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int8_t> yLocal = outYQueue_.template AllocTensor<int8_t>();
        Cast(yLocal, h, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_V>();
        outYQueue_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t t, uint32_t r)
    {
        const uint32_t n = r * W_;
        AscendC::LocalTensor<T> actLocal = outActQueue_.template DeQue<T>();
        DataCopy(actGm_[t * (int64_t)W_], actLocal, n);
        outActQueue_.FreeTensor(actLocal);

        AscendC::LocalTensor<int8_t> yLocal = outYQueue_.template DeQue<int8_t>();
        DataCopy(yGm_[t * (int64_t)W_], yLocal, n);
        outYQueue_.FreeTensor(yLocal);
    }

    // Rows are handed out to a core as one contiguous range, so a tile of
    // scales is contiguous in scaleGm_ too and goes out in a single copy.
    __aicore__ inline void FlushScales(const AscendC::LocalTensor<float> &scales,
                                       int64_t base, uint32_t n)
    {
        event_t evtS2MTE3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evtS2MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evtS2MTE3);

        AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(scaleGm_[base], scales, params);

        // the buffer is rewritten by scalar stores on the next tile, so the
        // copy must be done reading it first. Once per SCALE_TILE rows, so the
        // full barrier is noise.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue_, deltaQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outActQueue_, outYQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fGate_, fUp_, fTmp_, scaleBuf_, maxBuf_;
    AscendC::GlobalTensor<T> gateUpGm_, deltaGm_, actGm_;
    AscendC::GlobalTensor<int8_t> yGm_;
    AscendC::GlobalTensor<float> scaleGm_;
    uint32_t batch_, units_, unitsPerCore_, W_, hasDelta_, R_, tileElems_;
    float scaleVals_[R_MAX];
};
}  // namespace

#define ADD_LORA_SWIGLU_QUANT_DECLARE(TYPE)                                                           \
    extern "C" __global__ __aicore__ void add_lora_swiglu_quant_##TYPE(                               \
        __gm__ void* gateUp, __gm__ void* delta, __gm__ void* act, __gm__ void* y,                   \
        __gm__ void* scale, uint32_t batch, uint32_t units, uint32_t unitsPerCore,                    \
        uint32_t width, uint32_t hasDelta)                                                            \
    {                                                                                                 \
        AscendC::TPipe pipe;                                                                          \
        AddLoraSwigluQuant<TYPE> op(&pipe);                                                           \
        op.Init(gateUp, delta, act, y, scale, batch, units, unitsPerCore, width, hasDelta);           \
        op.Process();                                                                                 \
    }

ADD_LORA_SWIGLU_QUANT_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ADD_LORA_SWIGLU_QUANT_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

void add_lora_swiglu_quant_impl(AscendType type, void *stream, void *gate_up, void *delta,
                                void *act, void *y, void *scale, uint32_t batch, uint32_t width,
                                uint32_t has_delta, uint32_t aiv_num)
{
    uint32_t unitsPerCore = (batch + aiv_num - 1) / aiv_num;
    if (unitsPerCore == 0) {
        unitsPerCore = 1;
    }
    uint32_t gridDim = (batch + unitsPerCore - 1) / unitsPerCore;
    if (gridDim == 0) {
        return;
    }

    if (type == AscendType::FP16) {
        add_lora_swiglu_quant_half<<<gridDim, nullptr, stream>>>(
            gate_up, delta, act, y, scale, batch, batch, unitsPerCore, width, has_delta);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        add_lora_swiglu_quant_bfloat16_t<<<gridDim, nullptr, stream>>>(
            gate_up, delta, act, y, scale, batch, batch, unitsPerCore, width, has_delta);
#endif
    }
}
}  // namespace vllm_ascend
