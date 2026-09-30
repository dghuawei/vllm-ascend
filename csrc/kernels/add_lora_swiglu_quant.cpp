/*
 * Copyright (c) 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_operator.h"
#include "lib/activation/swiglu.h"
#include "types.h"

namespace {
constexpr uint32_t TILE_ELEMENTS = 8192;
constexpr uint32_t MAX_TOKENS_PER_TILE = 16;
constexpr uint32_t SCALE_TILE = 64;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t UB_BLOCK_FLOATS = 8;
constexpr float SWIGLU_BETA = 1.0f;
constexpr float INT8_MAX_VALUE = 127.0f;

template <typename scalar_t>
class AddLoraSwigluQuant {
public:
    __aicore__ inline AddLoraSwigluQuant(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *gateUp, __gm__ void *delta, __gm__ void *act,
                                __gm__ void *y, __gm__ void *scale, uint32_t batchSize,
                                uint32_t numTokensPerCore, uint32_t width, uint32_t hasDelta)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        width_ = width;
        hasDelta_ = hasDelta;

        numRowsPerTile_ = TILE_ELEMENTS / width_;
        if (numRowsPerTile_ == 0) {
            numRowsPerTile_ = 1;
        } else if (numRowsPerTile_ > MAX_TOKENS_PER_TILE) {
            numRowsPerTile_ = MAX_TOKENS_PER_TILE;
        }
        uint32_t tileElements = numRowsPerTile_ * width_;

        gateUpGm_.SetGlobalBuffer((__gm__ scalar_t *)gateUp);
        deltaGm_.SetGlobalBuffer((__gm__ scalar_t *)delta);
        actGm_.SetGlobalBuffer((__gm__ scalar_t *)act);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale, batchSize_);

        pipe_->InitBuffer(inQueueGateUp_, BUFFER_NUM, tileElements * 2 * sizeof(scalar_t));
        if (hasDelta_ != 0) {
            pipe_->InitBuffer(inQueueDelta_, BUFFER_NUM, tileElements * 2 * sizeof(scalar_t));
        }
        pipe_->InitBuffer(outQueueAct_, BUFFER_NUM, tileElements * sizeof(scalar_t));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, tileElements * sizeof(int8_t));
        pipe_->InitBuffer(gateBuffer_, tileElements * sizeof(float));
        pipe_->InitBuffer(upBuffer_, tileElements * sizeof(float));
        pipe_->InitBuffer(actBuffer_, tileElements * sizeof(float));
        pipe_->InitBuffer(maxBuffer_, MAX_TOKENS_PER_TILE * UB_BLOCK_FLOATS * sizeof(float));
        pipe_->InitBuffer(scaleBuffer_, SCALE_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t startIdx = AscendC::GetBlockIdx() * (int64_t)numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        endIdx = endIdx > (int64_t)batchSize_ ? batchSize_ : endIdx;

        AscendC::LocalTensor<float> scales = scaleBuffer_.Get<float>();
        int64_t scaleBase = startIdx;
        uint32_t numScales = 0;

        for (int64_t idx = startIdx; idx < endIdx; idx += numRowsPerTile_) {
            uint32_t numRows = (uint32_t)(endIdx - idx);
            numRows = numRows > numRowsPerTile_ ? numRowsPerTile_ : numRows;
            
            CopyIn(idx, numRows);
            Compute(numRows);
            CopyOut(idx, numRows);
            
            for (uint32_t i = 0; i < numRows; i++) {
                scales.SetValue(numScales, scaleValues_[i]);
                numScales++;
                if (numScales == SCALE_TILE) {
                    FlushScales(scales, scaleBase, numScales);
                    scaleBase = idx + (int64_t)i + 1;
                    numScales = 0;
                }
            }
        }
        if (numScales != 0) {
            FlushScales(scales, scaleBase, numScales);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t idx, uint32_t numRows)
    {
        int64_t offset = idx * (int64_t)width_ * 2;
        uint32_t numElements = numRows * width_ * 2;

        AscendC::LocalTensor<scalar_t> gateUpLocal = inQueueGateUp_.AllocTensor<scalar_t>();
        DataCopy(gateUpLocal, gateUpGm_[offset], numElements);
        inQueueGateUp_.EnQue(gateUpLocal);

        if (hasDelta_ != 0) {
            AscendC::LocalTensor<scalar_t> deltaLocal = inQueueDelta_.AllocTensor<scalar_t>();
            DataCopy(deltaLocal, deltaGm_[offset], numElements);
            inQueueDelta_.EnQue(deltaLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t numRows)
    {
        uint32_t numElements = numRows * width_;
        AscendC::LocalTensor<float> gate = gateBuffer_.Get<float>();
        AscendC::LocalTensor<float> up = upBuffer_.Get<float>();
        AscendC::LocalTensor<float> activated = actBuffer_.Get<float>();
        AscendC::LocalTensor<float> maxs = maxBuffer_.Get<float>();
      
        AscendC::LocalTensor<scalar_t> gateUpLocal = inQueueGateUp_.DeQue<scalar_t>();

        // unstack (gate,up) into two distinct buffers, widen to fp32
        for (uint32_t i = 0; i < numRows; i++) {
            Cast(gate[i * width_], gateUpLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            Cast(up[i * width_], gateUpLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE, width_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inQueueGateUp_.FreeTensor(gateUpLocal);

        if (hasDelta_ != 0) {
            // same for lora_delta, then add
            AscendC::LocalTensor<scalar_t> deltaLocal = inQueueDelta_.DeQue<scalar_t>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(gate, gate, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE,
                     width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(up, up, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueDelta_.FreeTensor(deltaLocal);
        }

        // swiglu
        AscendC::SwiGLU<float, false>(activated, up, gate, SWIGLU_BETA, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        
        // prepare bf_16 activations for W2 lora shink
        // TODO: check how quality degrades if we use quantized inputs instead
        AscendC::LocalTensor<scalar_t> actLocal = outQueueAct_.AllocTensor<scalar_t>();
        Cast(actLocal, activated, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueAct_.EnQue(actLocal);
        
#if !defined FAST_REDUCE
        // find scales, reuse gate buffer as tmp for abs values
        Abs(gate, activated, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < numRows; i++) {
            AscendC::ReduceMax<float>(maxs[i * UB_BLOCK_FLOATS], gate[i * width_], gate[i * width_],
                                      width_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        event_t eventVToS = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventVToS);
        for (uint32_t i = 0; i < numRows; i++) {
            scaleValues_[i] = maxs.GetValue(i * UB_BLOCK_FLOATS) / INT8_MAX_VALUE;
        }
        event_t eventSToV = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventSToV);
        for (uint32_t i = 0; i < numRows; i++) {
            float reciprocal = (scaleValues_[i] > 0.0f) ? (1.0f / scaleValues_[i]) : 0.0f;
            Muls(activated[i * width_], activated[i * width_], reciprocal, width_);
        }
        AscendC::PipeBarrier<PIPE_V>();
#else 
#endif 
        // there is no fp32 -> int8 cast, so we do fp32->half->int8
        AscendC::LocalTensor<half> halfLocal = up.ReinterpretCast<half>();
        Cast(halfLocal, activated, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY_.AllocTensor<int8_t>();
        Cast(yLocal, halfLocal, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t idx, uint32_t numRows)
    {
        int64_t offset = idx * (int64_t)width_;
        uint32_t numElements = numRows * width_;

        AscendC::LocalTensor<scalar_t> actLocal = outQueueAct_.DeQue<scalar_t>();
        DataCopy(actGm_[offset], actLocal, numElements);
        outQueueAct_.FreeTensor(actLocal);

        AscendC::LocalTensor<int8_t> yLocal = outQueueY_.DeQue<int8_t>();
        DataCopy(yGm_[offset], yLocal, numElements);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void FlushScales(const AscendC::LocalTensor<float> &scales, int64_t base,
                                       uint32_t numScales)
    {
        event_t eventSToMte3 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventSToMte3);

        AscendC::DataCopyExtParams params{1, numScales * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(scaleGm_[base], scales, params);

        event_t eventMte3ToS = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventMte3ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventMte3ToS);
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueGateUp_, inQueueDelta_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueAct_, outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gateBuffer_, upBuffer_, actBuffer_, maxBuffer_, scaleBuffer_;
    AscendC::GlobalTensor<scalar_t> gateUpGm_;
    AscendC::GlobalTensor<scalar_t> deltaGm_;
    AscendC::GlobalTensor<scalar_t> actGm_;
    AscendC::GlobalTensor<int8_t> yGm_;
    AscendC::GlobalTensor<float> scaleGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t width_;
    uint32_t hasDelta_;
    uint32_t numRowsPerTile_;
    float scaleValues_[MAX_TOKENS_PER_TILE];
};
}  // namespace

#define ADD_LORA_SWIGLU_QUANT_TYPE_DECLARE(TYPE)                                                     \
    extern "C" __global__ __aicore__ void add_lora_swiglu_quant_##TYPE(                              \
        __gm__ void* gateUp, __gm__ void* delta, __gm__ void* act, __gm__ void* y,                    \
        __gm__ void* scale, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t width,            \
        uint32_t hasDelta)                                                                           \
    {                                                                                                \
        AscendC::TPipe pipe;                                                                         \
        AddLoraSwigluQuant<TYPE> op(&pipe);                                                          \
        op.Init(gateUp, delta, act, y, scale, batchSize, numTokensPerCore, width, hasDelta);          \
        op.Process();                                                                                \
    }

// declare all dtype kernel
ADD_LORA_SWIGLU_QUANT_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ADD_LORA_SWIGLU_QUANT_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void add_lora_swiglu_quant_impl(AscendType type, void *stream, void *gate_up, void *delta,
                                       void *act, void *y, void *scale, uint32_t batch,
                                       uint32_t width, uint32_t has_delta, uint32_t aiv_num)
{
    uint32_t numTokensPerCore = (batch + aiv_num - 1) / aiv_num;
    if (numTokensPerCore == 0) {
        numTokensPerCore = 1;
    }
    uint32_t blockDim = (batch + numTokensPerCore - 1) / numTokensPerCore;

    if (type == AscendType::FP16) {
        add_lora_swiglu_quant_half<<<blockDim, nullptr, stream>>>(
            gate_up, delta, act, y, scale, batch, numTokensPerCore, width, has_delta);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        add_lora_swiglu_quant_bfloat16_t<<<blockDim, nullptr, stream>>>(
            gate_up, delta, act, y, scale, batch, numTokensPerCore, width, has_delta);
#endif
    }
}
}  // namespace vllm_ascend
