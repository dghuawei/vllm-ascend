/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_row_split.h
 * \brief Scatter Kernel (RowSplit) -- splits UPDATE ROWS across cores.
 *
 * Why this exists
 * ---------------
 * The NoSort kernel splits the OUTPUT ADDRESS SPACE across cores: core c owns
 * [start_, end_) of the destination and scans the whole index list, keeping only
 * the indices that land inside its own range. That makes duplicate indices
 * deterministic -- one core owns every write to a given slot and visits them in
 * index order -- but it costs:
 *   - a full index scan replicated on every core, and
 *   - all the work landing in one or two cores whenever the indices cluster,
 *     which is exactly what a KV-cache slot_mapping does: a near-contiguous run
 *     of slots.
 * Measured on a [11167,32,1,512] bf16 cache with 8271 rows: 1340 us of wall time
 * with aiv_time at 203 us, i.e. the cores idle 85% of the time; and a per-row
 * cost that HALVED going from 2068 to 8271 rows, which is the signature of one
 * core becoming two rather than of parallel work being added.
 *
 * This kernel splits the ROWS instead. Core c handles rows [rowStart_, rowEnd_)
 * of indices/updates and writes wherever those indices point, so the split is
 * even whatever the index distribution and no core reads another core's indices.
 * The trade is that two updates carrying the SAME index may be handled by
 * different cores concurrently, leaving it unspecified which one lands -- which
 * is precisely what the op's `use_locking=false` attribute licenses. The host
 * selects this tiling key only when use_locking is false.
 *
 * Shape of the loop
 * -----------------
 * Each core reads its whole index slice once (it is 4 bytes per row; the host
 * guarantees it fits) and pays ONE scalar sync for it, instead of the two full
 * pipe barriers per row that ProcessOneIndex pays. The update rows then stream
 * through a genuinely double-buffered queue: batch i+1's MTE2 load is issued
 * before batch i's MTE3 writes, so the load of the next batch overlaps the
 * scatter of the current one.
 */

#ifndef SCATTER_ND_UPDATE_ROW_SPLIT_H
#define SCATTER_ND_UPDATE_ROW_SPLIT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {

template<typename T>
class ScatterNdUpdateV2KernelRowSplit {
public:
    __aicore__ inline ScatterNdUpdateV2KernelRowSplit() = delete;
    __aicore__ inline ScatterNdUpdateV2KernelRowSplit(
        GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParam(tiling);
        InitBuffers(pipe);
        SetGmAddr(updates, output, workSpace);
    }

    __aicore__ inline void InitParam(const ScatterNdUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        totalIndexRow_ = tiling.linearIndexTiling.blockNum * tiling.linearIndexTiling.blockLength
                         + tiling.linearIndexTiling.blockRemainLength;

        // For this tiling key the host distributed INDEX ROWS, not output
        // elements, so frontNum/frontRow/tailRow describe a row range here.
        CalcBlockDistribution(blockIdx_, tiling.scatterTiling.frontNum, tiling.scatterTiling.frontRow,
                              tiling.scatterTiling.tailRow, computeRow_, rowStart_);
        if (rowStart_ > totalIndexRow_) {
            rowStart_ = totalIndexRow_;
        }
        rowEnd_ = rowStart_ + computeRow_;
        if (rowEnd_ > totalIndexRow_) {
            rowEnd_ = totalIndexRow_;
        }
        computeRow_ = rowEnd_ - rowStart_;

        scatterLength_ = tiling.scatterTiling.scatterLength;
        outputPhysicalRange_ = tiling.scatterTiling.outputPhysicalRange;
        scatterAlignLength_ = tiling.scatterTiling.scatterAlignLength;
        rowBatch_ = tiling.scatterTiling.rowBatch;
        if (rowBatch_ == 0) {
            rowBatch_ = 1;
        }
        updateBytes_ = static_cast<uint32_t>(scatterLength_ * sizeof(T));
        // A row spanning a whole number of 32-byte blocks can be fetched for the
        // entire batch in ONE multi-block DataCopyPad, packed (dstStride = 0),
        // and then row i sits at i * scatterLength_ == i * scatterAlignLength_.
        // Otherwise each row is fetched into its own 32-byte-aligned UB slot,
        // because a LocalTensor operand must start 32-byte aligned.
        rowPacked_ = (updateBytes_ % ALIGNED_BLOCK_NUM == 0);
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        // One int32 per row of this core's slice. The host only picks this tiling
        // key when that stays small (see ROW_SPLIT_INDEX_LIMIT in the tiling).
        uint64_t indexBytes = (computeRow_ * sizeof(int) + ALIGNED_BLOCK_NUM - 1)
                              / ALIGNED_BLOCK_NUM * ALIGNED_BLOCK_NUM;
        if (indexBytes == 0) {
            indexBytes = ALIGNED_BLOCK_NUM;
        }
        pipe.InitBuffer(indexBuf_, indexBytes);
        // One TBuf holding BOTH halves, not a TQue. This buffer is produced by MTE2
        // (DataCopyPad in) and consumed by MTE3 (DataCopyPad out), and a TQue cannot
        // express that pair: its position fixes the producer/consumer pipes
        // (VECIN = MTE2->V, VECOUT = V->MTE3), so whichever one is chosen, the
        // EnQue/DeQue and Alloc/Free events land on the VECTOR pipe and neither the
        // read-after-write nor the buffer-reuse hazard is actually covered. The
        // MTE2_MTE3 / MTE3_MTE2 flags below are issued by hand instead.
        batchElems_ = rowBatch_ * scatterAlignLength_;
        pipe.InitBuffer(updateBuf_, ROW_SPLIT_BUFFER_NUM * batchElems_ * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace)
    {
        // The LinearIndexKernel pre-pass wrote one int32 linear index per row at
        // the front of the workspace, in row order.
        linearIndicesGm_.SetGlobalBuffer((__gm__ int*)workSpace);
        updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        if (computeRow_ == 0) {
            return;
        }
        LoadIndexSlice();
        updateLocal_ = updateBuf_.Get<T>();

        uint64_t row = rowStart_;
        uint64_t curRow = BatchLen(row);
        // Batch b uses half (b % 2); batch b and batch b+2 share one half, so from
        // the third batch on every copy-in must first wait for the writes of the
        // batch two back to have drained out of that half.
        uint32_t buf = 0;
        CopyInUpdates(buf, row, curRow);
        while (curRow != 0) {
            uint64_t nextRow = row + curRow;
            uint64_t nextRowLen = (nextRow < rowEnd_) ? BatchLen(nextRow) : 0;
            uint32_t nextBuf = buf ^ 1u;
            if (nextRowLen != 0) {
                // Issued before this batch's writes so the next load overlaps them.
                CopyInUpdates(nextBuf, nextRow, nextRowLen);
            }
            ScatterOutUpdates(buf, row, curRow);
            row = nextRow;
            curRow = nextRowLen;
            buf = nextBuf;
        }
        DrainOutstanding();
    }

private:
    __aicore__ inline uint64_t BatchLen(uint64_t row)
    {
        uint64_t remain = rowEnd_ - row;
        return remain < rowBatch_ ? remain : rowBatch_;
    }

    // Whole slice of linear indices, one MTE2 and one scalar sync for the core.
    __aicore__ inline void LoadIndexSlice()
    {
        indexLocal_ = indexBuf_.Get<int>();
        DataCopyExtParams indexParams{1, static_cast<uint32_t>(computeRow_ * sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int> indexPad{false, 0, 0, 0};
        DataCopyPad(indexLocal_, linearIndicesGm_[rowStart_], indexParams, indexPad);
        PipeMte2ToS();
    }

    // Four DISTINCT ids. EVENT_IDn indexes one shared set of flag registers, not a
    // per-pipe-pair set, so reusing an id across the two directions would let a
    // WaitFlag be released by the wrong SetFlag -- which shows up as a handful of
    // torn rows per launch, varying run to run.
    __aicore__ inline event_t InEvent(uint32_t buf) const
    {
        return static_cast<event_t>(buf == 0 ? EVENT_ID0 : EVENT_ID1);
    }

    __aicore__ inline event_t OutEvent(uint32_t buf) const
    {
        return static_cast<event_t>(buf == 0 ? EVENT_ID2 : EVENT_ID3);
    }

    __aicore__ inline void CopyInUpdates(uint32_t buf, uint64_t row, uint64_t curRow)
    {
        // Write-after-read: this half may still be being read out by the batch two
        // back. Nothing else guards it -- MTE2 and MTE3 are independent pipes.
        if (outPending_[buf]) {
            WaitFlag<HardEvent::MTE3_MTE2>(OutEvent(buf));
            outPending_[buf] = false;
        }
        LocalTensor<T> updateLocal = updateLocal_[buf * batchElems_];
        DataCopyPadExtParams<T> updatePad{false, 0, 0, 0};
        if (rowPacked_) {
            DataCopyExtParams updateParams{static_cast<uint16_t>(curRow), updateBytes_, 0, 0, 0};
            DataCopyPad(updateLocal, updatesGm_[row * scatterLength_], updateParams, updatePad);
        } else {
            DataCopyExtParams updateParams{1, updateBytes_, 0, 0, 0};
            for (uint64_t i = 0; i < curRow; ++i) {
                DataCopyPad(updateLocal[i * scatterAlignLength_], updatesGm_[(row + i) * scatterLength_],
                            updateParams, updatePad);
            }
        }
        // Read-after-write for the scatter-out of THIS batch.
        SetFlag<HardEvent::MTE2_MTE3>(InEvent(buf));
        inPending_[buf] = true;
    }

    __aicore__ inline void ScatterOutUpdates(uint32_t buf, uint64_t row, uint64_t curRow)
    {
        WaitFlag<HardEvent::MTE2_MTE3>(InEvent(buf));
        inPending_[buf] = false;
        LocalTensor<T> updateData = updateLocal_[buf * batchElems_];
        uint64_t indexOffset = row - rowStart_;
        DataCopyExtParams outParams{1, updateBytes_, 0, 0, 0};
        for (uint64_t i = 0; i < curRow; ++i) {
            int64_t linearIndex = static_cast<int64_t>(indexLocal_.GetValue(indexOffset + i));
            // PAD slots. The caller pads slot_mapping with -1, which reaches us as
            // a negative linear index. NoSort drops such a row implicitly -- it
            // keeps only indices inside the core's own output range, and an
            // out-of-range index is inside nobody's -- but this kernel addresses
            // outputGm_ directly, so it has to say so.
            if (linearIndex < 0 ||
                static_cast<uint64_t>(linearIndex) + scatterLength_ > outputPhysicalRange_) {
                continue;
            }
            // Issued back to back: consecutive MTE3 copies pipeline in order.
            DataCopyPad(outputGm_[linearIndex], updateData[i * scatterAlignLength_], outParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(OutEvent(buf));
        outPending_[buf] = true;
    }

    // Every flag set must be waited on before the kernel ends, or it is left
    // standing for whatever runs next on this core.
    __aicore__ inline void DrainOutstanding()
    {
        for (uint32_t buf = 0; buf < ROW_SPLIT_BUFFER_NUM; ++buf) {
            if (inPending_[buf]) {
                WaitFlag<HardEvent::MTE2_MTE3>(InEvent(buf));
                inPending_[buf] = false;
            }
            if (outPending_[buf]) {
                WaitFlag<HardEvent::MTE3_MTE2>(OutEvent(buf));
                outPending_[buf] = false;
            }
        }
    }

    GlobalTensor<int> linearIndicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> updateBuf_;
    LocalTensor<int> indexLocal_;
    LocalTensor<T> updateLocal_;

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t rowStart_;
    uint64_t rowEnd_;
    uint64_t totalIndexRow_;
    uint64_t scatterLength_;
    uint64_t outputPhysicalRange_;
    uint64_t scatterAlignLength_;
    uint64_t rowBatch_;
    uint64_t batchElems_;
    uint32_t updateBytes_;
    bool rowPacked_;
    bool inPending_[ROW_SPLIT_BUFFER_NUM] = {false, false};
    bool outPending_[ROW_SPLIT_BUFFER_NUM] = {false, false};
};

} // namespace ScatterNdUpdateV2

#endif // SCATTER_ND_UPDATE_ROW_SPLIT_H
