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
 * \file scatter_nd_update_v2_tiling.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_key.h"
#include "scatter_nd_update_v2_tiling.h"

namespace optiling {
// using namespace Ops::NN::Optiling;
constexpr uint64_t MAX_DIM_NUM = 8;
constexpr uint64_t MAX_LENGTH_INT32 = (1LL << 31) - 1;
constexpr uint64_t MAX_FLOAT_EXPRESS_INT32 = (1LL << 24) - 1;
constexpr uint64_t SORT_USE_GM_NUM = 2;
constexpr uint64_t SORT_BLOCK_LENGTH = 4096;
constexpr uint64_t GATHER_USE_NUM = 2;
constexpr uint64_t ALIGNED_NUM = 8;
constexpr uint64_t ALIGNED_SIZE = 32;
constexpr uint64_t ATTR_STRIDE = 0;
constexpr uint64_t ATTR_USE_LOCKING = 1;
// RowSplit 路径：真正的双缓冲，所以每批的 UB 预算要除以 2
constexpr uint64_t ROW_SPLIT_BUFFER_NUM = 2;
// DataCopyExtParams::blockCount 的上限
constexpr uint64_t MAX_BLOCK_COUNT = 4095;
// RowSplit 的每个核把自己那段行索引一次性读进 UB（每行 4 字节），只付一次 scalar
// 同步。这段 UB 的上限；超过则说明单核行数太多，退回 NoSort 路径。
constexpr uint64_t ROW_SPLIT_INDEX_LIMIT = 16 * 1024;
class ScatterNdUpdateV2Tiling {
public:
    explicit ScatterNdUpdateV2Tiling(gert::TilingContext* context) : tilingContext_(context){}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline bool IsSort(uint64_t totalLength, uint64_t indexRow);
    inline bool IsLinearIndex(uint64_t totalLength);
    inline bool CanRowSplit(uint64_t indexRow);
    inline size_t CalcWorkSpaceSize(uint64_t indexRow);
    inline void SetTilingKeyMode();
    inline void GetDtypeSize();
    inline void Tiling4Scatter(uint64_t totalLength, uint64_t indexRow);
    inline void Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim);

    ScatterNdUpdateV2TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;

    uint64_t coreNum_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t isLinearIndex_ = false;
    uint64_t isSort_ = false;
    uint64_t sortWorkspace_ = 0;
    uint64_t dataTypeSize_ = 0;
    uint64_t isInt64Indices_ = false;
    uint64_t needLargeIndexKernel_ = false;
    uint64_t useLocking_ = false;
    uint64_t isRowSplit_ = false;
    uint64_t rowBatch_ = 0;
    // Largest valid linear index + one row; the RowSplit kernel drops any
    // index outside [0, this), which is what keeps PAD slots in bounds.
    uint64_t outputPhysicalRange_ = 0;

private:
    // LinearIndex
    uint64_t indexDim_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t blockNum_ = 0;
    uint64_t blockRemainLength_ = 0;
    uint64_t tailBlockNum_ = 0;
    uint64_t frontBlockNum_ = 0;
    uint64_t frontCoreNum_ = 0;
    uint64_t tailCoreNum_ =  0;
    uint64_t indicesMask_[MAX_DIM_NUM] = {0};

    // Scatter
    uint64_t scatterLength_ = 1;
    uint64_t tailRow_ = 0;
    uint64_t frontRow_ = 0;
    uint64_t frontNum_ = 0;
    uint64_t tailNum_ = 0;
    uint64_t ubLengthForUpdates_ = 0;
    uint64_t scatterAlignLength_ = 0;
    uint64_t formDim_ = 0;
    uint64_t copyRow_ = 0;
    uint64_t scatterTileNum_ = 1;
    uint64_t scatterTileLength_ = 0;
    uint64_t scatterTileTail_ = 0;
    uint64_t scatterTileAlignLength_ = 0;
};

inline void ScatterNdUpdateV2Tiling::SetTilingKeyMode()
{
    // tilingKey: indexType * 10 + modeFlag (indexType: 1=int32, 2=int64(cast), 3=int64(large))
    // modeFlag: 0=非排序(按输出地址切分), 1=排序, 2=按 updates 行切分
    uint64_t indexType;
    if (!isInt64Indices_) {
        indexType = 1;
    } else if (needLargeIndexKernel_) {
        indexType = 3;
    } else {
        indexType = 2;
    }
    uint64_t modeFlag;
    if (indexType == 3) {
        modeFlag = 0;
    } else if (isRowSplit_) {
        modeFlag = 2;
    } else {
        modeFlag = isSort_ ? 1 : 0;
    }
    tilingKey_ = indexType * 10 + modeFlag;

    tilingContext_->SetTilingKey(tilingKey_);
    OP_LOGD(tilingContext_, "isLinearIndex=%lu, isSort=%lu, isRowSplit=%lu, rowBatch=%lu, useLocking=%lu, isInt64Indices=%lu, needLargeIndexKernel=%lu, tilingKey=%lu (indexType=%lu, modeFlag=%lu)",
            isLinearIndex_, isSort_, isRowSplit_, rowBatch_, useLocking_, isInt64Indices_, needLargeIndexKernel_,
            tilingKey_, indexType, modeFlag);
}

inline bool ScatterNdUpdateV2Tiling::IsLinearIndex(uint64_t totalLength)
{
    return totalLength <= MAX_LENGTH_INT32;
}

inline bool ScatterNdUpdateV2Tiling::IsSort(uint64_t totalLength, uint64_t indexRow)
{
    return totalLength <= MAX_FLOAT_EXPRESS_INT32;
}

/*
 * 是否可以按 updates 的行切分到各个核（RowSplit 路径）。
 *
 * 默认的 NoSort 路径把 **输出地址空间** 切给各核，每个核再扫描整个索引表、只保留
 * 落在自己区间内的索引。这样重复索引的结果是确定的（同一个槽位的所有写入都由同一
 * 个核按索引顺序执行），但当索引聚集时 —— KV cache 的 slot_mapping 正是一段近乎
 * 连续的槽位 —— 所有行都落进一两个核的区间，其余 AI vector core 全部空转。
 * 按行切分则与索引分布无关，恒定均匀；代价是两个相同索引可能由不同核并发写入，
 * 谁最后落盘不确定 —— 这正是 use_locking=false 所允许的。
 */
inline bool ScatterNdUpdateV2Tiling::CanRowSplit(uint64_t indexRow)
{
    // use_locking=true 表示调用方要求确定性，必须保留按输出地址切分的路径。
    // LargeIndex 路径没有 linearIndex 预处理，workspace 里没有行索引可读。
    if (useLocking_ || needLargeIndexKernel_ || indexRow == 0 || scatterLength_ == 0) {
        return false;
    }
    uint64_t reserved = SORT_BLOCK_LENGTH * SORT_USE_GM_NUM * sizeof(int);
    if (dataTypeSize_ == 0 || ubSize_ <= reserved) {
        return false;
    }
    uint64_t scatterAlignNum = ALIGNED_SIZE / dataTypeSize_;
    uint64_t alignLength = (scatterLength_ + scatterAlignNum - 1) & ~(scatterAlignNum - 1);
    uint64_t budget = (ubSize_ - reserved) / ALIGNED_SIZE * ALIGNED_SIZE;
    uint64_t rowBytes = alignLength * dataTypeSize_;
    // 双缓冲下单行必须放得进半个预算，否则退回按行分块搬运的 NoSort 路径。
    if (rowBytes == 0 || rowBytes > budget / ROW_SPLIT_BUFFER_NUM) {
        return false;
    }
    // 单核最多承担的行数（Tiling4Scatter 会按行均分，frontRow 是上界）。
    uint64_t rowsPerCore = (indexRow + coreNum_ - 1) / coreNum_;
    uint64_t indexBytes = (rowsPerCore * sizeof(int32_t) + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
    if (indexBytes > ROW_SPLIT_INDEX_LIMIT || indexBytes >= budget) {
        return false;
    }
    uint64_t avail = budget - indexBytes;
    if (rowBytes > avail / ROW_SPLIT_BUFFER_NUM) {
        return false;
    }
    uint64_t batch = avail / (ROW_SPLIT_BUFFER_NUM * rowBytes);
    // EXP-10 probe: halve the UB-derived batch to test flag-handshake
    // sensitivity. Only binds where the UB batch (31) beats rowsPerCore.
    batch = (batch <= 1) ? 1 : batch / 2;
    if (batch == 0) {
        return false;
    }
    // 不必超过单核的行数，多出来的只是白占 UB。
    rowBatch_ = std::min(batch, std::min(rowsPerCore, MAX_BLOCK_COUNT));
    // EXP-17/19/20/21: width-1 单核只有一个 batch，MTE2/MTE3 完全串行；
    // 分批打开双缓冲重叠。加深持续有收益且每次 paired 3/3:
    // 64->32 -10.4%, 32->16 -4.7%, 16->8 -2.9%。8 行为轴终点。
    if (scatterLength_ == 1) {
        rowBatch_ = std::min(rowBatch_, (uint64_t)8);
    }
    // EXP-18: int8 w128 同理，单批 -> 3 批，打开双缓冲重叠。
    // EXP-18b: int8 w128 sits in the single-batch regime too: rowsPerCore
    // = 2068/22 = 94 == rowBatch, so one batch per core, no MTE2/MTE3
    // overlap. Paired A/B 3/3 pairs, mean -2.6%. 32 rows = 4 KB x2, fits.
    if (dataTypeSize_ == 1 && scatterLength_ == 128) {
        rowBatch_ = std::min(rowBatch_, (uint64_t)32);
    }
    return true;
}

inline void ScatterNdUpdateV2Tiling::Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim)
{
    OP_LOGD(tilingContext_, "linearIndexTiling start");
    auto attrs = tilingContext_->GetAttrs();
    auto stridesPtr = attrs->GetListInt(ATTR_STRIDE);
    for (uint64_t i = 0; i < indexDim; ++i) {
        indicesMask_[i] = static_cast<uint64_t>(stridesPtr->GetData()[i]);
    }
    uint64_t coeff = isInt64Indices_ ? (2 * indexDim + 3) : (indexDim + 3);
    uint64_t maxBlockLength = ubSize_ / coeff / sizeof(int);
    blockLength_ = (maxBlockLength / ALIGNED_SIZE) * ALIGNED_SIZE;
    blockLength_ = std::min(blockLength_, (uint64_t)SORT_BLOCK_LENGTH);
    // EXP-29/30: blockLength 3264 parks the whole linearIndex pre-pass on
    // one core (at 8271 rows that core computes 3264 + 2543-remainder = 5807
    // of 8271 rows) while every other core waits at SyncAll. Clamping to 256
    // spreads it over 33 blocks/core-chunks; paired 3/3 with separated ranges
    // on all three 8271 distributions, mean -9.4% (EXP-30). The same clamp at
    // 2068 rows costs +1.0 us for a reason I cannot explain (EXP-29), so the
    // gate is restricted to indexRow >= 4096.
    if (!useLocking_ && indexRow >= 4096) {
        blockLength_ = std::min(blockLength_, (uint64_t)256);
    }
    blockNum_ = indexRow / blockLength_;
    blockRemainLength_ = indexRow % blockLength_;

    if (blockNum_ == 0) {
        tailBlockNum_ = 0;
        frontBlockNum_ = 0;
        frontCoreNum_ = 1;
        tailCoreNum_ = 0;
    } else {
        tailBlockNum_ = blockNum_ / coreNum_;
        frontBlockNum_ = tailBlockNum_ + 1;
        frontCoreNum_ = blockNum_ % coreNum_;
        tailCoreNum_ =  tailBlockNum_ == 0 ? 0 : coreNum_ - frontCoreNum_;
    }
    OP_LOGD(tilingContext_, "linearIndexTiling finish");
}

inline void ScatterNdUpdateV2Tiling::Tiling4Scatter(uint64_t totalLength, uint64_t indexRow)
{
    OP_LOGD(tilingContext_, "scatterTiling start new");
    uint64_t scatterAlignNum = ALIGNED_SIZE / dataTypeSize_;
    tailRow_ = totalLength / coreNum_;
    frontRow_ = tailRow_ + 1;
    frontNum_ = totalLength % coreNum_;
    tailNum_ = tailRow_ == 0 ? 0 : coreNum_ - frontNum_;
    ubLengthForUpdates_ = ((ubSize_ - SORT_BLOCK_LENGTH * SORT_USE_GM_NUM * sizeof(int)) / ALIGNED_SIZE * ALIGNED_SIZE) / dataTypeSize_;
    scatterAlignLength_ = (scatterLength_ + scatterAlignNum - 1) & ~(scatterAlignNum - 1);
    formDim_ = scatterAlignLength_ / ubLengthForUpdates_;

    scatterTileLength_ = std::min(scatterLength_, ubLengthForUpdates_);
    if (scatterTileLength_ == 0) {
        scatterTileLength_ = 1;
    }
    scatterTileNum_ = (scatterLength_ + scatterTileLength_ - 1) / scatterTileLength_;
    scatterTileTail_ = scatterLength_ - (scatterTileNum_ - 1) * scatterTileLength_;
    scatterTileAlignLength_ = (scatterTileLength_ + scatterAlignNum - 1) & ~(scatterAlignNum - 1);

    if (scatterTileNum_ > 1) {
        copyRow_ = 1;
    } else {
        copyRow_ = formDim_ == 0 ? ubLengthForUpdates_ / scatterAlignLength_ : 1;
    }
    OP_LOGD(tilingContext_, "scatterTiling finish");
}

inline void ScatterNdUpdateV2Tiling::GetDtypeSize()
{
    uint64_t varDtype = tilingContext_->GetInputDesc(0)->GetDataType();
    switch (varDtype){
        case ge::DT_FLOAT:
            dataTypeSize_ = 4;
            break;
        case ge::DT_BF16:
            dataTypeSize_ = 2;
            break;
        case ge::DT_FLOAT16:
            dataTypeSize_ = 2;
            break;
        case ge::DT_BOOL:
            dataTypeSize_ = 1;
            break;
        case ge::DT_INT64:
            dataTypeSize_ = 8;
            break;
        case ge::DT_INT32:
            dataTypeSize_ = 4;
            break;
        case ge::DT_INT16:
            dataTypeSize_ = 2;
            break;
        case ge::DT_INT8:
            dataTypeSize_ = 1;
            break;
        default:
            break;
    }
}


ge::graphStatus ScatterNdUpdateV2Tiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    tilingData_.linearIndexTiling.set_indexDim(indexDim_);
    tilingData_.linearIndexTiling.set_ubSize(ubSize_);
    tilingData_.linearIndexTiling.set_indicesMask(indicesMask_);
    tilingData_.linearIndexTiling.set_coreNum(coreNum_);
    tilingData_.linearIndexTiling.set_blockLength(blockLength_);
    tilingData_.linearIndexTiling.set_blockNum(blockNum_);
    tilingData_.linearIndexTiling.set_blockRemainLength(blockRemainLength_);
    tilingData_.linearIndexTiling.set_tailBlockNum(tailBlockNum_);
    tilingData_.linearIndexTiling.set_frontBlockNum(frontBlockNum_);
    tilingData_.linearIndexTiling.set_frontCoreNum(frontCoreNum_);
    tilingData_.linearIndexTiling.set_tailCoreNum(tailCoreNum_);
    tilingData_.linearIndexTiling.set_sortWorkspace(sortWorkspace_);
    tilingData_.linearIndexTiling.set_isInt64Indices(isInt64Indices_);
    tilingData_.linearIndexTiling.set_needLargeIndexKernel(needLargeIndexKernel_);
    tilingData_.scatterTiling.set_scatterLength(scatterLength_);
    tilingData_.scatterTiling.set_tailRow(tailRow_);
    tilingData_.scatterTiling.set_frontRow(frontRow_);
    tilingData_.scatterTiling.set_frontNum(frontNum_);
    tilingData_.scatterTiling.set_tailNum(tailNum_);
    tilingData_.scatterTiling.set_ubLengthForUpdates(ubLengthForUpdates_);
    tilingData_.scatterTiling.set_scatterAlignLength(scatterAlignLength_);
    tilingData_.scatterTiling.set_formDim(formDim_);
    tilingData_.scatterTiling.set_copyRow(copyRow_);
    tilingData_.scatterTiling.set_scatterTileNum(scatterTileNum_);
    tilingData_.scatterTiling.set_scatterTileLength(scatterTileLength_);
    tilingData_.scatterTiling.set_scatterTileTail(scatterTileTail_);
    tilingData_.scatterTiling.set_scatterTileAlignLength(scatterTileAlignLength_);
    tilingData_.scatterTiling.set_rowBatch(rowBatch_);
    tilingData_.scatterTiling.set_outputPhysicalRange(outputPhysicalRange_);
    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

inline size_t ScatterNdUpdateV2Tiling::CalcWorkSpaceSize(uint64_t indexRow)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t indexRowAligned = (indexRow + ALIGNED_NUM - 1) & ~(ALIGNED_NUM - 1);
    sortWorkspace_ = indexRowAligned;
    size_t totalWorkspace = sysWorkspaceSize;
    if (isLinearIndex_) {
        totalWorkspace += sortWorkspace_ * SORT_USE_GM_NUM * sizeof(int);
    }
    if (isSort_) {
        totalWorkspace += sortWorkspace_ * SORT_USE_GM_NUM * sizeof(int);
    }
    return totalWorkspace;
}

void ScatterNdUpdateV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "coreNum:                   %lu", coreNum_);
    OP_LOGD(tilingContext_, "tilingKey:                 %lu", tilingKey_);
    OP_LOGD(tilingContext_, "isInt64Indices:            %lu", isInt64Indices_);
    OP_LOGD(tilingContext_, "needLargeIndexKernel:      %lu", needLargeIndexKernel_);
    OP_LOGD(tilingContext_, "useLocking:                %lu", useLocking_);
    OP_LOGD(tilingContext_, "isRowSplit:                %lu", isRowSplit_);
    OP_LOGD(tilingContext_, "outputPhysicalRange:       %lu", outputPhysicalRange_);
    OP_LOGD(tilingContext_, "rowBatch:                  %lu", rowBatch_);
    OP_LOGD(tilingContext_, "tiling for LinearIndex--------");
    OP_LOGD(tilingContext_, "indexDim:                  %lu", indexDim_);
    OP_LOGD(tilingContext_, "ubSize:                    %lu", ubSize_);
    OP_LOGD(tilingContext_, "blockLength:               %lu", blockLength_);
    OP_LOGD(tilingContext_, "blockNum:                  %lu", blockNum_);
    OP_LOGD(tilingContext_, "blockRemainLength:         %lu", blockRemainLength_);
    OP_LOGD(tilingContext_, "tailBlockNum:              %lu", tailBlockNum_);
    OP_LOGD(tilingContext_, "frontBlockNum:             %lu", frontBlockNum_);
    OP_LOGD(tilingContext_, "frontCoreNum:              %lu", frontCoreNum_);
    OP_LOGD(tilingContext_, "tailCoreNum:               %lu", tailCoreNum_);
    OP_LOGD(tilingContext_, "sortWorkspace:             %lu", sortWorkspace_);
    for (size_t i = 0; i < indexDim_; i++) {
        OP_LOGD(tilingContext_, "indicesMask[%lu]:            %lu", i, indicesMask_[i]);
    }
    OP_LOGD(tilingContext_, "tiling for Scatter------------");
    OP_LOGD(tilingContext_, "scatterLength:             %lu", scatterLength_);
    OP_LOGD(tilingContext_, "tailRow:                   %lu", tailRow_);
    OP_LOGD(tilingContext_, "frontRow:                  %lu", frontRow_);
    OP_LOGD(tilingContext_, "frontNum:                  %lu", frontNum_);
    OP_LOGD(tilingContext_, "tailNum:                   %lu", tailNum_);
    OP_LOGD(tilingContext_, "ubLengthForUpdates:        %lu", ubLengthForUpdates_);
    OP_LOGD(tilingContext_, "scatterAlignLength:        %lu", scatterAlignLength_);
    OP_LOGD(tilingContext_, "formDim:                   %lu", formDim_);
    OP_LOGD(tilingContext_, "copyRow:                   %lu", copyRow_);
    OP_LOGD(tilingContext_, "scatterTileNum:            %lu", scatterTileNum_);
    OP_LOGD(tilingContext_, "scatterTileLength:         %lu", scatterTileLength_);
    OP_LOGD(tilingContext_, "scatterTileTail:           %lu", scatterTileTail_);
    OP_LOGD(tilingContext_, "scatterTileAlignLength:    %lu", scatterTileAlignLength_);
}

ge::graphStatus ScatterNdUpdateV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling initing");
    auto compileInfo = static_cast<const ScatterNdUpdateV2CompileInfo*>(tilingContext_->GetCompileInfo());
    auto varRefShape = tilingContext_->GetInputShape(0)->GetStorageShape();
    auto indicesShape = tilingContext_->GetInputShape(1)->GetStorageShape();
    auto updatesShape = tilingContext_->GetInputShape(2)->GetStorageShape();
    uint64_t varDimNum = varRefShape.GetDimNum();
    indexDim_ = indicesShape.GetDim(indicesShape.GetDimNum() - 1);

    auto initAttrs = tilingContext_->GetAttrs();
    const bool* useLockingPtr = (initAttrs == nullptr) ? nullptr : initAttrs->GetAttrPointer<bool>(ATTR_USE_LOCKING);
    useLocking_ = (useLockingPtr != nullptr && *useLockingPtr) ? 1 : 0;

    auto indicesDtype = tilingContext_->GetInputDesc(1)->GetDataType();
    isInt64Indices_ = (indicesDtype == ge::DT_INT64);
    OP_LOGD(tilingContext_, "indicesDtype=%d, isInt64Indices=%lu", indicesDtype, isInt64Indices_);

    uint64_t totalLength = 1;
    for (uint64_t i = 0; i < indexDim_; ++i) {
        totalLength *= varRefShape.GetDim(i);
    }

    if (isInt64Indices_) {
        needLargeIndexKernel_ = !IsLinearIndex(totalLength);
    }

    if (varDimNum > indexDim_) {
        for (uint64_t i = indexDim_; i < varDimNum; i++) {
            scatterLength_ *= varRefShape.GetDim(i);
        }
    }
    uint64_t indexRow = 1;
    for (uint64_t i = 0; i < indicesShape.GetDimNum() - 1; i++) {
        indexRow *= indicesShape.GetDim(i);
    }

    if (needLargeIndexKernel_) {
        isSort_ = false;
        isLinearIndex_ = false;
    } else {
        isSort_ = false;
        isLinearIndex_ = IsLinearIndex(totalLength);
    }
    coreNum_ = std::min(compileInfo->totalCoreNum,
                    std::min(static_cast<uint64_t>(totalLength), static_cast<uint64_t>(indexRow)));
    coreNum_ = coreNum_ == 0 ? 1 : coreNum_;
    // 启动成本 F(T) ~= 1.6 + 0.16*T us（EXP-2/EXP-3 实测确认：F 与 pre-pass、
    // SyncAll 无关，就是唤醒核本身；dec48 40->8 核实测 8.22 -> 3.1 us）。
    // 每行边际工作 ~36 ns，最小化 F(T) + R*0.036/T 得 T* = sqrt(0.225*R)：
    // 下面循环就是 ceil(sqrt(9R/40)) 的整数形式（EXP-7：去掉人为的 8 核下限，
    // 让 R=8/20/48 的 decode 例落到 sqrt 最优点 2/3/4），上限由 coreNum_
    // 自身的 40 封顶。R=2068 -> 22 核，R>=8271 -> 40。
    uint64_t neededCores = 2;
    while (neededCores * neededCores * 40 < indexRow * 9 && neededCores < 40) {
        ++neededCores;
    }
    // EXP-14/22: width-1 (C 系) 单独设核。EXP-15 定 8 是在单批（无重叠）
    // 时代；EXP-17-21 打开重叠后 mte3 饱和，重新升核 8->16 paired 3/3
    // -6.0%（wake-up 1.7us < mte3 减半收益）。
    if (scatterLength_ == 1) {
        neededCores = 16;
    }
    coreNum_ = std::min(coreNum_, neededCores);
    ubSize_ = compileInfo->ubSizePlatForm;
    GetDtypeSize();
    Tiling4LinearIndex(indexRow, indexDim_);
    uint64_t maxPhysicalOffset = 0;
    for (uint64_t i = 0; i < indexDim_; ++i) {
        maxPhysicalOffset += (varRefShape.GetDim(i) - 1) * indicesMask_[i];
    }
    uint64_t totalPhysicalRange = maxPhysicalOffset + scatterLength_;
    outputPhysicalRange_ = totalPhysicalRange;
    if (!needLargeIndexKernel_) {
        isSort_ = IsSort(totalPhysicalRange, indexRow);
    }
    isRowSplit_ = CanRowSplit(indexRow);
    if (isRowSplit_) {
        isSort_ = false;
    }
    SetTilingKeyMode();
    tilingContext_->SetScheduleMode(1);
    // RowSplit 把 **行** 切给各核，其余路径切的是输出地址区间。两者复用
    // frontNum/frontRow/tailRow 这几个字段，只有被选中的那条路径会读它们。
    Tiling4Scatter(isRowSplit_ ? indexRow : totalPhysicalRange, indexRow);
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = CalcWorkSpaceSize(indexRow);
    OP_LOGD(tilingContext_, "Tiling inited");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ScatterNdUpdateV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("ScatterNdUpdateV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling for ScatterNdUpdateV2 start.");
    ScatterNdUpdateV2Tiling tilingOp(context);
    if (tilingOp.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling init fail");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling for ScatterNdUpdateV2 end.");
    return tilingOp.SetKernelTiling();
}

ge::graphStatus TilingPrepare4ScatterNdUpdateV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For ScatterNdUpdateV2 start.");
    auto compileInfo = context->GetCompiledInfo<ScatterNdUpdateV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        OP_LOGE(context, "coreNum %lu", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ubSizePlatForm is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For ScatterNdUpdateV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterNdUpdateV2).Tiling(Tiling4ScatterNdUpdateV2).TilingParse<ScatterNdUpdateV2CompileInfo>(TilingPrepare4ScatterNdUpdateV2);
}
