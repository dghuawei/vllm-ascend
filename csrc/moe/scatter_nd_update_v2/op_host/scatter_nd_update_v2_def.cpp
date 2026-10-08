/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file scatter_nd_update_v2.cpp
 * \brief ScatterNdUpdateV2 ophost
 */
#include "register/op_def_registry.h"

namespace ops {
class ScatterNdUpdateV2 : public OpDef {
 public:
  explicit ScatterNdUpdateV2(const char* name) : OpDef(name) {
    this->Input("var")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("indices")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("updates")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("var")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("strides").AttrType(REQUIRED).ListInt();
    // use_locking selects how the work is split across cores:
    //   false (default): rows of `updates` are split evenly across cores, so the
    //     kernel is fast whatever the index distribution, and WHICH of several
    //     updates sharing one index lands is unspecified.
    //   true: the output address space is split across cores, so one core owns
    //     every write to a given slot and visits them in index order, making
    //     duplicate indices deterministic (last write wins) -- at the cost of a
    //     replicated index scan and of collapsing onto one core when the indices
    //     cluster, as a KV-cache slot_mapping does.
    // Pass true only if duplicate indices are possible AND the result must be
    // deterministic. See CanRowSplit in scatter_nd_update_v2_tiling.cpp.
    this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true);
    this->AICore().AddConfig("ascend910b", aicore_config);
    this->AICore().AddConfig("ascend910_93", aicore_config);
  }
};

OP_ADD(ScatterNdUpdateV2);
}  // namespace ops