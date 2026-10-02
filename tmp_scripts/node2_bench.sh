#!/bin/bash
# Local driver: ship the standalone A/B to researchagent-node2, into the
# lab-th_va25_ds_lora_infer container, build and run it. Stages only into
# /tmp -- touches no vllm-ascend checkout on the remote side.
set -euo pipefail

REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/standalone
HOST=researchagent-node2
CTR=${CTR:-lab-th_va25_ds_lora_infer}
STAGE=/tmp/swiglu_ab

rm -rf "$SRC/.stage_node2" && mkdir -p "$SRC/.stage_node2"
cp "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp"                       "$SRC/.stage_node2/new.cpp"
cp "$REPO/tmp_scripts/variants/add_lora_swiglu_quant_r5_baseline.cpp"   "$SRC/.stage_node2/base.cpp"
cp "$REPO/tmp_scripts/variants/add_lora_swiglu_quant_wrm2pass.cpp"      "$SRC/.stage_node2/wrm2pass.cpp"
cp "$REPO/csrc/kernels/types.h"                                          "$SRC/.stage_node2/"
cp "$SRC/CMakeLists.txt" "$SRC/main.cpp" "$SRC/node2_remote_all.sh" "$SRC/pipe_summary.py" \
   "$SRC/.stage_node2/"

tar -czf "$SRC/.stage_node2.tgz" -C "$SRC/.stage_node2" .
scp -q "$SRC/.stage_node2.tgz" "$HOST:/tmp/swiglu_ab.tgz"

ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/swiglu_ab.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
sudo docker exec $CTR bash $STAGE/node2_remote_all.sh
"
