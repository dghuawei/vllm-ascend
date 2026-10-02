#!/bin/bash
# Local driver: ship the standalone A/B to researchagent-node1, into the
# va25_ds container, build and run it. Stages only into /tmp -- touches no
# vllm-ascend checkout on the remote side.
set -euo pipefail

REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/standalone
HOST=researchagent-node1
CTR=va25_ds
STAGE=/tmp/swiglu_ab

rm -rf "$SRC/.stage_node1" && mkdir -p "$SRC/.stage_node1"
cp "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp"                   "$SRC/.stage_node1/new.cpp"
cp "$REPO/tmp_scripts/variants/add_lora_swiglu_quant_r5_baseline.cpp" "$SRC/.stage_node1/base.cpp"
cp "$REPO/csrc/kernels/types.h"                                      "$SRC/.stage_node1/"
cp "$SRC/CMakeLists.txt" "$SRC/main.cpp" "$SRC/node1_remote_ab.sh" "$SRC/pipe_summary.py" \
   "$SRC/.stage_node1/"

tar -czf "$SRC/.stage_node1.tgz" -C "$SRC/.stage_node1" .
scp -q "$SRC/.stage_node1.tgz" "$HOST:/tmp/swiglu_ab.tgz"

ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/swiglu_ab.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
sudo docker exec $CTR bash $STAGE/node1_remote_ab.sh
"
