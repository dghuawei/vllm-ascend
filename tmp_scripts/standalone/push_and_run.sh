#!/bin/bash
# Local driver: ship the standalone kernel build to bz-ascend, into the
# vllm_misha container, build it and run the bench. Nothing here touches the
# repo checked out in the container.
set -euo pipefail

REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/standalone
HOST=bz-ascend
CTR=vllm_misha
STAGE=/tmp/swiglu_standalone

rm -rf "$SRC/.stage" && mkdir -p "$SRC/.stage"
cp "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp" "$SRC/.stage/"
cp "$REPO/csrc/kernels/types.h" "$SRC/.stage/"
cp "$SRC/CMakeLists.txt" "$SRC/main.cpp" "$SRC/remote_build_run.sh" "$SRC/.stage/"

tar -czf "$SRC/.stage.tgz" -C "$SRC/.stage" .
scp -q "$SRC/.stage.tgz" "$HOST:/tmp/swiglu_standalone.tgz"

ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/swiglu_standalone.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
sudo docker exec $CTR bash $STAGE/remote_build_run.sh
"
