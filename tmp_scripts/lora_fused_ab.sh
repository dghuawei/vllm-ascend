#!/bin/bash
# Local driver: ship the add_lora_fused A/B to node1, build and run in /tmp.
set -euo pipefail
REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/lora_fused
HOST=${HOST:-researchagent-node1}
CTR=${CTR:-va25_ds}
STAGE=/tmp/lora_fused
BUILD_ONLY=${BUILD_ONLY:-0}
SHAPES=${SHAPES:-"48 4096 16 1 200"}
PROF=${PROF:-0}
VARIANTS=${VARIANTS:-"base new"}
LOAD=${LOAD:-0}
MEMLOAD=${MEMLOAD:-}

rm -rf "$SRC/.stage" && mkdir -p "$SRC/.stage"
cp "$SRC"/base.cpp "$SRC"/new.cpp "$SRC"/abl_*.cpp "$SRC"/opt*.cpp "$SRC/main.cpp" "$SRC/CMakeLists.txt" "$SRC/remote_ab.sh" "$SRC/.stage/"
cp "$REPO/csrc/kernels/types.h" "$SRC/.stage/"
tar -czf "$SRC/.stage.tgz" -C "$SRC/.stage" .
scp -q "$SRC/.stage.tgz" "$HOST:/tmp/lora_fused.tgz"
ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/lora_fused.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
sudo docker exec -e BUILD_ONLY='"$BUILD_ONLY"' -e SHAPES='"$SHAPES"' -e PROF='"$PROF"' -e VARIANTS='"$VARIANTS"' -e LOAD='"$LOAD"' -e MEMLOAD='"$MEMLOAD"' $CTR bash $STAGE/remote_ab.sh
"
