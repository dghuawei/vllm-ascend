#!/bin/bash
# Local driver: stage the two-delta A/B (HEAD kernel vs work tree) onto node1,
# build and run it inside the container. Nothing is built on this machine.
set -euo pipefail
REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/standalone
HOST=${HOST:-researchagent-node1}
CTR=${CTR:-va25_ds}
STAGE=/tmp/swiglu_twodelta
ROUNDS=${ROUNDS:-3}
STG=$SRC/.stage_twodelta

rm -rf "$STG" && mkdir -p "$STG"
# baseline = committed kernel + committed harness (one [T,2W] delta)
git -C "$REPO" show HEAD:csrc/kernels/add_lora_swiglu_quant.cpp > "$STG/base.cpp"
# tmp_scripts is untracked and the archived .stage_matrix/main.cpp predates the
# swiglu_limit argument, so derive the baseline harness by reverting exactly the
# two-delta edits -- everything else stays identical between the variants
python3 "$SRC/make_main_base.py" "$SRC/main.cpp" "$STG/main_base.cpp"
# candidate = work tree (two [T,W] deltas)
cp "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp" "$STG/new.cpp"
cp "$SRC/main.cpp"                                "$STG/main_new.cpp"
python3 "$SRC/make_rowcast_variant.py" "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp" \
        "$STG/new_rowcast.cpp"
cp "$REPO/csrc/kernels/types.h" "$SRC/CMakeLists.txt" "$SRC/node1_run_twodelta.sh" "$STG/"

tar -czf "$STG.tgz" -C "$STG" .
scp -q "$STG.tgz" "$HOST:/tmp/swiglu_twodelta.tgz"
ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/swiglu_twodelta.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
sudo docker exec -e ROUNDS='$ROUNDS' -e EXTRA_VARIANTS='${EXTRA_VARIANTS:-}' $CTR bash $STAGE/node1_run_twodelta.sh
"
