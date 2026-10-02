#!/bin/bash
# Round 5: run the standalone harness against an ARBITRARY kernel source file,
# so variants can be A/B'd without mutating csrc/kernels/.
#
# Usage: run_kernel_variant.sh <kernel.cpp> <label> [repeats]
#
# Same staging path as standalone/push_and_run.sh: ship to bz-ascend, into the
# vllm_misha container, build standalone (no torch/vllm), run the bench.
set -euo pipefail

KERNEL="${1:?usage: run_kernel_variant.sh <kernel.cpp> <label> [repeats]}"
LABEL="${2:?usage: run_kernel_variant.sh <kernel.cpp> <label> [repeats]}"
REPEATS="${3:-1}"

REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/standalone
HOST=bz-ascend
CTR=vllm_misha
STAGE=/tmp/swiglu_variant_$LABEL

rm -rf "$SRC/.stage_$LABEL" && mkdir -p "$SRC/.stage_$LABEL"
# the harness always compiles a file named add_lora_swiglu_quant.cpp
cp "$KERNEL"                  "$SRC/.stage_$LABEL/add_lora_swiglu_quant.cpp"
cp "$REPO/csrc/kernels/types.h" "$SRC/.stage_$LABEL/"
cp "$SRC/CMakeLists.txt" "$SRC/main.cpp" "$SRC/.stage_$LABEL/"
# remote_build_run.sh hardcodes WORK=/tmp/swiglu_standalone; repoint it at this
# variant's stage dir so the two arms cannot collide on the remote.
sed "s|^WORK=.*|WORK=$STAGE|" "$SRC/remote_build_run.sh" > "$SRC/.stage_$LABEL/remote_build_run.sh"

tar -czf "$SRC/.stage_$LABEL.tgz" -C "$SRC/.stage_$LABEL" .
scp -q "$SRC/.stage_$LABEL.tgz" "$HOST:/tmp/swiglu_variant_$LABEL.tgz"

ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/swiglu_variant_$LABEL.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
for i in \$(seq 1 $REPEATS); do
  echo \"########## $LABEL run \$i ##########\"
  sudo docker exec $CTR bash $STAGE/remote_build_run.sh
done
"
