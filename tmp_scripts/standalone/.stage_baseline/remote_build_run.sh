#!/bin/bash
# Runs INSIDE the vllm_misha container. Builds add_lora_swiglu_quant standalone
# (no torch, no vllm) and runs the bench across a few shapes.
set -euo pipefail

WORK=/tmp/swiglu_variant_baseline
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=ascend910b4

cd "$WORK"
find . -type f -exec touch {} +
rm -rf build && mkdir -p build && cd build

echo "=== cmake ==="
cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" 2>&1 | tail -15
echo "=== make ==="
make -j16 2>&1 | tail -30

echo "=== run ==="
export LD_LIBRARY_PATH="$WORK/build/lib:$WORK/build:${LD_LIBRARY_PATH:-}"
for shape in "8192 1408" "8192 2048" "8192 4096" "32768 2048"; do
    ./bench $shape 1 50
done
echo "=== no-delta variant ==="
./bench 8192 2048 0 50
