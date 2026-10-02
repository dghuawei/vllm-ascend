#!/bin/bash
# Verify the tail fix: pretail (per-row ReduceMax, handles any W) vs the new
# two-pass+tail kernel, over widths that stress every branch.
set -uo pipefail
WORK=/tmp/swiglu_matrix
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b3}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"
source "$WORK/shapes.env"
source "$WORK/run_matrix_common.sh"

cat > "$WORK/shapes_verify.txt" <<SHP
4096 32 1 50
4096 48 1 50
4096 64 1 50
4096 112 1 50
4096 256 1 50
4096 528 1 50
4096 768 1 50
4096 1024 1 50
4096 1040 1 50
4096 1392 1 50
4096 1408 1 50
4096 2048 1 50
4096 2064 1 50
2048 4096 1 30
2048 4112 1 30
1024 8192 1 30
128 256 0 200
128 1392 0 200
SHP

for v in pretail new; do
    echo "================ $v ================"
    build_variant "$v" "$v.cpp" || continue
    V=$WORK/v_$v
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    ( cd "$V/build" && ./bench --matrix "$WORK/shapes_verify.txt" 2>&1 ) \
        | grep -E '^T=|^ref=fp32|non-finite|rror'
done
echo "================ DONE verify ================"
