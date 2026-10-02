#!/bin/bash
# Extra widths not in the main matrix: W=4096 is a shipping width, and a
# non-64-divisible W checks whether wrm2pass has a correctness cliff the
# baseline does not.
set -uo pipefail
WORK=/tmp/swiglu_matrix
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b3}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"
source "$WORK/shapes.env"
source "$WORK/run_matrix_common.sh"

cat > "$WORK/shapes_extra.txt" <<SHP
8 4096 0 200
128 4096 0 200
2048 4096 1 50
8192 4096 1 50
32768 4096 1 20
4096 2112 1 50
SHP

for v in baseline wrm2pass; do
    src=$v.cpp
    echo "================ $v ================"
    build_variant "$v" "$src" || continue
    V=$WORK/v_$v
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    ( cd "$V/build" && ./bench --matrix "$WORK/shapes_extra.txt" 2>&1 ) \
        | grep -E '^T=|^ref=fp32|non-finite|rror'
done
echo "================ DONE extra ================"
