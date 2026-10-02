#!/bin/bash
# Script 1 of 3 -- BASELINE: the current kernel (scalar-free quantize path).
# Two passes: wall clock (what the launcher costs) and msprof device time
# (what the kernel costs, which is what vLLM sees in compiled mode).
set -uo pipefail

WORK=/tmp/swiglu_matrix
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b3}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

source "$WORK/shapes.env"
source "$WORK/run_matrix_common.sh"

echo "================ BUILD baseline ================"
build_variant baseline baseline.cpp || exit 1
echo
echo "================ WALL baseline ================"
run_wall baseline
echo
echo "================ DEVICE baseline ================"
run_device baseline
echo "================ DONE baseline ================"
