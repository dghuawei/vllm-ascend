#!/bin/bash
# Script 2 of 3 -- WHOLEREDUCE: row max via two-pass WholeReduceMax.
# NOTE: widens maxBuffer_ 128 -> 256 floats (+512 B UB); W=768 needs 160.
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

echo "================ BUILD wrm2pass ================"
build_variant wrm2pass wrm2pass.cpp || exit 1
echo
echo "================ WALL wrm2pass ================"
run_wall wrm2pass
echo
echo "================ DEVICE wrm2pass ================"
run_device wrm2pass
echo "================ DONE wrm2pass ================"
