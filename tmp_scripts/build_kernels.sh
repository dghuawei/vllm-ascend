#!/bin/bash
# Rebuild vllm-ascend custom kernels in the container so the merged
# add_lora_swiglu_quant / combined_lora_idx ops land in _C_ascend.
set -euo pipefail

cd /vllm-workspace/vllm-ascend

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=${SOC_VERSION:-ascend910b4}
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=${MAX_JOBS:-32}

echo "=== SOC_VERSION=$SOC_VERSION MAX_JOBS=$MAX_JOBS ==="
python3 setup.py build_ext --inplace 2>&1 | tail -40
echo "=== build rc=$? ==="
