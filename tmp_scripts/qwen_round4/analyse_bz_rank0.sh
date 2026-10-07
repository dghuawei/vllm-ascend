#!/bin/bash
# qwen/ round-6: analyse ONLY rank0 of the bz capture (per user: per-device
# traces are ~identical, device 0 is enough). Runs the torch_npu profiler
# analyser on the rank0 dir; leaves ranks 1-7 raw on bz, unanalysed.
set -u
PROF=${PROF:-/home/russia_mmo/qwen/logs/prof_bz}
R0=$(ls -d "${PROF}"/*_rank0_* 2>/dev/null | head -1)
[ -n "${R0}" ] || { echo "no rank0 dir in ${PROF}"; ls "${PROF}" 2>/dev/null | head; exit 1; }
echo "analysing: ${R0}"
python -c "import torch_npu, os; torch_npu.profiler.profiler.analyse('${R0}')"
OUT="${R0}/ASCEND_PROFILER_OUTPUT"
ls -la "${OUT}" 2>/dev/null | head -12
TARBALL=/home/russia_mmo/qwen/logs/bz_rank0_analysis.tar.gz
tar -C "$(dirname "${OUT}")" -czf "${TARBALL}" "$(basename "${OUT}")"
ls -la "${TARBALL}"
