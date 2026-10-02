#!/bin/bash
# Guard: refuse to benchmark unless the NPUs are actually free.
# RUN THIS BEFORE EVERY BENCHMARK. A busy device does not just give bad
# numbers, it can OOM someone else's job.
#
# usage: check_npu_idle.sh <ssh-host> [samples] [interval_s]
# exit 0 = idle, 1 = busy
set -uo pipefail

HOST=${1:-researchagent-node1}
N=${2:-6}
IVL=${3:-5}

echo "=== processes holding NPUs on $HOST ==="
PROCS=$(ssh "$HOST" "npu-smi info 2>/dev/null | awk '/Process id/,0' | grep -E '^\| [0-9]+ +[0-9]'" 2>/dev/null)
if [ -n "$PROCS" ]; then echo "$PROCS"; else echo "(none)"; fi

echo
echo "=== AICore% per chip | total HBM, ${N} samples ${IVL}s apart ==="
ssh "$HOST" "for i in \$(seq $N); do
    npu-smi info 2>/dev/null | grep '0000:' \
      | awk '{printf \"%3s\", \$6} END {}'
    npu-smi info 2>/dev/null | grep '0000:' \
      | awk '{s+=\$10} END {printf \"   | hbm_total=%d MB\n\", s}'
    sleep $IVL
done" 2>/dev/null

echo
if [ -n "$PROCS" ]; then
    echo "!! BUSY: processes are holding the NPUs. DO NOT BENCHMARK."
    echo "   Identify them:  ssh $HOST 'ps -o pid,etime,comm -p <pid>'"
    exit 1
fi
echo "OK: no processes on the NPUs."
echo "   Still eyeball the HBM column: a steadily climbing total means a"
echo "   job is loading and has not yet registered in the process table."
exit 0
