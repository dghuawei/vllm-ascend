#!/bin/bash
# Run the pristine baseline for all six agent slots AT ONCE, which is also the
# concurrency test: 6 builds and 6 msprof sessions in one container, each
# pinned to its own chip. If the numbers here match the ones measured one at a
# time, the agents can work in parallel without poisoning each other's
# measurements.
set -u
ARENA=/Users/arabel1a/work/huawei/qlora/kernel_arena
OUT=${1:-/tmp/arena_baseline_all}
mkdir -p "$OUT"

for a in a1 a2 b1 b2 c1 c2; do
    ( bash "$ARENA/run.sh" "$a" --baseline > "$OUT/$a.txt" 2>&1; echo "$a rc=$?" ) &
done
wait
echo "=== done ==="
for a in a1 a2 b1 b2 c1 c2; do
    echo "--- $a ---"
    grep -E "^(DEV|# failures|!!)" "$OUT/$a.txt" | head -20
done
