#!/bin/bash
# Repeat the workload N times against ONE server boot, so run-to-run spread is
# measured without boot-to-boot variance mixed in. Usage: e2e_repeat.sh <label> [reps]
set -u

LABEL="${1:?usage: e2e_repeat.sh <label> [reps]}"
REPS="${2:-5}"
DIR=/home/misha/e2e
BOOT_TIMEOUT_S=${BOOT_TIMEOUT_S:-1500}
PORT=1995

pkill -f "vllm serve" 2>/dev/null
sleep 10

echo "[$LABEL] cap: $(grep '_W13_FUSED_MAX_HALF_WIDTH = ' \
  /vllm-workspace/vllm-ascend/vllm_ascend/lora/quant_moe.py)"

nohup "$DIR/serve_dsv4_lora.sh" > "$DIR/serve_${LABEL}.log" 2>&1 &
SERVE_PID=$!

READY=0
for i in $(seq 1 $((BOOT_TIMEOUT_S / 10))); do
    if curl -sf -m 3 -o /dev/null "http://127.0.0.1:${PORT}/health"; then READY=1; break; fi
    if ! kill -0 $SERVE_PID 2>/dev/null; then echo "[$LABEL] serve died early"; break; fi
    sleep 10
done
[ "$READY" != "1" ] && { echo "[$LABEL] NOT READY"; tail -30 "$DIR/serve_${LABEL}.log"; exit 1; }
echo "[$LABEL] server ready"

for r in $(seq 1 "$REPS"); do
    echo "===== $LABEL rep $r ====="
    python3 "$DIR/bench_e2e.py" --label "${LABEL}_r${r}" \
        --out "$DIR/result_${LABEL}_r${r}.json" ${BENCH_ARGS:-} 2>&1 | tail -20
done

pkill -f "vllm serve" 2>/dev/null
sleep 15
echo "[$LABEL] all reps done"
