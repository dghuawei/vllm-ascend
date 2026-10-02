#!/bin/bash
# One arm of the fused-w13-epilogue A/B. Usage: e2e_run.sh <label>
# Boots the server, waits for health, runs the fixed workload, tears down.
# Identical procedure for both arms so only the kernel toggle differs.
set -u

LABEL="${1:?usage: e2e_run.sh <label>}"
DIR=/home/misha/e2e
BOOT_TIMEOUT_S=${BOOT_TIMEOUT_S:-1500}
PORT=1995

pkill -f "vllm serve" 2>/dev/null
sleep 10

echo "[$LABEL] fused-epilogue cap: $(grep -n '_W13_FUSED_MAX_HALF_WIDTH = ' \
  /vllm-workspace/vllm-ascend/vllm_ascend/lora/quant_moe.py)"

nohup "$DIR/serve_dsv4_lora.sh" > "$DIR/serve_${LABEL}.log" 2>&1 &
SERVE_PID=$!

READY=0
for i in $(seq 1 $((BOOT_TIMEOUT_S / 10))); do
    if curl -sf -m 3 -o /dev/null "http://127.0.0.1:${PORT}/health"; then READY=1; break; fi
    if ! kill -0 $SERVE_PID 2>/dev/null; then echo "[$LABEL] serve died early"; break; fi
    sleep 10
done

if [ "$READY" != "1" ]; then
    echo "[$LABEL] NOT READY after ${BOOT_TIMEOUT_S}s; tail of log:"
    tail -40 "$DIR/serve_${LABEL}.log"
    pkill -f "vllm serve" 2>/dev/null
    exit 1
fi
echo "[$LABEL] server ready"

# unquoted on purpose: empty BENCH_ARGS must expand to no argument at all
python3 "$DIR/bench_e2e.py" --label "$LABEL" --out "$DIR/result_${LABEL}.json" \
    ${BENCH_ARGS:-} 2>&1 | tail -30
RC=$?

pkill -f "vllm serve" 2>/dev/null
sleep 15
echo "[$LABEL] done rc=$RC"
