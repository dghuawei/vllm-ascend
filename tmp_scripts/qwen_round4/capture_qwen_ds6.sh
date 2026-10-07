#!/bin/bash
# qwen/ round-4 capture driver (identical on BOTH nodes).
# Serve must already be running via run_qwen_ds6_lora_*.sh.
# Recipe copied from scripts/profile_ds.sh: warmup 8x(8192in/16out) seed 42,
# then profile window over 8x(8192in/128out) seed 43, parallel 8.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXPERIMENT=${EXPERIMENT:-qwen_ds6_$(date +%Y%m%d_%H%M)}
PORT=${PORT:-1995}
MODEL=${MODEL:-/data/models/DeepSeek-V4-Flash-0731-w8a8-6L}
PROFILE_MODEL=${PROFILE_MODEL:-lora-adapter1}
URL=http://localhost:${PORT}/v1/chat/completions
OUT=${SCRIPT_DIR}/logs/${EXPERIMENT}
mkdir -p "${OUT}"

READY=0
for i in $(seq 1 120); do
    if curl -sf -m 3 -o /dev/null http://localhost:${PORT}/health; then READY=1; break; fi
    sleep 10
done
[ "$READY" = "1" ] || { echo "service not ready, aborting"; exit 1; }

echo "== warmup =="
python "${SCRIPT_DIR}/run_evalscope.py" \
    --number 8 --parallel 8 \
    --model "$PROFILE_MODEL" --api openai --url "$URL" \
    --dataset random --seed 42 \
    --tokenizer-path "$MODEL" \
    --min-prompt-length 8192 --max-prompt-length 8192 \
    --min-tokens 16 --max-tokens 16 \
    --prefix-length 0 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir "${OUT}" --rate -1 || { echo "warmup failed"; exit 1; }

echo "== profile window =="
curl -sf -X POST http://localhost:${PORT}/start_profile || { echo "start_profile failed"; exit 1; }
python "${SCRIPT_DIR}/run_evalscope.py" \
    --number 8 --parallel 8 \
    --model "$PROFILE_MODEL" --api openai --url "$URL" \
    --dataset random --seed 43 \
    --tokenizer-path "$MODEL" \
    --min-prompt-length 8192 --max-prompt-length 8192 \
    --min-tokens 128 --max-tokens 128 \
    --prefix-length 0 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir "${OUT}" --rate -1
curl -sf -X POST http://localhost:${PORT}/stop_profile || echo "WARNING: stop_profile failed"

echo "== kill serve =="
PID=$(ps -ef | grep "vllm serve ${MODEL}" | grep -v grep | awk '{print $2}')
kill $PID 2>/dev/null
sleep 30
for p in $(ps -eo pid,args | grep -E "VLLM::|EngineCore|vllm serve" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 3
curl -sf -m 2 -o /dev/null http://localhost:${PORT}/health && echo "ERROR: still serving" || echo "service stopped cleanly"
echo "OUT=${OUT}"
