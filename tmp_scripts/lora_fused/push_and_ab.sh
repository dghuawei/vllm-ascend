#!/bin/bash
# Local driver: push the harness into the va25_ds container on
# researchagent-node1 and run the A/B. Device idleness must be checked first.
#
#   ./push_and_ab.sh "base mte2_asis mte2_fix"        wall + correctness
#   PROF=1 ./push_and_ab.sh "base mte2_fix"           + msprof device time
set -uo pipefail
HOST=${HOST:-researchagent-node1}
CTR=${CTR:-va25_ds}
VARIANTS=${1:-"base mte2_asis mte2_fix"}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== pushing to $HOST:$CTR =="
tar czf - -C "$HERE" --exclude='v_*' --exclude='*.log' . \
  | ssh -o BatchMode=yes "$HOST" "sudo docker exec -i $CTR bash -c 'mkdir -p /tmp/lora_fused && tar xzf - -C /tmp/lora_fused && find /tmp/lora_fused -type f -exec touch {} +'"

echo "== running (VARIANTS='$VARIANTS' PROF=${PROF:-0}) =="
ssh -o BatchMode=yes "$HOST" \
  "sudo docker exec -e VARIANTS='$VARIANTS' -e PROF=${PROF:-0} -i $CTR bash /tmp/lora_fused/remote_ab.sh"
