#!/bin/bash
# qwen/ round-6 attempt 4: bz 910B4 boards are 32 GiB; gmu 0.93 (node1's value,
# fine on its 64 GiB boards) OOMed 3x with a deterministic extra 770 MiB alloc
# in the first real prefill. bz-native boot scripts for this image/model run at
# gmu 0.85-0.90 -> set native 0.85, drop the kv-cache-memory pin.
set -u
P=/home/russia_mmo/qwen/run_qwen_ds6_lora_bz.sh
cp "$P" "$P.attempt3.bak"
python3 - "$P" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
assert "--gpu-memory-utilization 0.93" in s and "--kv-cache-memory" in s, "flags not as expected"
s = s.replace("  --gpu-memory-utilization 0.93 \\\n  --kv-cache-memory 17080836097 \\\n",
              "  --gpu-memory-utilization 0.85 \\\n")
marker = "# the 8x8192-token capture, so no request behavior changes. gmu 0.93 kept.\n"
assert marker in s, "comment anchor not found"
add = marker + """#
# 2026-10-07 attempts 2 and 3 still OOMed with the SAME deterministic
# "Tried to allocate 770.00 MiB" during the first real prefill: the piecewise
# prefill eager path needs that extra 770 MiB of NON-reserved device memory,
# which at gmu 0.93 (27.42 of 29.49 GiB usable) does not exist on these
# 32 GiB boards. bz-native working scripts for this exact image/model
# (/home/russia_mmo/tmp_scripts/2026072*_dsv4*boot*.sh) run at gmu 0.85-0.90
# with max-num-seqs 8, so 0.93 was never attainable here. Attempt 4 = native
# 0.85, no kv pin. Deviation from node1 flags (node1: plain 0.93 on 64 GiB
# boards): gmu changes pool sizing only, not kernel shapes/call counts, so the
# per-call b3-vs-b4 comparison stays valid.
"""
s = s.replace(marker, add)
open(p, "w").write(s)
print("patched")
PYEOF
grep -nE "gpu-memory|kv-cache" "$P"
