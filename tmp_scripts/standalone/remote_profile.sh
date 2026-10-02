#!/bin/bash
# Runs INSIDE the vllm_misha container. Profiles the standalone bench with
# msprof PipeUtilization to answer: which pipe is the bottleneck?
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="$WORK/build/lib:$WORK/build:${LD_LIBRARY_PATH:-}"

T=${1:-8192}
W=${2:-2048}
DELTA=${3:-1}
OUT=/tmp/prof_${T}_${W}_${DELTA}

rm -rf "$OUT" && mkdir -p "$OUT"
cd "$WORK/build"

msprof --output="$OUT" \
       --application="$WORK/build/bench $T $W $DELTA 20" \
       --aic-metrics=PipeUtilization \
       --ai-core=on --task-time=on --ascendcl=on \
       2>&1 | tail -8

echo "=== op_summary ==="
SUM=$(find "$OUT" -name 'op_summary*.csv' | head -1)
if [ -z "$SUM" ]; then
    echo "no op_summary found; tree:"
    find "$OUT" -name '*.csv' | head -20
    exit 0
fi
echo "file: $SUM"
python3 - "$SUM" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows = [r for r in rows if 'swiglu' in (r.get('Op Name') or r.get('op_name') or '').lower()]
if not rows:
    rows = list(csv.DictReader(open(sys.argv[1])))
    print("(no swiglu rows; showing all op names)")
    print(sorted({(r.get('Op Name') or r.get('op_name') or '?') for r in rows}))
    sys.exit()
keys = list(rows[0].keys())
want = [k for k in keys if any(s in k.lower() for s in
        ('op name','task duration','aiv_','aic_','ratio','vec_','mte','scalar','cube','bound'))]
print("n rows:", len(rows))
for k in want:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[k]))
        except (TypeError, ValueError):
            vals.append(None)
    nums = [v for v in vals if v is not None]
    if nums:
        print(f"  {k:<40s} mean={sum(nums)/len(nums):.4f}  max={max(nums):.4f}")
PY
