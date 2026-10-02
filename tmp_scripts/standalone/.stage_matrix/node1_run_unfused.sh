#!/bin/bash
# Script 3 of 3 -- unfused eager chain. Wall pass, then a profiled pass whose
# device time comes from msprof Task Duration rather than the Python loop.
set -uo pipefail
WORK=/tmp/swiglu_matrix
source /usr/local/Ascend/ascend-toolkit/set_env.sh
MARKER=${MARKER:-cumsum}

echo "================ WALL unfused ================"
python3 "$WORK/node1_run_unfused.py" wall

echo
echo "================ DEVICE unfused ================"
OUT=/tmp/prof_matrix_unfused
rm -rf "$OUT" && mkdir -p "$OUT"
msprof --output="$OUT" \
    --application="python3 $WORK/node1_run_unfused.py prof" \
    --ai-core=on --task-time=on --ascendcl=on >"$OUT/msprof.log" 2>&1

APP="$OUT/app.log"
grep -E '^MANIFEST ' "$OUT/msprof.log" > "$APP" 2>/dev/null || true
if ! grep -q MANIFEST "$APP"; then
    echo "   !! no MANIFEST in msprof stdout; see $OUT/msprof.log"
    tail -20 "$OUT/msprof.log"
    exit 1
fi
SUM=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
if [ -z "$SUM" ]; then
    echo "   !! no op_summary under $OUT"; tail -20 "$OUT/msprof.log"; exit 1
fi
cp "$SUM" "$OUT/op_summary_used.csv"
echo "   distinct op names seen:"
python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$SUM')))
k=[x for x in rows[0] if x.strip().lower() in ('op name','op_name')][0]
from collections import Counter
for n,c in Counter((r[k] or '?').strip() for r in rows).most_common(12):
    print(f'     {c:8d}  {n}')
"
python3 "$WORK/device_time.py" --summary "$SUM" --log "$APP" \
        --variant unfused_dev --marker "$MARKER" --unfused
echo "================ DONE unfused ================"
