#!/bin/bash
# Runs INSIDE the container. Uses ALREADY-BUILT binaries in /tmp/lora_fused/v_*/build.
#   VARIANTS="base mte2_fix" [PROF=1] bash remote_run_only.sh
set -uo pipefail
WORK=/tmp/lora_fused
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

SHAPES=("6 4096 16 1 500 96" "36 4096 16 2 500 96" "48 4096 16 2 500 96" \
        "48 4096 16 3 500 96" "288 4096 16 2 300 96" "288 4096 16 3 300 96" \
        "63 4096 16 2 500 7" "288 2048 32 2 300 96" "128 4096 64 1 300 96")

echo "================ WALL + CORRECTNESS ================"
for v in ${VARIANTS:-base mte2_fix}; do
    V=$WORK/v_$v
    [ -x "$V/build/bench" ] || { echo "!! no binary $v"; continue; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $v ---"
    for a in "${SHAPES[@]}"; do ( cd "$V/build" && ./bench $a ); done
done

[ "${PROF:-0}" = "1" ] || exit 0

echo "================ DEVICE TIME PER KERNEL (msprof) ================"
for v in ${VARIANTS:-base mte2_fix}; do
    V=$WORK/v_$v
    [ -x "$V/build/bench" ] || continue
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $v ---"
    for sh in "6 4096 16 1 30 96" "48 4096 16 2 30 96" "288 4096 16 2 30 96" "128 4096 64 1 30 96"; do
        OUT=/tmp/prof_ro_${v}_$(echo $sh | tr ' ' '_'); rm -rf "$OUT"; mkdir -p "$OUT"
        ( cd "$V/build" && LF_NOCHECK=1 msprof --output="$OUT" --application="$V/build/bench $sh" \
            --aic-metrics=PipeUtilization --ai-core=on --task-time=on ) >"$OUT/msprof.log" 2>&1
        CSV=$(find "$OUT" -name 'kernel_details.csv' 2>/dev/null | head -1)
        [ -z "$CSV" ] && CSV=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
        [ -z "$CSV" ] && { echo "  $sh : NO CSV (see $OUT/msprof.log)"; continue; }
        python3 - "$CSV" "$sh" <<'PY'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
sh=sys.argv[2].split()
nk=[k for k in rows[0] if k.strip().lower() in ('name','op name','op_name')][0]
dk=[k for k in rows[0] if 'duration' in k.lower()][0]
out=[f"  B={sh[0]:>4} H1={sh[1]} R={sh[2]:>2} S={sh[3]}"]
for pref in ('add_lora_z1','add_lora_z2'):
    sel=[r for r in rows if r[nk].startswith(pref)]
    if not sel: continue
    d=[float(r[dk]) for r in sel]
    seg=f"{pref[-2:]}={st.median(d):8.2f}us n={len(d):<4}"
    for c in ('aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'):
        if c in sel[0]:
            v=[float(r[c]) for r in sel if r[c] not in ('','N/A')]
            if v: seg+=f" {c.split('_')[1][:3]}={st.median(v):.2f}"
    out.append(seg)
print("  ".join(out))
PY
    done
done
