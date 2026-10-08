#!/bin/bash
# Runs INSIDE a container. Night run 2026-10-07: A/B the z1/z2 kernel at the
# PRODUCTION decode geometries, with the index pattern production actually has
# (expert-sorted => runs of equal slots) as well as the historical round-robin
# worst case. Builds in /tmp only. One NPU, set by DEV.
set -uo pipefail
WORK=/tmp/lora_fused
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b4}
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-6}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

# B H1 R nSlices iters slots ngroups
SHAPES=${SHAPES:-"288 4096 16 2 300 96 32|288 4096 16 2 300 96 0|48 4096 16 2 500 96 1|48 4096 16 2 500 96 0|288 4096 16 2 300 96 96|96 4096 16 2 400 96 32"}

build() {
    local name=$1 V=$WORK/v_$1
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$WORK/$name.cpp" "$V/add_lora_fused.cpp"
    ( cd "$V/build" && cmake .. -DSOC_VERSION="$SOC_VERSION" \
        -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 && make -j16 >make.log 2>&1 ) \
      || { echo "!! BUILD FAILED $name"; tail -30 "$V/build/make.log" 2>/dev/null; return 1; }
    echo "   built $name"
}
run() {
    local V=$WORK/v_$1
    local _shapes
    [ -x "$V/build/bench" ] || { echo "   (no binary $1)"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- WALL $1 ---"
    local OLDIFS="$IFS" a
    IFS='|' read -ra _shapes <<< "$SHAPES"
    IFS="$OLDIFS"
    for a in "${_shapes[@]}"; do ( cd "$V/build" && ./bench $a ); done
}
prof() {
    local name=$1
    local V=$WORK/v_$name
    [ -x "$V/build/bench" ] || return 1
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- DEV $name ---"
    local OLDIFS="$IFS" sh
    IFS='|' read -ra _pshapes <<< "${PROF_SHAPES:-288 4096 16 2 30 96 32|48 4096 16 2 30 96 1}"
    IFS="$OLDIFS"
    for sh in "${_pshapes[@]}"; do
        OUT=/tmp/prof_night_${name}_$(echo $sh | tr ' ' '_'); rm -rf "$OUT" && mkdir -p "$OUT"
        ( cd "$V/build" && LF_NOCHECK=1 msprof --output="$OUT" --application="$V/build/bench $sh" \
            --aic-metrics=PipeUtilization --ai-core=on --task-time=on ) >"$OUT/msprof.log" 2>&1
        CSV=$(find "$OUT" -name 'kernel_details.csv' 2>/dev/null | head -1)
        # standalone msprof writes op_summary_*.csv, not kernel_details.csv
        [ -z "$CSV" ] && CSV=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
        [ -z "$CSV" ] && { echo "   $sh : NO CSV (see $OUT/msprof.log)"; continue; }
        python3 "$WORK/pipes.py" "$CSV" "$sh"
    done
}
cat > "$WORK/pipes.py" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
sh = sys.argv[2].split()
nk = [k for k in rows[0] if k.strip().lower() in ('name','op name','op_name')][0]
dk = [k for k in rows[0] if 'duration' in k.lower()][0]
out = [f"  B={sh[0]:>4} S={sh[3]} slots={sh[5]:>3} grp={sh[6] if len(sh)>6 else '-':>3}"]
for pref in ('add_lora_z1','add_lora_z2'):
    sel = [r for r in rows if r[nk].startswith(pref)]
    if not sel: continue
    d = [float(r[dk]) for r in sel]
    seg = f"{pref[-2:]}={st.median(d):8.2f}us n={len(d):4d}"
    for c in ('aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'):
        if c in sel[0]:
            v = [float(r[c]) for r in sel if r[c] not in ('','N/A')]
            if v: seg += f" {c.split('_')[1][:3]}={st.median(v):.2f}"
    out.append(seg)
print("  ".join(out))
PY
echo "================ BUILD (soc=$SOC_VERSION dev=$ASCEND_RT_VISIBLE_DEVICES) ================"
for v in ${VARIANTS:-base}; do build $v || exit 1; done
echo "================ RUN ================"
for v in ${VARIANTS:-base}; do run $v; done
if [ "${PROF:-1}" = "1" ]; then
  echo "================ DEVICE TIME + PIPES ================"
  for v in ${VARIANTS:-base}; do prof $v; done
fi
echo "================ DONE ================"
