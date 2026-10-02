#!/bin/bash
# Runs INSIDE the container. A/B the z1 scalar-accumulate kernel vs the
# vector-accumulate rewrite. Builds in /tmp only.
set -uo pipefail
WORK=/tmp/lora_fused
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b3}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

build() {
    local name=$1 src=$2 V=$WORK/v_$1
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$WORK/$src" "$V/add_lora_fused.cpp"
    ( cd "$V/build" && cmake .. -DSOC_VERSION="$SOC_VERSION" \
        -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 && make -j16 >make.log 2>&1 ) \
      || { echo "!! BUILD FAILED $name"; tail -30 "$V/build/make.log" 2>/dev/null || tail -20 "$V/build/cmake.log"; return 1; }
    echo "   built $name"
}

run() {
    local V=$WORK/v_$1
    [ -x "$V/build/bench" ] || { echo "   (no binary $1)"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $1 ---"
    # B H1 R nSlices iters   (decode and prefill-ish row counts from the profile)
    for a in "6 4096 16 1 500 96" "36 4096 16 2 500 96" "48 4096 16 2 500 96" \
             "48 4096 16 3 500 96" "288 4096 16 2 300 96" "288 4096 16 3 300 96" \
             "63 4096 16 2 500 7" "288 2048 32 2 300 96" "128 4096 64 1 300 96"; do
        ( cd "$V/build" && ./bench $a )
    done
}

echo "================ BUILD ================"
for v in ${VARIANTS:-base new}; do build $v $v.cpp || exit 1; done
if [ "${BUILD_ONLY:-0}" = "1" ]; then
    echo "BUILD_ONLY=1 -> skipping run"
    exit 0
fi
echo
echo "================ RUN ================"
for v in ${VARIANTS:-base new}; do run $v; done
echo "================ DONE ================"

if [ "${PROF:-0}" = "1" ]; then
    echo "================ DEVICE TIME + PIPES (decode shapes) ================"
    for name in ${VARIANTS:-base new}; do
        V=$WORK/v_$name
        [ -x "$V/build/bench" ] || continue
        export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
        echo "--- $name ---"
        for sh in "288 4096 16 2 30 96"; do
            OUT=/tmp/prof_lf_${name}_$(echo $sh | tr ' ' '_'); rm -rf "$OUT" && mkdir -p "$OUT"
            LOADPIDS=""
            for i in $(seq ${LOAD:-0}); do
                ( cd "$V/build" && LF_NOCHECK=1 ./bench 288 4096 16 2 100000 96 >/dev/null 2>&1 ) &
                LOADPIDS="$LOADPIDS $!"
            done
            [ -n "$LOADPIDS" ] && sleep 3
            ( cd "$V/build" && LF_NOCHECK=1 LF_MEMLOAD_MB="${MEMLOAD:-}" msprof --output="$OUT" --application="$V/build/bench $sh" \
                --aic-metrics=PipeUtilization --ai-core=on --task-time=on ) >"$OUT/msprof.log" 2>&1
            [ -n "$LOADPIDS" ] && kill $LOADPIDS 2>/dev/null
            CSV=$(find "$OUT" -name 'kernel_details.csv' 2>/dev/null | head -1)
            [ -z "$CSV" ] && CSV=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
            [ -z "$CSV" ] && { echo "   B=${sh%% *} S=? : no csv"; continue; }
            python3 - "$CSV" "$sh" <<'PY'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
sh=sys.argv[2].split()
nk=[k for k in rows[0] if k.strip().lower() in ('name','op name','op_name')][0]
dk=[k for k in rows[0] if 'duration' in k.lower()][0]
out=[f"  B={sh[0]:>4} S={sh[3]} slots={sh[5] if len(sh)>5 else 4:>4}"]
for pref in ('add_lora_z1','add_lora_z2'):
    sel=[r for r in rows if r[nk].startswith(pref)]
    if not sel: continue
    d=[float(r[dk]) for r in sel]
    seg=f"{pref[-2:]}={st.median(d):7.2f}us"
    for c in ('aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'):
        if c in sel[0]:
            v=[float(r[c]) for r in sel if r[c] not in ('','N/A')]
            if v: seg+=f" {c.split('_')[1][:3]}={st.median(v):.2f}"
    out.append(seg)
print("  ".join(out))
PY
        done
    done
fi
