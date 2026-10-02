#!/bin/bash
# Runs INSIDE vllm_misha. Tests variant A (bf16 delta add) and B (A + half
# working chain) against the base kernel.
#
# Stage 1 isolates the change at a fixed bn=2 tile=5120.
# Stage 2 gives each variant the biggest tile its UB footprint allows at bn=1.
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
K="$WORK/add_lora_swiglu_quant.cpp"

build() { # name variant bn tile
    local name=$1 var=$2 bn=$3 tile=$4
    local V=$WORK/v_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    if [ "$var" = "base" ]; then
        python3 "$WORK/make_variant.py" "$K" "$V/add_lora_swiglu_quant.cpp" NONE "$bn" "$tile"
    else
        python3 "$WORK/make_variant.py" "$K" "$V/add_lora_swiglu_quant.cpp" "$var" "$bn" "$tile"
    fi
    cd "$V/build"
    cmake .. -DSOC_VERSION=$SOC_VERSION -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >/dev/null 2>&1
    if ! make -j16 >"$V/build.log" 2>&1; then
        echo "### $name BUILD FAILED"; tail -20 "$V/build.log"; return 1
    fi
}

run() { # name T W delta amp
    local name=$1 T=$2 W=$3 D=$4 amp=$5
    local V=$WORK/v_$name
    export LD_LIBRARY_PATH="$V/build/lib:${LD_LIBRARY_PATH:-}"
    local out; out=$("$V/build/bench" $T $W $D 50 $amp 2>&1)
    local us; us=$(echo "$out" | sed -n 's/.*| \([0-9.]*\) us.*/\1/p')
    printf "  %-8s T=%-6s W=%-5s d=%s amp=%-4s %8s us\n" "$name" "$T" "$W" "$D" "$amp" "$us"
    echo "$out" | grep -E '^ref=|non-finite' | sed 's/^/      /'
}

prof() { # name
    local V=$WORK/v_$1 P=/tmp/prof_ab_$1
    export LD_LIBRARY_PATH="$V/build/lib:${LD_LIBRARY_PATH:-}"
    rm -rf $P; mkdir -p $P
    msprof --output=$P --application="$V/build/bench 8192 2048 1 20" \
           --aic-metrics=PipeUtilization --ai-core=on --task-time=on >/dev/null 2>&1 || true
    local SUM; SUM=$(find $P -name 'op_summary*.csv' 2>/dev/null | head -1)
    [ -z "$SUM" ] && { echo "      (no profile)"; return; }
    python3 - "$SUM" <<'PY' | sed 's/^/      /'
import csv,sys
rows=[r for r in csv.DictReader(open(sys.argv[1])) if 'swiglu' in (r.get('Op Name') or '').lower()]
def m(k):
    v=[float(r[k]) for r in rows if r.get(k) not in (None,'','N/A')]
    return sum(v)/len(v) if v else float('nan')
print("vec=%.3f scalar=%.3f mte2=%.3f mte3=%.3f" %
      (m('aiv_vec_ratio'),m('aiv_scalar_ratio'),m('aiv_mte2_ratio'),m('aiv_mte3_ratio')))
PY
}

echo "##### stage 1: isolate the change, bn=2 tile=5120 #####"
for v in base A B; do
    n="s1$v"
    build "$n" "$v" 2 5120 || continue
    echo "--- $v ---"
    run "$n" 8192 2048 1 2.0
    run "$n" 8192 2048 0 2.0
    prof "$n"
done

echo
echo "##### stage 2: best tile per variant at bn=1 #####"
build s2base base 1 8192 && { echo "--- base bn=1 tile=8192 ---"; run s2base 8192 2048 1 2.0; prof s2base; }
build s2A    A    1 8192 && { echo "--- A    bn=1 tile=8192 ---"; run s2A    8192 2048 1 2.0; prof s2A; }
build s2B    B    1 10240 && { echo "--- B    bn=1 tile=10240 ---"; run s2B  8192 2048 1 2.0; prof s2B; }

echo
echo "##### half-range stress (variant B overflow tail) #####"
for amp in 2.0 8.0 20.0; do
    run s1B 8192 2048 1 $amp
done
echo "--- base at the same amplitudes, for comparison ---"
for amp in 2.0 8.0 20.0; do
    run s1base 8192 2048 1 $amp
done
