#!/bin/bash
# Runs INSIDE the vllm_misha container.
#
# Two experiments on add_lora_swiglu_quant:
#   1. BUFFER_NUM sweep (TILE_ELEMENTS re-derived so each depth uses ~the same UB)
#   2. ablation, to find where aiv_scalar_ratio comes from
#
# Each variant is a sed'ed COPY of the kernel; the reviewed kernel is untouched.
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}

T=8192; W=2048; D=1
K="$WORK/add_lora_swiglu_quant.cpp"

# variant <name> <buffer_num> <tile_elements> <ablation>
build_and_measure() {
    local name=$1 bn=$2 tile=$3 abl=$4
    local V=$WORK/var_$name
    rm -rf "$V"; mkdir -p "$V"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$K" "$V/add_lora_swiglu_quant.cpp"
    local KV="$V/add_lora_swiglu_quant.cpp"

    sed -i "s/^constexpr int32_t BUFFER_NUM = .*/constexpr int32_t BUFFER_NUM = $bn;/" "$KV"
    sed -i "s/^constexpr uint32_t TILE_ELEMENTS = .*/constexpr uint32_t TILE_ELEMENTS = $tile;/" "$KV"

    case "$abl" in
      none) ;;
      noswiglu)
        # plain Mul in place of the library SwiGLU: removes its internal
        # SetMaskCount/SetVectorMask/SetMaskNorm/ResetMask + round/tail scalar math
        sed -i "s|AscendC::SwiGLU<float, false>(activated, up, gate, SWIGLU_BETA, numElements);|Mul(activated, up, gate, numElements);|" "$KV"
        ;;
      noreduce)
        # fixed scale: drops ReduceMax, the V_S/S_V round trip and the scalar loop
        python3 - "$KV" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
start=s.index('        Abs(gate, activated, numElements);')
end=s.index('        AscendC::LocalTensor<half> halfLocal')
s=s[:start]+"""        for (uint32_t i = 0; i < numRows; i++) {
            scaleValues_[i] = 1.0f / INT8_MAX_VALUE;
            Muls(activated[i * width_], activated[i * width_], INT8_MAX_VALUE, width_);
        }
        AscendC::PipeBarrier<PIPE_V>();

"""+s[end:]
open(p,'w').write(s)
PY
        ;;
      unaligned)
        # put the ReduceMax destinations back at 4-byte spacing
        sed -i "s/maxs\[i \* UB_BLOCK_FLOATS\]/maxs[i]/" "$KV"
        sed -i "s/maxs.GetValue(i \* UB_BLOCK_FLOATS)/maxs.GetValue(i)/" "$KV"
        ;;
    esac

    mkdir -p "$V/build"; cd "$V/build"
    if ! cmake .. -DSOC_VERSION=$SOC_VERSION -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >/dev/null 2>&1 \
       || ! make -j16 >"$V/build.log" 2>&1; then
        printf "%-12s bn=%d tile=%-5d  BUILD FAILED (see %s)\n" "$name" "$bn" "$tile" "$V/build.log"
        return
    fi

    export LD_LIBRARY_PATH="$V/build/lib:${LD_LIBRARY_PATH:-}"
    local out us
    out=$("$V/build/bench" $T $W $D 50 2>&1 | tail -1)
    us=$(echo "$out" | sed -n 's/.*| \([0-9.]*\) us.*/\1/p')

    local P=/tmp/prof_var_$name
    rm -rf $P; mkdir -p $P
    msprof --output=$P --application="$V/build/bench $T $W $D 20" \
           --aic-metrics=PipeUtilization --ai-core=on --task-time=on >/dev/null 2>&1 || true
    local SUM
    SUM=$(find $P -name 'op_summary*.csv' 2>/dev/null | head -1)
    local ratios="n/a"
    if [ -n "$SUM" ]; then
        ratios=$(python3 - "$SUM" <<'PY'
import csv,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]))
      if 'swiglu' in (r.get('Op Name') or '').lower()]
def m(k):
    v=[float(r[k]) for r in rows if r.get(k) not in (None,'','N/A')]
    return sum(v)/len(v) if v else float('nan')
print("vec=%.3f scalar=%.3f mte2=%.3f mte3=%.3f" %
      (m('aiv_vec_ratio'),m('aiv_scalar_ratio'),m('aiv_mte2_ratio'),m('aiv_mte3_ratio')))
PY
)
    fi
    printf "%-12s bn=%d tile=%-5d  %8s us  %s\n" "$name" "$bn" "$tile" "$us" "$ratios"
}

echo "=== T=$T W=$W delta=$D ==="
echo "--- BUFFER_NUM sweep (UB budget held ~constant) ---"
build_and_measure bn1 1 8192 none
build_and_measure bn2 2 5120 none
build_and_measure bn3 3 4096 none
build_and_measure bn4 4 3072 none

echo "--- ablation at bn=2 tile=5120 ---"
build_and_measure base       2 5120 none
build_and_measure noswiglu   2 5120 noswiglu
build_and_measure noreduce   2 5120 noreduce
build_and_measure unaligned  2 5120 unaligned
