#!/bin/bash
# Runs INSIDE the va25_ds container on researchagent-node1 (910B3).
# A/B: committed scalar-sync baseline vs the scalar-free kernel, each at
# BUFFER_NUM 1 and 2, then msprof PipeUtilization on the bn1 pair.
#
# Everything lives under /tmp/swiglu_ab. Touches no vllm-ascend checkout.
set -uo pipefail

WORK=/tmp/swiglu_ab
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=ascend910b3

build() {  # name srcfile buffer_num
    local name=$1 src=$2 bn=$3
    local V=$WORK/v_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    sed -E "s/^constexpr int32_t BUFFER_NUM = .*$/constexpr int32_t BUFFER_NUM = ${bn};/" \
        "$WORK/$src" > "$V/add_lora_swiglu_quant.cpp"
    if ! grep -q "BUFFER_NUM = ${bn};" "$V/add_lora_swiglu_quant.cpp"; then
        echo "!! BUFFER_NUM substitution failed for $name"; return 1
    fi
    ( cd "$V/build" \
      && cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 \
      && make -j16 >make.log 2>&1 )
    if [ $? -ne 0 ]; then
        echo "!! BUILD FAILED: $name"
        tail -40 "$V/build/make.log" 2>/dev/null || tail -25 "$V/build/cmake.log"
        return 1
    fi
    echo "   built $name"
}

run() {  # name
    local name=$1
    local V=$WORK/v_$name
    [ -x "$V/build/bench" ] || { echo "   (no binary for $name)"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:${LD_LIBRARY_PATH_BASE:-}"
    echo "--- $name ---"
    for shape in "8192 1408" "8192 2048" "8192 4096" "32768 2048"; do
        ( cd "$V/build" && ./bench $shape 1 50 )
    done
    echo "  no-delta:"
    ( cd "$V/build" && ./bench 8192 2048 0 50 )
}

export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

echo "================ BUILD ================"
build base_bn1 base.cpp 1
build base_bn2 base.cpp 2
build new_bn1  new.cpp  1
build new_bn2  new.cpp  2

echo
echo "================ RUN ================"
run base_bn1
run base_bn2
run new_bn1
run new_bn2

echo
echo "================ PIPE UTILIZATION ================"
for name in base_bn1 new_bn1; do
    V=$WORK/v_$name
    [ -x "$V/build/bench" ] || continue
    OUT=/tmp/prof_$name
    rm -rf "$OUT" && mkdir -p "$OUT"
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:${LD_LIBRARY_PATH_BASE:-}"
    echo "--- msprof $name (8192 2048 delta=1) ---"
    ( cd "$V/build" && msprof --output="$OUT" \
        --application="$V/build/bench 8192 2048 1 20" \
        --aic-metrics=PipeUtilization \
        --ai-core=on --task-time=on --ascendcl=on ) >"$OUT/msprof.log" 2>&1
    SUM=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
    if [ -z "$SUM" ]; then
        echo "   no op_summary; msprof tail:"; tail -12 "$OUT/msprof.log"
        continue
    fi
    python3 "$WORK/pipe_summary.py" "$SUM"
done
echo "================ DONE ================"
