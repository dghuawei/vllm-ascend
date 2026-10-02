#!/bin/bash
# Runs INSIDE the container on researchagent-node2 (910B3, CANN 9.0.1).
# Everything lives under /tmp/swiglu_ab. Touches no vllm-ascend checkout.
#
#   phase A  base vs new vs wrm2pass, bn=1 tile=8192, all shapes
#   phase B  bn=1 vs bn=2 at tile=4096 (bn=2 does not fit at 8192)
#   phase C  msprof PipeUtilization on base/new/wrm2pass at bn=1 tile=8192
set -uo pipefail

WORK=/tmp/swiglu_ab
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=ascend910b3
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

build() {  # name srcfile buffer_num tile
    local name=$1 src=$2 bn=$3 tile=$4
    local V=$WORK/v_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    sed -E -e "s/^constexpr int32_t BUFFER_NUM = .*$/constexpr int32_t BUFFER_NUM = ${bn};/" \
           -e "s/^constexpr uint32_t TILE_ELEMENTS = .*$/constexpr uint32_t TILE_ELEMENTS = ${tile};/" \
        "$WORK/$src" > "$V/add_lora_swiglu_quant.cpp"
    grep -q "BUFFER_NUM = ${bn};" "$V/add_lora_swiglu_quant.cpp" || { echo "!! bn subst failed $name"; return 1; }
    grep -q "TILE_ELEMENTS = ${tile};" "$V/add_lora_swiglu_quant.cpp" || { echo "!! tile subst failed $name"; return 1; }
    ( cd "$V/build" \
      && cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 \
      && make -j16 >make.log 2>&1 ) || { echo "!! BUILD FAILED $name"; tail -30 "$V/build/make.log"; return 1; }
    echo "   built $name"
}

run() {  # name [extra]
    local V=$WORK/v_$1
    [ -x "$V/build/bench" ] || { echo "   (no binary $1)"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $1 ---"
    for shape in "8192 1408" "8192 2048" "8192 4096" "32768 2048"; do
        ( cd "$V/build" && ./bench $shape 1 50 )
    done
    echo "  no-delta:"
    ( cd "$V/build" && ./bench 8192 2048 0 50 )
}

echo "================ BUILD ================"
build base      base.cpp     1 8192
build new       new.cpp      1 8192
build wrm2pass  wrm2pass.cpp 1 8192
build base_t4096_bn1 base.cpp 1 4096
build base_t4096_bn2 base.cpp 2 4096
build new_t4096_bn1  new.cpp  1 4096
build new_t4096_bn2  new.cpp  2 4096

echo
echo "================ PHASE A: bn=1 tile=8192 ================"
run base
run new
run wrm2pass

echo
echo "================ PHASE B: bn1 vs bn2 at tile=4096 ================"
for n in base_t4096_bn1 base_t4096_bn2 new_t4096_bn1 new_t4096_bn2; do
    V=$WORK/v_$n
    [ -x "$V/build/bench" ] || { echo "   (no binary $n)"; continue; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $n ---"
    for shape in "8192 1408" "8192 2048" "8192 4096"; do
        ( cd "$V/build" && ./bench $shape 1 50 )
    done
done

echo
echo "================ PHASE C: PIPES (8192x2048 d=1) ================"
for name in base new wrm2pass; do
    V=$WORK/v_$name
    [ -x "$V/build/bench" ] || continue
    OUT=/tmp/prof_$name; rm -rf "$OUT" && mkdir -p "$OUT"
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $name ---"
    ( cd "$V/build" && msprof --output="$OUT" \
        --application="$V/build/bench 8192 2048 1 20" \
        --aic-metrics=PipeUtilization --ai-core=on --task-time=on --ascendcl=on ) \
        >"$OUT/msprof.log" 2>&1
    SUM=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
    [ -z "$SUM" ] && { echo "   no op_summary"; tail -10 "$OUT/msprof.log"; continue; }
    python3 "$WORK/pipe_summary.py" "$SUM"
done
echo "================ DONE ================"
