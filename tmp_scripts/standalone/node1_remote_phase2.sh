#!/bin/bash
# Phase 2 on researchagent-node1 / va25_ds (910B3).
#  (a) full PipeUtilization for the already-built base_bn1 and new_bn1
#  (b) the pipelining question: BUFFER_NUM 1 vs 2 at a tile that actually
#      FITS in UB. bn=2 at TILE_ELEMENTS=8192 overflows UB (ACL 507035),
#      so both bn values are rebuilt at TILE_ELEMENTS=4096 for a fair test.
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
    grep -q "BUFFER_NUM = ${bn};" "$V/add_lora_swiglu_quant.cpp" || { echo "!! bn sub failed $name"; return 1; }
    grep -q "TILE_ELEMENTS = ${tile};" "$V/add_lora_swiglu_quant.cpp" || { echo "!! tile sub failed $name"; return 1; }
    ( cd "$V/build" \
      && cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 \
      && make -j16 >make.log 2>&1 ) || { echo "!! BUILD FAILED $name"; tail -25 "$V/build/make.log"; return 1; }
    echo "   built $name (bn=$bn tile=$tile)"
}

run() {  # name
    local V=$WORK/v_$1
    [ -x "$V/build/bench" ] || { echo "   (no binary $1)"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $1 ---"
    for shape in "8192 1408" "8192 2048" "8192 4096"; do
        ( cd "$V/build" && ./bench $shape 1 50 2>&1 | tail -1 )
    done
}

echo "================ (a) PIPE UTILIZATION ================"
for name in base_bn1 new_bn1; do
    V=$WORK/v_$name
    [ -x "$V/build/bench" ] || { echo "   ($name not built)"; continue; }
    OUT=/tmp/prof2_$name
    rm -rf "$OUT" && mkdir -p "$OUT"
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $name (8192 2048 delta=1) ---"
    ( cd "$V/build" && msprof --output="$OUT" \
        --application="$V/build/bench 8192 2048 1 20" \
        --aic-metrics=PipeUtilization \
        --ai-core=on --task-time=on --ascendcl=on ) >"$OUT/msprof.log" 2>&1
    SUM=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
    [ -z "$SUM" ] && { echo "   no op_summary"; tail -10 "$OUT/msprof.log"; continue; }
    python3 "$WORK/pipe_summary.py" "$SUM"
done

echo
echo "================ (b) PIPELINING: bn1 vs bn2 at tile=4096 ================"
build base_bn1_t4096 base.cpp 1 4096
build base_bn2_t4096 base.cpp 2 4096
build new_bn1_t4096  new.cpp  1 4096
build new_bn2_t4096  new.cpp  2 4096
echo
run base_bn1_t4096
run base_bn2_t4096
run new_bn1_t4096
run new_bn2_t4096
echo "================ DONE ================"
