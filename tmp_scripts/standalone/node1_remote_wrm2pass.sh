#!/bin/bash
# Phase 3: does doing the WHOLE reduction with WholeReduceMax beat
# per-row ReduceMax + a compaction pass? Same everything else.
set -uo pipefail

WORK=/tmp/swiglu_ab
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=ascend910b3
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"

build() {  # name srcfile
    local name=$1 src=$2
    local V=$WORK/v_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$WORK/$src" "$V/add_lora_swiglu_quant.cpp"
    ( cd "$V/build" \
      && cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 \
      && make -j16 >make.log 2>&1 ) || { echo "!! BUILD FAILED $name"; tail -25 "$V/build/make.log"; return 1; }
    echo "   built $name"
}

run() {  # name
    local V=$WORK/v_$1
    [ -x "$V/build/bench" ] || { echo "   (no binary $1)"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $1 ---"
    for shape in "8192 1408" "8192 2048" "8192 4096" "32768 2048"; do
        ( cd "$V/build" && ./bench $shape 1 50 2>&1 | tail -3 )
    done
}

build new_ref   new.cpp
build wrm2pass  wrm2pass.cpp
echo
run new_ref
run wrm2pass

echo
echo "================ PIPES ================"
for name in new_ref wrm2pass; do
    V=$WORK/v_$name
    [ -x "$V/build/bench" ] || continue
    OUT=/tmp/prof3_$name; rm -rf "$OUT" && mkdir -p "$OUT"
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    echo "--- $name ---"
    ( cd "$V/build" && msprof --output="$OUT" \
        --application="$V/build/bench 8192 2048 1 20" \
        --aic-metrics=PipeUtilization --ai-core=on --task-time=on --ascendcl=on ) \
        >"$OUT/msprof.log" 2>&1
    SUM=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
    [ -z "$SUM" ] && { echo "   no op_summary"; continue; }
    python3 "$WORK/pipe_summary.py" "$SUM" | grep -E "Task Duration|vec_|scalar_|mte2_|mte3_|total_cycles"
done
echo "================ DONE ================"
