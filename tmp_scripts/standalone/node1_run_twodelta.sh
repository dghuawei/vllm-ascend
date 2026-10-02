#!/bin/bash
# Runs INSIDE the container. A/B the single-interleaved-delta kernel (HEAD)
# against the two-separate-delta-buffer kernel (work tree).
#
# The impl SIGNATURE changed, so each variant needs its own main.cpp:
#   base -> main_base.cpp (one [T,2W] delta buffer)
#   new  -> main_new.cpp  (two [T,W] buffers, de-interleaved on the host from
#                          the SAME h_delta array, so the fp32 reference and
#                          the RNG stream are byte-identical between variants)
set -uo pipefail
WORK=/tmp/swiglu_twodelta
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b3}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"
ROUNDS=${ROUNDS:-3}

build() {  # name kernelsrc mainsrc
    local name=$1 ksrc=$2 msrc=$3 V=$WORK/v_$1
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/CMakeLists.txt" "$V/"
    cp "$WORK/$ksrc" "$V/add_lora_swiglu_quant.cpp"
    cp "$WORK/$msrc" "$V/main.cpp"
    ( cd "$V/build" \
      && cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 \
      && make -j16 >make.log 2>&1 ) \
      || { echo "!! BUILD FAILED $name"; tail -40 "$V/build/make.log" 2>/dev/null \
           || tail -20 "$V/build/cmake.log"; return 1; }
    echo "   built $name"
}

run_list() {  # name shapefile [grep]
    local name=$1 list=$2 pat=${3:-'^T=|^ref=fp32|non-finite|rror'} V=$WORK/v_$1
    [ -x "$V/build/bench" ] || { echo "!! no binary $name"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    ( cd "$V/build" && ./bench --matrix "$list" 2>&1 ) | grep -E "$pat"
}

# Every branch of the width logic, all multiples of 16 (the new delta guard).
# 4112 / 8192 are left out: they abort by design and would kill the process
# before the later shapes run. Last two rows are delta=0 controls.
cat > "$WORK/shapes_verify.txt" <<SHP
4096 32 1 50
4096 48 1 50
4096 64 1 50
4096 112 1 50
4096 256 1 50
4096 528 1 50
4096 768 1 50
4096 1024 1 50
4096 1040 1 50
4096 1392 1 50
4096 1408 1 50
4096 2048 1 50
4096 2064 1 50
2048 4096 1 30
128 256 0 200
128 1392 0 200
SHP

# Timing: the production widths plus the span, delta=1 only (delta=0 does not
# touch the changed code).
cat > "$WORK/shapes_time.txt" <<SHP
4096 256 1 200
4096 768 1 200
4096 1024 1 200
4096 1408 1 200
4096 2048 1 200
2048 4096 1 100
16384 256 1 50
16384 2048 1 50
SHP

echo "================ BUILD ================"
build base base.cpp main_base.cpp || exit 1
build new  new.cpp  main_new.cpp  || exit 1
# third variant only when staged: two buffers, per-row Casts (isolates which
# half of the change moved the int8 LSB stats at W%64 != 0)
[ -f "$WORK/new_rowcast.cpp" ] && { build new_rowcast new_rowcast.cpp main_new.cpp || exit 1; }

for v in base new ${EXTRA_VARIANTS:-}; do
    echo "================ VERIFY $v ================"
    run_list "$v" "$WORK/shapes_verify.txt"
done

echo "================ TIMING (interleaved, $ROUNDS rounds) ================"
for r in $(seq "$ROUNDS"); do
    for v in base new; do
        run_list "$v" "$WORK/shapes_time.txt" '^T=' | sed "s/^/RUN $v /"
    done
done

echo "================ GUARD: W=40 with delta must abort ================"
echo "4096 40 1 10" > "$WORK/shapes_guard.txt"
run_list new "$WORK/shapes_guard.txt" '.' || true
echo "   (exit $? -- 134 = abort, which is the expected outcome)"

echo "================ DONE twodelta ================"
