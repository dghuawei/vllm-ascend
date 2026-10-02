#!/bin/bash
# What does the tail call cost on widths where it is NOT needed?
# Variant-to-variant differences here are ~0.5%, run-to-run drift is ~4%, so
# a single A-then-B pass cannot see it. Alternate A/B/A/B... and take the MIN
# per shape: min is the estimator least polluted by transient interference.
set -uo pipefail
WORK=/tmp/swiglu_matrix
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910b3}
export LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH:-}"
source "$WORK/shapes.env"
source "$WORK/run_matrix_common.sh"
ROUNDS=${ROUNDS:-5}

cat > "$WORK/shapes_tail.txt" <<SHP
4096 256 1 200
4096 768 1 200
4096 1024 1 200
4096 1408 1 200
4096 2048 1 200
2048 4096 1 100
SHP

build_variant wrm2pass wrm2pass.cpp || exit 1
build_variant new      new.cpp      || exit 1

for r in $(seq "$ROUNDS"); do
    for v in wrm2pass new; do
        V=$WORK/v_$v
        export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
        ( cd "$V/build" && ./bench --matrix "$WORK/shapes_tail.txt" 2>&1 ) \
            | grep -E '^T=' | sed "s/^/RUN $v /"
    done
done
echo "================ DONE tailcost ================"
