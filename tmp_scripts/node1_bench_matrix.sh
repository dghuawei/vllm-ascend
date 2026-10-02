#!/bin/bash
# Local driver: ship the three variant benchmarks to researchagent-node1,
# into the va25_ds container, and run them over the shared shape matrix.
# Stages only into /tmp -- touches no vllm-ascend checkout on the remote side.
#
#   usage: node1_bench_matrix.sh [baseline|wrm2pass|unfused|all|table]
#   FORCE=1 skips the idle guard (do not).
set -euo pipefail

REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SRC=$REPO/tmp_scripts/standalone
HOST=${HOST:-researchagent-node1}
CTR=${CTR:-va25_ds}
STAGE=/tmp/swiglu_matrix
OUT=$REPO/tmp_scripts/results
WHICH=${1:-all}

mkdir -p "$OUT"

table() {
    echo
    echo ">>> ===== DEVICE TIME (msprof Task Duration -- the real number) ====="
    python3 "$SRC/tabulate_matrix.py" --variants unfused_dev,baseline_dev,wrm2pass_dev \
        "$OUT"/*.log
    echo
    echo ">>> ===== WALL CLOCK (launcher included) ====="
    python3 "$SRC/tabulate_matrix.py" --variants unfused,baseline,wrm2pass "$OUT"/*.log
}

if [ "$WHICH" = table ]; then table; exit 0; fi

if [ "${FORCE:-0}" != "1" ]; then
    echo ">>> idle guard on $HOST"
    if ! bash "$REPO/tmp_scripts/check_npu_idle.sh" "$HOST" 6 5; then
        echo ">>> ABORTING: NPUs are not idle. Re-run when they are, or FORCE=1 to override."
        exit 1
    fi
fi

echo ">>> staging to $HOST:$CTR:$STAGE"
rm -rf "$SRC/.stage_matrix" && mkdir -p "$SRC/.stage_matrix"
cp "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp"                      "$SRC/.stage_matrix/baseline.cpp"
cp "$REPO/tmp_scripts/variants/add_lora_swiglu_quant_wrm2pass.cpp"     "$SRC/.stage_matrix/wrm2pass.cpp"
cp "$REPO/tmp_scripts/variants/add_lora_swiglu_quant_pretail.cpp"      "$SRC/.stage_matrix/pretail.cpp"
cp "$REPO/csrc/kernels/add_lora_swiglu_quant.cpp"                      "$SRC/.stage_matrix/new.cpp"
cp "$REPO/csrc/kernels/types.h"                                         "$SRC/.stage_matrix/"
cp "$SRC/CMakeLists.txt" "$SRC/main.cpp" "$SRC/shapes.env" "$SRC/run_matrix_common.sh" \
   "$SRC/device_time.py" \
   "$SRC/node1_run_baseline.sh" "$SRC/node1_run_wrm2pass.sh" \
   "$SRC/node1_run_unfused.sh" "$SRC/node1_run_unfused.py" "$SRC/node1_run_extra.sh" \
   "$SRC/node1_run_verify.sh" "$SRC/node1_run_tailcost.sh" \
   "$SRC/.stage_matrix/"

tar -czf "$SRC/.stage_matrix.tgz" -C "$SRC/.stage_matrix" .
scp -q "$SRC/.stage_matrix.tgz" "$HOST:/tmp/swiglu_matrix.tgz"
ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/swiglu_matrix.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
"

run_one() {  # name script
    echo
    echo ">>> ===== $1 ====="
    ssh "$HOST" "sudo docker exec $CTR bash $STAGE/$2" 2>&1 | tee "$OUT/$1.log"
}

case "$WHICH" in
    baseline) run_one baseline node1_run_baseline.sh ;;
    wrm2pass) run_one wrm2pass node1_run_wrm2pass.sh ;;
    unfused)  run_one unfused  node1_run_unfused.sh ;;
    extra)    run_one extra    node1_run_extra.sh ;;
    verify)   run_one verify   node1_run_verify.sh ;;
    tailcost) run_one tailcost node1_run_tailcost.sh ;;
    all)
        run_one baseline node1_run_baseline.sh
        run_one wrm2pass node1_run_wrm2pass.sh
        run_one unfused  node1_run_unfused.sh
        table
        ;;
    *) echo "unknown target: $WHICH"; exit 1 ;;
esac
