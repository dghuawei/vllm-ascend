#!/bin/bash
# Create the three-kernel optimisation arena outside both the vault and the
# main work repo, and populate it with the baseline kernel sources + the
# AscendC reference docs.
#
#   bash tmp_scripts/2026-10-07_arena_setup.sh
#
# Idempotent: re-running refreshes the baselines and docs but never touches
# agents/*/kernel (their working copies) once those exist.
set -euo pipefail

ARENA=/Users/arabel1a/work/huawei/qlora/kernel_arena
VA=/Users/arabel1a/work/huawei/qlora/vllm-ascend
SC=/Users/arabel1a/work/huawei/qlora/scatter_v2_acccelerate
MANUAL=/Users/arabel1a/agents/pearl/manual
VAULT=/Users/arabel1a/agents/vllm-ascend

mkdir -p "$ARENA"/{docs/ascendc,docs/notes}
for k in z1z2 swiglu scatter; do
    mkdir -p "$ARENA/$k"/{harness,baseline,results}
done

# ---- baselines -------------------------------------------------------------
cp "$VA/csrc/kernels/add_lora_fused.cpp" "$ARENA/z1z2/baseline/"
cp "$VA/csrc/kernels/types.h"            "$ARENA/z1z2/baseline/"

git -C "$VA" show origin/fused_addlora_swiglu_quant:csrc/kernels/add_lora_swiglu_quant.cpp \
    > "$ARENA/swiglu/baseline/add_lora_swiglu_quant.cpp"
git -C "$VA" show origin/fused_addlora_swiglu_quant:csrc/kernels/types.h \
    > "$ARENA/swiglu/baseline/types.h"

# scatter: the whole op dir is the build unit; the editable file is the one header
rm -rf "$ARENA/scatter/baseline/scatter_nd_update_v2"
cp -R "$SC/csrc/moe/scatter_nd_update_v2" "$ARENA/scatter/baseline/"
cp "$SC/csrc/moe/scatter_nd_update_v2/op_kernel/scatter_nd_update_row_split.h" \
   "$ARENA/scatter/baseline/scatter_nd_update_row_split.h"

# ---- reference docs (read-only copies; NOT the vault itself) ---------------
if [ -d "$MANUAL" ]; then
    rsync -a --delete "$MANUAL/" "$ARENA/docs/ascendc/"
fi
for f in ascendc-vector-geometry.md; do
    [ -f "$VAULT/$f" ] && cp "$VAULT/$f" "$ARENA/docs/notes/"
done
for f in ascendc-tbuf-tque-semantics.md ascendc-function-qualifiers.md \
         scalar-roundtrip-is-the-bubble.md bf16-add-blocked-dav-c220.md; do
    [ -f "$VAULT/summary/$f" ] && cp "$VAULT/summary/$f" "$ARENA/docs/notes/"
done

echo "arena at $ARENA"
find "$ARENA" -maxdepth 2 -type d | sort
echo "--- docs ---"
echo "ascendc manual files: $(ls "$ARENA/docs/ascendc" | wc -l)"
echo "distilled notes     : $(ls "$ARENA/docs/notes" | wc -l)"
