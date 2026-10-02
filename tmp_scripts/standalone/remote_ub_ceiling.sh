#!/bin/bash
# Runs INSIDE vllm_misha. Finds the TRUE UB ceiling.
#
# At W=512 with a high MAX_TOKENS_PER_TILE, numRowsPerTile_ = TILE_ELEMENTS/512
# is not clamped, so the allocation really is TILE_ELEMENTS columns and the
# budget is fully exercised. UB overfill is SILENT garbage here, so the verdict
# comes from the correctness check, not from whether it runs.
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
K="$WORK/add_lora_swiglu_quant.cpp"

build() { # name bn tile maxtok
    local V=$WORK/z_$1
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$K" "$V/add_lora_swiglu_quant.cpp"
    sed -i "s/^constexpr int32_t BUFFER_NUM = .*/constexpr int32_t BUFFER_NUM = $2;/" "$V/add_lora_swiglu_quant.cpp"
    sed -i "s/^constexpr uint32_t TILE_ELEMENTS = .*/constexpr uint32_t TILE_ELEMENTS = $3;/" "$V/add_lora_swiglu_quant.cpp"
    sed -i "s/^constexpr uint32_t MAX_TOKENS_PER_TILE = .*/constexpr uint32_t MAX_TOKENS_PER_TILE = $4;/" "$V/add_lora_swiglu_quant.cpp"
    cd "$V/build"
    cmake .. -DSOC_VERSION=$SOC_VERSION -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >/dev/null 2>&1
    make -j16 >"$V/build.log" 2>&1
}

check() { # name T W delta
    local V=$WORK/z_$1
    export LD_LIBRARY_PATH="$V/build/lib:${LD_LIBRARY_PATH:-}"
    local out
    if ! out=$("$V/build/bench" $2 $3 $4 30 2>&1); then echo "  RUNTIME FAILURE"; return; fi
    local us off1 offm
    us=$(echo "$out"  | sed -n 's/.*| \([0-9.]*\) us.*/\1/p')
    off1=$(echo "$out" | sed -n 's/^ref=fp32add.*y_off1=\([0-9.]*\) *%.*/\1/p')
    offm=$(echo "$out" | sed -n 's/^ref=fp32add.*y_off>1=\([0-9.eE+-]*\)%.*/\1/p')
    local v=PASS
    awk -v a="$off1" 'BEGIN{exit !(a>2.0)}'   && v="CORRUPT(off1=$off1%)"
    awk -v a="$offm" 'BEGIN{exit !(a>0.01)}'  && v="CORRUPT(off>1=$offm%)"
    echo "$out" | grep -q 'non-finite' && v="CORRUPT(nonfinite)"
    printf "%9s us   %s\n" "$us" "$v"
}

echo "bn=1, W=512, delta=1, MAX_TOKENS_PER_TILE=64 -> rows = TILE/512, unclamped"
echo "23 bytes per tile column at bn=1+delta, so UB used ~= 23*TILE"
for tile in 7680 8192 8704 9216 9728; do
    rows=$((tile/512)); ub=$((23*tile))
    printf "  tile=%-5s rows=%-3s UB~%6s B (%3s KB)  " "$tile" "$rows" "$ub" "$((ub/1024))"
    if build "c$tile" 1 $tile 64 2>/dev/null; then check "c$tile" 16384 512 1; else echo "BUILD FAILED"; fi
done

echo
echo "same, bn=1 WITHOUT delta (19 bytes per column -> more rows should fit)"
for tile in 8192 9216 10240 11264; do
    rows=$((tile/512)); ub=$((19*tile))
    printf "  tile=%-5s rows=%-3s UB~%6s B (%3s KB)  " "$tile" "$rows" "$ub" "$((ub/1024))"
    if [ -d "$WORK/z_c$tile/build" ] || build "c$tile" 1 $tile 64 2>/dev/null; then
        check "c$tile" 16384 512 0
    else echo "BUILD FAILED"; fi
done
