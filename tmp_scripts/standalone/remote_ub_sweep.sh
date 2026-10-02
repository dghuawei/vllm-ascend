#!/bin/bash
# Runs INSIDE vllm_misha.
#
# Stage 1: how big can TILE_ELEMENTS get before UB overflows? UB overfill is
#          SILENT garbage on this hardware, so every point is correctness-checked.
# Stage 2: sweep MAX_TOKENS_PER_TILE x TILE_ELEMENTS across widths.
# Stage 3: best config vs current baseline over the prefill T range.
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
K="$WORK/add_lora_swiglu_quant.cpp"

build() { # name bn tile maxtok
    local name=$1 bn=$2 tile=$3 mt=$4
    local V=$WORK/u_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$K" "$V/add_lora_swiglu_quant.cpp"
    local KV="$V/add_lora_swiglu_quant.cpp"
    sed -i "s/^constexpr int32_t BUFFER_NUM = .*/constexpr int32_t BUFFER_NUM = $bn;/" "$KV"
    sed -i "s/^constexpr uint32_t TILE_ELEMENTS = .*/constexpr uint32_t TILE_ELEMENTS = $tile;/" "$KV"
    sed -i "s/^constexpr uint32_t MAX_TOKENS_PER_TILE = .*/constexpr uint32_t MAX_TOKENS_PER_TILE = $mt;/" "$KV"
    cd "$V/build"
    cmake .. -DSOC_VERSION=$SOC_VERSION -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >/dev/null 2>&1
    make -j16 >"$V/build.log" 2>&1 || { echo "  $name: BUILD FAILED"; return 1; }
}

# run one shape, print time + a PASS/FAIL verdict from the bf16add reference
one() { # name T W delta
    local V=$WORK/u_$1 T=$2 W=$3 D=$4
    export LD_LIBRARY_PATH="$V/build/lib:${LD_LIBRARY_PATH:-}"
    local out; out=$("$V/build/bench" $T $W $D 30 2>&1) || { echo "RUNFAIL"; return; }
    local us off1 offm nf
    us=$(echo "$out"  | sed -n 's/.*| \([0-9.]*\) us.*/\1/p')
    off1=$(echo "$out" | sed -n 's/^ref=fp32add.*y_off1=\([0-9.]*\) *%.*/\1/p')
    offm=$(echo "$out" | sed -n 's/^ref=fp32add.*y_off>1=\([0-9.eE+-]*\)%.*/\1/p')
    nf=$(echo "$out" | grep -c 'non-finite' || true)
    local verdict=ok
    awk -v a="$off1" 'BEGIN{exit !(a>2.0)}' && verdict=BAD_off1
    awk -v a="$offm" 'BEGIN{exit !(a>0.01)}' && verdict=BAD_offmore
    [ "$nf" != "0" ] && verdict=BAD_nonfinite
    printf "%8s us  %-12s" "$us" "$verdict"
}

echo "##### stage 1: max TILE_ELEMENTS at bn=1 (correctness-gated) #####"
for tile in 5120 7040 8192 8448 8704 9216; do
    n="t$tile"
    if build "$n" 1 $tile 16; then
        printf "  tile=%-5s " "$tile"; one "$n" 16384 2048 1; echo
    fi
done

echo
echo "##### stage 2: MAX_TOKENS_PER_TILE x TILE_ELEMENTS, T=16384, delta=1 #####"
printf "%-22s %s\n" "config" "W=512        W=1024       W=1408       W=2048       W=4096"
for cfg in "1 5120 8" "1 8192 8" "1 8192 16" "1 8192 32" "1 8448 16" "1 8448 32" "2 5120 8"; do
    set -- $cfg; bn=$1; tile=$2; mt=$3
    n="c${bn}_${tile}_${mt}"
    build "$n" $bn $tile $mt || continue
    printf "bn=%s tile=%-5s mt=%-3s " "$bn" "$tile" "$mt"
    for W in 512 1024 1408 2048 4096; do
        printf "%s" "$(one "$n" 16384 $W 1)"
    done
    echo
done

echo
echo "##### stage 3: prefill T range, W=2048, delta=1 #####"
printf "%-22s %s\n" "config" "T=4096     T=8192     T=16384    T=32768    T=65536"
for cfg in "2 5120 8" "1 8192 8" "1 8192 16"; do
    set -- $cfg; bn=$1; tile=$2; mt=$3
    n="c${bn}_${tile}_${mt}"
    [ -d "$WORK/u_$n/build" ] || build "$n" $bn $tile $mt || continue
    printf "bn=%s tile=%-5s mt=%-3s " "$bn" "$tile" "$mt"
    for T in 4096 8192 16384 32768 65536; do
        printf "%s" "$(one "$n" $T 2048 1)"
    done
    echo
done
