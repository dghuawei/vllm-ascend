#!/bin/bash
# Runs INSIDE vllm_misha. Budget-derived tile (rows from the measured 184 KB UB
# ceiling, accounting for hasDelta) vs the fixed-TILE_ELEMENTS configs.
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
K="$WORK/add_lora_swiglu_quant.cpp"

build() { # name variant bn tile maxtok
    local V=$WORK/b_$1
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    python3 "$WORK/make_variant.py" "$K" "$V/add_lora_swiglu_quant.cpp" "$2" "$3" "$4"
    sed -i "s/^constexpr uint32_t MAX_TOKENS_PER_TILE = .*/constexpr uint32_t MAX_TOKENS_PER_TILE = $5;/" \
        "$V/add_lora_swiglu_quant.cpp"
    cd "$V/build"
    cmake .. -DSOC_VERSION=$SOC_VERSION -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >/dev/null 2>&1
    make -j16 >"$V/build.log" 2>&1 || { echo "  $1 BUILD FAILED"; tail -15 "$V/build.log"; return 1; }
}

one() { # name T W delta
    local V=$WORK/b_$1
    export LD_LIBRARY_PATH="$V/build/lib:${LD_LIBRARY_PATH:-}"
    local out
    if ! out=$("$V/build/bench" $2 $3 $4 30 2>&1); then printf "%12s " "ABORT"; return; fi
    local us off1 offm v
    us=$(echo "$out"  | sed -n 's/.*| \([0-9.]*\) us.*/\1/p')
    off1=$(echo "$out" | sed -n 's/^ref=fp32add.*y_off1=\([0-9.]*\) *%.*/\1/p')
    offm=$(echo "$out" | sed -n 's/^ref=fp32add.*y_off>1=\([0-9.eE+-]*\)%.*/\1/p')
    v=""
    awk -v a="$off1" 'BEGIN{exit !(a>2.0)}'  && v="!"
    awk -v a="$offm" 'BEGIN{exit !(a>0.01)}' && v="!"
    echo "$out" | grep -q 'non-finite' && v="!"
    printf "%11s%s " "$us" "${v:- }"
}

# TILE_ELEMENTS is irrelevant for BUDGET; MAX_TOKENS_PER_TILE=16 throughout
build fix1  NONE   1 8192 16 || exit 1
build fix2  NONE   2 5120 8  || exit 1
build bud1  BUDGET 1 0    16 || exit 1
build bud2  BUDGET 2 0    16 || exit 1

echo "##### widths, T=16384 #####"
for d in 1 0; do
  echo "--- delta=$d ---"
  printf "%-26s %s\n" "config" "W=512       W=1024      W=1408      W=2048      W=4096      W=5120"
  for c in "fix2 bn=2 tile=5120 mt=8" "fix1 bn=1 tile=8192 mt=16" "bud2 bn=2 budget mt=16" "bud1 bn=1 budget mt=16"; do
    set -- $c; n=$1; shift
    printf "%-26s " "$*"
    for W in 512 1024 1408 2048 4096 5120; do one "$n" 16384 $W $d; done
    echo
  done
done

echo
echo "##### prefill T range, W=2048 #####"
printf "%-26s %s\n" "config" "T=4096      T=8192      T=16384     T=32768     T=65536     T=131072"
for c in "fix2 bn=2 tile=5120 mt=8" "fix1 bn=1 tile=8192 mt=16" "bud1 bn=1 budget mt=16"; do
  set -- $c; n=$1; shift
  printf "%-26s " "$*"
  for T in 4096 8192 16384 32768 65536 131072; do one "$n" $T 2048 1; done
  echo
done

echo
echo "##### does BUDGET raise the max supported width past 5120? #####"
for W in 6144 8192 8704; do
    printf "  W=%-5s bn=1 budget: " "$W"; one bud1 4096 $W 1; echo
done
echo "  (a '!' marks a failed correctness check; ABORT means InitBuffer rejected it)"
