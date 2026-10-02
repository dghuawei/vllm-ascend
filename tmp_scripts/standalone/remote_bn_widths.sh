#!/bin/bash
# Runs INSIDE vllm_misha. Does BUFFER_NUM=1(big tile) vs 2(current) hold up
# across widths, and is bn=1 still correct? Prints the bench check line too.
set -euo pipefail

WORK=/tmp/swiglu_standalone
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910b4
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
K="$WORK/add_lora_swiglu_quant.cpp"

build() {
    local name=$1 bn=$2 tile=$3
    local V=$WORK/var_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$K" "$V/add_lora_swiglu_quant.cpp"
    sed -i "s/^constexpr int32_t BUFFER_NUM = .*/constexpr int32_t BUFFER_NUM = $bn;/" "$V/add_lora_swiglu_quant.cpp"
    sed -i "s/^constexpr uint32_t TILE_ELEMENTS = .*/constexpr uint32_t TILE_ELEMENTS = $tile;/" "$V/add_lora_swiglu_quant.cpp"
    cd "$V/build"
    cmake .. -DSOC_VERSION=$SOC_VERSION -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >/dev/null 2>&1
    make -j16 >"$V/build.log" 2>&1 || { echo "$name BUILD FAILED"; return 1; }
}

build bn1 1 8192
build bn2 2 5120

for shape in "8192 1408" "8192 2048" "8192 4096" "32768 2048" "2048 2048"; do
  for d in 1 0; do
    for v in bn1 bn2; do
      export LD_LIBRARY_PATH="$WORK/var_$v/build/lib:${LD_LIBRARY_PATH:-}"
      out=$("$WORK/var_$v/build/bench" $shape $d 50 2>&1)
      chk=$(echo "$out" | sed -n 's/.*y off-by-1=\([0-9.]*\)%.*y off>1=\([0-9.]*\)%.*/off1=\1% off>1=\2%/p')
      us=$(echo "$out" | sed -n 's/.*| \([0-9.]*\) us.*/\1/p')
      gb=$(echo "$out" | sed -n 's/.*| \([0-9.]*\) GB\/s.*/\1/p')
      printf "%-11s d=%s %-5s  %8s us  %7s GB/s   %s\n" "$shape" "$d" "$v" "$us" "$gb" "$chk"
    done
  done
done
