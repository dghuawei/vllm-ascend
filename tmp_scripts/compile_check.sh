#!/bin/bash
# Does the work tree's csrc actually compile? Ships the current tree to node1 and
# builds the custom ops in an ISOLATED /tmp copy inside the container, so the
# shared /vllm-workspace/vllm-ascend checkout is never touched.
#
# This is the gate for an e2e run: the kernel was verified standalone, but
# torch_binding.cpp / torch_binding_meta.cpp have never been through a compiler.
set -euo pipefail
REPO=/Users/arabel1a/work/huawei/qlora/vllm-ascend
HOST=${HOST:-researchagent-node1}
CTR=${CTR:-va25_ds}
DEST=${DEST:-/tmp/ab_build}
SOC=${SOC:-ascend910b3}
JOBS=${JOBS:-32}
TGZ=$REPO/tmp_scripts/.tree.tgz

echo "=== packing work tree (no .git, no tmp_scripts) ==="
# NOTE: the excludes must be ANCHORED (./build, not build): csrc/cmake/third_party/
# build/modules/patch holds patch files the aclnn op-host build needs, and a bare
# --exclude=build silently drops them -> "protobuf-hide_absl_symbols.patch: No such file"
# Do NOT exclude "build": macOS bsdtar matches that pattern against nested paths
# too and silently drops csrc/cmake/third_party/build/modules/patch, which the
# aclnn op-host build needs ("protobuf-hide_absl_symbols.patch: No such file").
tar --exclude=./.git --exclude=./tmp_scripts --exclude=./.swp \
    --exclude=./vim -czf "$TGZ" -C "$REPO" .
ls -lh "$TGZ" | awk '{print "   "$5}'

scp -q "$TGZ" "$HOST:/tmp/ab_tree.tgz"
ssh "$HOST" "
set -e
sudo docker exec $CTR bash -lc 'rm -rf $DEST && mkdir -p $DEST'
sudo docker cp /tmp/ab_tree.tgz $CTR:/tmp/ab_tree.tgz
sudo docker exec $CTR bash -lc 'tar -xzf /tmp/ab_tree.tgz -C $DEST'
# catlass is a submodule we do not have locally and the build cannot git-fetch it
# from an unpacked tarball; borrow the shared checkout's copy (read-only on their side)
sudo docker exec $CTR bash -lc '
    src=/vllm-workspace/vllm-ascend/csrc/third_party/catlass
    [ -d "\$src" ] || { echo "!! no catlass at \$src"; exit 1; }
    mkdir -p $DEST/csrc/third_party
    cp -a "\$src" $DEST/csrc/third_party/
    ls $DEST/csrc/third_party/catlass | wc -l | xargs echo "   catlass entries:"'
sudo docker exec -e SOC_VERSION='$SOC' -e MAX_JOBS='$JOBS' $CTR bash -lc '
    set -o pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export COMPILE_CUSTOM_KERNELS=1
    cd $DEST
    echo \"=== build_ext (SOC=\$SOC_VERSION jobs=\$MAX_JOBS) ===\"
    python3 setup.py build_ext --inplace > /tmp/ab_build.log 2>&1
    rc=\$?
    echo \"rc=\$rc\"
    exit \$rc
' || {
    echo '!! BUILD FAILED -- errors below'
    sudo docker exec $CTR bash -lc 'grep -nE \"error:|Error [0-9]|undefined reference|No such file\" /tmp/ab_build.log | head -40; echo ...; tail -25 /tmp/ab_build.log'
    exit 1
}
echo '=== OK. our files in the log: ==='
sudo docker exec $CTR bash -lc 'grep -E \"torch_binding|add_lora_swiglu_quant|torch_binding_meta\" /tmp/ab_build.log | tail -10; ls -lh $DEST/vllm_ascend/*.so 2>/dev/null | head'
"
