#!/bin/bash
# Stage and run the GroupedMatmulV4 strided-output probe. Builds in /tmp only.
set -euo pipefail
HOST=${HOST:-researchagent-node1}
CTR=${CTR:-va25_ds}
SRC=/Users/arabel1a/work/huawei/qlora/vllm-ascend/tmp_scripts/probe_b
STAGE=/tmp/probe_b

tar -czf /tmp/probe_b.tgz -C "$SRC" .
scp -q /tmp/probe_b.tgz "$HOST:/tmp/probe_b.tgz"
ssh "$HOST" "
set -e
rm -rf $STAGE && mkdir -p $STAGE
tar -xzf /tmp/probe_b.tgz -C $STAGE
sudo docker exec $CTR rm -rf $STAGE
sudo docker cp $STAGE $CTR:$STAGE
sudo docker exec $CTR bash -lc '
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $STAGE && TORCH_EXTENSIONS_DIR=$STAGE/build python3 run_probe.py 2>&1 | tail -40
'
"
