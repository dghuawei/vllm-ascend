#!/bin/bash
# Instantiate per-agent AGENT.md (kernel brief + common brief) and a fresh
# LOG.md from the template. Idempotent for AGENT.md; refuses to clobber a
# LOG.md that already has content beyond the template.
set -euo pipefail
ARENA=/Users/arabel1a/work/huawei/qlora/kernel_arena

emit() {  # emit <agent> <kernel> <dev>
    local id=$1 k=$2 dev=$3
    local dir="$ARENA/$k/agents/$id"
    mkdir -p "$dir"
    local rel="$k/agents/$id"
    {
        sed -e "s|@ID@|$id|g" -e "s|@DEV@|$dev|g" -e "s|@AGENTDIR@|$rel|g" \
            "$ARENA/$k/harness/AGENT_BRIEF.md"
        echo
        echo "---"
        echo
        sed -e "s|@ID@|$id|g" -e "s|@DEV@|$dev|g" -e "s|@AGENTDIR@|$rel|g" \
            "$ARENA/COMMON_BRIEF.md"
    } > "$dir/AGENT.md"

    if [ ! -f "$dir/LOG.md" ]; then
        sed -e "s|<ID>|$id|g" -e "s|<KERNEL>|$k|g" -e "s|<DEV>|$dev|g" \
            -e "s|run.sh <ID>|run.sh $id|g" \
            "$ARENA/LOG_TEMPLATE.md" > "$dir/LOG.md"
    else
        echo "   (LOG.md for $id exists, left alone)"
    fi
    echo "   $dir/AGENT.md  ($(wc -l < "$dir/AGENT.md") lines)"
}

emit a1 z1z2    0
emit a2 z1z2    1
emit b1 swiglu  2
emit b2 swiglu  3
emit c1 scatter 4
emit c2 scatter 5
echo "done"
