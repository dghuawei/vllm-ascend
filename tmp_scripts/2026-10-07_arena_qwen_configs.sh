#!/bin/bash
# Build one CLAUDE_CONFIG_DIR per arena agent, so each Qwen worker can only
# WRITE inside its own agent directory. The vault denies are inherited from
# ~/.claude-qwen. Deny rules take precedence over bypass-permissions mode, so
# this is a real fence for Edit/Write, not just advice.
#
# Only Edit(path) rules are honoured by the file-permission checks -- a
# Write(path) rule is silently ignored and warned about. So every fence below
# is an Edit() rule, which covers Write, Edit and NotebookEdit alike.
#
# Deny must name the SIBLING directories explicitly: a blanket
# deny on agents/** would also cover the agent's own tree, and deny beats
# allow.
#
# Bash remains the escape hatch for a worker that actively tries to get out;
# that is caught after the fact instead -- the harness MANIFEST.sha256 check
# refuses to run against a modified harness, and every result row in
# results/*.csv carries the sha256 of the kernel source that produced it.
set -euo pipefail
ARENA=/Users/arabel1a/work/huawei/qlora/kernel_arena

ALL="a1:z1z2 a2:z1z2 b1:swiglu b2:swiglu c1:scatter c2:scatter"

make_cfg() {  # make_cfg <agent> <kernel>
    local id=$1 k=$2
    local cfg="$HOME/.claude-qwen-$id"
    mkdir -p "$cfg"
    : > "$cfg/CLAUDE.md"
    [ -f "$cfg/.claude.json" ] || cp "$HOME/.claude-qwen/.claude.json" "$cfg/.claude.json" 2>/dev/null || true

    # every other agent's directory, plus every harness and baseline
    local sib=""
    for pair in $ALL; do
        local oid=${pair%%:*} ok=${pair##*:}
        if [ "$oid" != "$id" ]; then
            sib="$sib      \"Edit(/$ARENA/$ok/agents/$oid/**)\",
"
        fi
    done
    for ok in z1z2 swiglu scatter; do
        sib="$sib      \"Edit(/$ARENA/$ok/harness/**)\",
      \"Edit(/$ARENA/$ok/baseline/**)\",
"
    done

    cat > "$cfg/settings.json" <<JSON
{
  "permissions": {
    "allow": [
      "Read",
      "Edit(/$ARENA/$k/agents/$id/**)"
    ],
    "deny": [
      "Read(//Users/arabel1a/agents/**)",
      "Edit(//Users/arabel1a/agents/**)",
      "Bash(*arabel1a/agents*)",
      "Bash(*~/agents*)",
      "Read(//Users/arabel1a/.claude/**)",
      "Edit(//Users/arabel1a/.claude/**)",
$sib      "Edit(/$ARENA/run.sh)",
      "Edit(/$ARENA/README.md)",
      "Edit(/$ARENA/SHAPES.md)",
      "Edit(/$ARENA/COMMON_BRIEF.md)",
      "Edit(/$ARENA/LOG_TEMPLATE.md)",
      "Edit(/$ARENA/docs/**)"
    ],
    "additionalDirectories": ["$ARENA"]
  },
  "autoMemoryEnabled": false,
  "skipDangerousModePermissionPrompt": true
}
JSON
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$cfg/settings.json" \
      || { echo "!! bad json for $id"; exit 1; }
    echo "  $cfg  (own: $k/agents/$id)"
}

for pair in $ALL; do
    make_cfg "${pair%%:*}" "${pair##*:}"
done
echo done
