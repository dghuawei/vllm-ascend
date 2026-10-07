#!/bin/bash
# Launch the six Qwen workers, one per arena agent, each with its own
# CLAUDE_CONFIG_DIR (write-fenced to its own directory by
# 2026-10-07_arena_qwen_configs.sh) and its own NPU on bz-ascend.
#
#   bash tmp_scripts/2026-10-07_arena_spawn.sh [agent ...]
#
# With no arguments it launches all six. Output goes to
# <agentdir>/worker_stdout.txt; the real deliverable is <agentdir>/LOG.md.
set -u
ARENA=/Users/arabel1a/work/huawei/qlora/kernel_arena
AGENTS=${*:-"a1 a2 b1 b2 c1 c2"}

kernel_of() { case $1 in a?) echo z1z2 ;; b?) echo swiglu ;; c?) echo scatter ;; esac; }
dev_of()    { case $1 in a1) echo 0;; a2) echo 1;; b1) echo 2;; b2) echo 3;; c1) echo 4;; c2) echo 5;; esac; }
file_of()   { case $1 in
                a?) echo "kernel/add_lora_fused.cpp" ;;
                b?) echo "kernel/add_lora_swiglu_quant.cpp" ;;
                c?) echo "kernel/scatter_nd_update_v2/ (mainly op_kernel/scatter_nd_update_row_split.h)" ;;
              esac; }

launch() {
    local id=$1 k dev dir
    k=$(kernel_of "$id"); dev=$(dev_of "$id"); dir="$ARENA/$k/agents/$id"
    local cfg="$HOME/.claude-qwen-$id"
    [ -d "$cfg" ] || { echo "!! no config dir $cfg -- run 2026-10-07_arena_qwen_configs.sh first"; return 1; }
    [ -f "$dir/AGENT.md" ] || { echo "!! no $dir/AGENT.md"; return 1; }

    local pf="$dir/worker_prompt.txt"
    cat > "$pf" <<EOF
You are a performance engineer optimising a single AscendC kernel for a Huawei
Ascend 910B4 NPU. You are agent "$id". Your NPU is device $dev. Your directory
is $dir.

READ THESE THREE FILES IN FULL, FIRST, BEFORE ANYTHING ELSE:
  1. $dir/AGENT.md        -- your brief: what the kernel computes, the
                             production shapes, the MEASURED baseline with pipe
                             utilisation ratios, the leads worth checking, the
                             constraints, and the method you must follow.
  2. $ARENA/README.md     -- the arena rules.
  3. $ARENA/SHAPES.md     -- where every production shape came from.
Then read the kernel source you are optimising: $dir/$(file_of "$id")

THE JOB
Make that kernel as fast as possible on the production shapes without breaking
correctness. You may edit ONLY files under $dir/ . Never edit anything under
any harness/ or baseline/ directory, never edit another agent's directory, and
never edit run.sh, README.md or SHAPES.md. The measurement driver verifies the
harness by sha256 and will refuse to run if it was touched.

HOW TO MEASURE -- exactly one command, from anywhere:
    bash $ARENA/run.sh $id
    bash $ARENA/run.sh $id --quick      (fewer iterations, for a smoke test)
    bash $ARENA/run.sh $id --baseline   (the pristine kernel, for reference)
It builds your kernel on the remote machine, runs it on device $dev only, and
prints the results. Do not ssh anywhere yourself. Do not set
ASCEND_RT_VISIBLE_DEVICES. Never touch NPU 6 or 7.

A build-and-measure cycle takes a couple of minutes, so think before you
compile rather than trying things at random. Run these commands in the
FOREGROUND. Never run run.sh with run_in_background, and never end your turn
while a background task is still pending -- this session's process dies when
you stop, and any backgrounded result dies with it.

WHICH NUMBER COUNTS
The "CASE," lines are wall-clock and include host launch overhead, which
dominates the small cases and is meaningless there. The "DEV," lines are
msprof device time: <kernel>=<aiv_us>(dur=<task_us>)[vec:scalar:mte2:mte3].
aiv_us is the metric. The bracket is the fraction of kernel time spent in each
pipe -- READ IT BEFORE AND AFTER EVERY CHANGE. If the kernel is MTE2 bound,
removing arithmetic cannot help it; if it is scalar bound, moving fewer bytes
cannot help it. Your log must say which pipe each experiment targets.

Any case printing FAIL means the experiment failed, however fast it was. The
correctness gates encode what the kernel is contracted to compute. You cannot
relax them and must not try to work around them.

METHOD -- this is not optional, it is half of what is being asked for
Work scientifically and leave a record someone else can audit:
  * Start with a baseline run and paste it into $dir/LOG.md.
  * Write down, in LOG.md, what you think the kernel is bound by at each
    production size, what that rules OUT, and your candidate changes in order,
    each tied to a pipe ratio rather than to general advice.
  * ONE change per experiment. Two changes at once teaches you nothing.
  * WRITE THE PREDICTION INTO LOG.md AND SAVE THE FILE BEFORE YOU RUN: which
    cases will move, in which direction, roughly how much, and what you would
    see if your hypothesis is wrong. "It will be faster" is not a prediction.
    Never edit a prediction after seeing the result -- that would make the
    whole exercise worthless.
  * Record refutations. A change that did nothing, or made things worse, is a
    result worth keeping, with your reading of why. Most of the value in your
    log will be in what did not work.
  * Revert anything you do not keep, and say in the log what you reverted to.
  * If you cannot explain a measurement, write INCONCLUSIVE. Do not invent a
    mechanism to fit a number.
$dir/LOG.md already contains the structure to follow. Keep it append-only
except for the final Summary section.

SCOPE AND STOPPING
Aim for at least six logged experiments, and keep going while you still have
ideas with a mechanism behind them. Do NOT stop after one or two. Do not ask
questions -- nobody is available to answer, you have everything you need, so
make the call, write down the reasoning, and continue. If an idea turns out to
be impossible, record that in LOG.md and move to the next one. If LOG.md
already contains EXP entries, earlier sessions of yours did that work: read
the log and 'git log --oneline' first, continue the EXP numbering, and do not
redo finished experiments or re-run a baseline the log already has.

COMPUTING IS BOOKED FOR YOU THROUGH THE NIGHT, AND A FRESH SESSION RESUMES
automatically whenever this one ends. An early "I'm done" therefore does not
finish the job -- it just burns tonight's compute. When you feel done, that is
the moment to force yourself to write three more candidate ideas into LOG.md,
ranked by mechanism, and start the most promising one. Never write "nothing
left to try"; write "open leads" and say what the next session should attempt
first. Six experiments is the floor, not the goal.

When you do reach a natural stopping point, fill in the Summary section of
LOG.md: your best configuration, the per-case device-time table against the
baseline, correctness status, what you learned, what failed and why, and what
you would try next. Then stop.

GIT -- COMMIT AT MEANINGFUL MILESTONES
$dir is its own git repository (only your directory is tracked). Commits are
how the BEST result gets picked later, not just whatever the last edit
happened to be:
  * Commit after every experiment you KEEP, before starting any risky
    refactor, and once more before your session ends:
      cd $dir && git add -A && git commit -m "$id: EXP-<n> <what changed + measured effect>"
  * When the current tree is your best measured configuration, tag it:
      git tag ${id}-best-EXP<n>
  * Never amend, rebase or reset; never delete commits or tags. Committing a
    broken in-flight state is fine -- history is the deliverable.

REFERENCE MATERIAL
$ARENA/docs/ascendc/ is the CANN 9.0.0 Ascend C API manual as markdown (~550
files; 0001_目_录.md is the contents). Look up the exact semantics of an
intrinsic there rather than guessing at a signature, and check the 产品支持情况
table at the top of an API -- this chip is Atlas A2 and several APIs are not
available on it. $ARENA/docs/notes/ holds distilled notes from earlier work on
these same kernels; read ascendc-vector-geometry.md before reasoning about any
repeat or stride value.
EOF

    echo "== launching $id (kernel=$k npu=$dev)"
    (
      CLAUDE_CONFIG_DIR="$cfg" \
      ANTHROPIC_BASE_URL=http://127.0.0.1:18010 ANTHROPIC_AUTH_TOKEN=none \
      ANTHROPIC_MODEL=qwen3.8-flash-next ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-flash-next \
      ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash-next \
      CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 CLAUDE_CODE_EFFORT_LEVEL=xhigh \
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 API_TIMEOUT_MS=600000 \
      cd "$dir" && claude -p --dangerously-skip-permissions "$(cat "$pf")" \
        > "$dir/worker_stdout.txt" 2>&1
      echo "== $id finished rc=$? -> $dir/worker_stdout.txt"
    ) &
}

for a in $AGENTS; do launch "$a"; done
echo "-- all launched; waiting --"
wait
echo "== all workers done =="
