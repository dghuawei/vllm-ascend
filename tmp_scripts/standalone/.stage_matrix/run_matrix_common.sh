# Sourced by the per-variant runners. Expects WORK, SOC_VERSION, ASCEND_HOME_PATH.

build_variant() {  # name srcfile
    local name=$1 src=$2
    local V=$WORK/v_$name
    rm -rf "$V"; mkdir -p "$V/build"
    cp "$WORK/types.h" "$WORK/main.cpp" "$WORK/CMakeLists.txt" "$V/"
    cp "$WORK/$src" "$V/add_lora_swiglu_quant.cpp"
    ( cd "$V/build" \
      && cmake .. -DSOC_VERSION="$SOC_VERSION" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" >cmake.log 2>&1 \
      && make -j16 >make.log 2>&1 ) \
      || { echo "!! BUILD FAILED $name"; tail -30 "$V/build/make.log" 2>/dev/null \
           || tail -20 "$V/build/cmake.log"; return 1; }
    echo "   built $name"
}

# bench prints bare result lines; the tabulator keys off '## variant=' headers,
# so synthesise one in front of each.
tag_results() {  # variant
    awk -v v="$1" '/^T=[0-9]/{print "## variant=" v " " $1 " " $2 " " $3} {print}'
}

# Pass 1: wall clock over the whole matrix in ONE process. Reports host submit
# cost beside wall time, so host-bound shapes announce themselves.
run_wall() {  # name
    local name=$1 V=$WORK/v_$1
    [ -x "$V/build/bench" ] || { echo "!! no binary for $name"; return 1; }
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    emit_shapes wall > "$WORK/shapes_wall.txt"
    echo "--- wall clock, $(wc -l < "$WORK/shapes_wall.txt") shapes ---"
    ( cd "$V/build" && ./bench --matrix "$WORK/shapes_wall.txt" 2>&1 ) | tag_results "$name"
}

# Pass 2: the real number. One msprof run over the whole matrix; per-shape
# device time comes from hardware Task Duration, not from the launch loop.
run_device() {  # name
    local name=$1 V=$WORK/v_$1
    [ -x "$V/build/bench" ] || return 1
    export LD_LIBRARY_PATH="$V/build/lib:$V/build:$LD_LIBRARY_PATH_BASE"
    emit_shapes prof > "$WORK/shapes_prof.txt"
    local OUT=/tmp/prof_matrix_$name
    rm -rf "$OUT" && mkdir -p "$OUT"
    echo "--- msprof device time, $(wc -l < "$WORK/shapes_prof.txt") shapes ---"
    ( cd "$V/build" && msprof --output="$OUT" \
        --application="$V/build/bench --matrix $WORK/shapes_prof.txt" \
        --aic-metrics=PipeUtilization --ai-core=on --task-time=on --ascendcl=on ) \
        >"$OUT/msprof.log" 2>&1

    local APP="$OUT/app.log"
    grep -E '^MANIFEST ' "$OUT/msprof.log" > "$APP" 2>/dev/null || true
    if ! grep -q MANIFEST "$APP"; then
        # msprof swallowed the application's stdout. The manifest is a pure
        # function of the shapes file, so regenerate it with a plain run.
        echo "   (msprof hid app stdout; regenerating manifest unprofiled)"
        ( cd "$V/build" && ./bench --matrix "$WORK/shapes_prof.txt" 2>/dev/null ) \
            | grep -E '^MANIFEST ' > "$APP"
    fi
    if ! grep -q MANIFEST "$APP"; then
        echo "   !! no MANIFEST; see $OUT/msprof.log"; tail -15 "$OUT/msprof.log"; return 1
    fi

    local SUM
    SUM=$(find "$OUT" -name 'op_summary*.csv' 2>/dev/null | head -1)
    [ -z "$SUM" ] && { echo "   !! no op_summary under $OUT"; tail -15 "$OUT/msprof.log"; return 1; }
    cp "$SUM" "$OUT/op_summary_used.csv"
    python3 "$WORK/device_time.py" --summary "$SUM" --log "$APP" \
            --variant "${name}_dev" --filter swiglu
}
