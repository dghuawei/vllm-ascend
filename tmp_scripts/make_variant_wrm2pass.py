#!/usr/bin/env python3
"""Generate a variant that does the WHOLE reduction with WholeReduceMax,
instead of per-row ReduceMax followed by a compaction pass.

  pass 1 (per row): W/64 partials, written at a padded pitch P = ceil8(W/64)
                    so pass 2's srcRepStride is a whole number of datablocks
  pass 2 (one call): reduce each row's P-slot group to one value, contiguous

Requires W % 64 == 0 and W/64 <= 64 (the fp32 mask ceiling), i.e. W <= 4096.

maxBuffer_ must hold numRows*pitch floats, not numRows*8. Bound:
  sum_rows pitch <= numRows*(W/64 + 7) = TILE_ELEMENTS/64 + 7*numRows
                 <= 128 + 7*16 = 240
so the variant widens maxBuffer_ to 256 floats (+512 B UB vs the base).
W=768 needs 160 and overflows the base 128 -- this is why.

Located by markers, not exact text, so incidental whitespace edits to the
base kernel do not break it. Asserts the region looks like what we expect.

usage: make_variant_wrm2pass.py <base.cpp> <out.cpp>
"""
import sys

NEW = """        AscendC::LocalTensor<float> rowMax = rowMaxBuffer_.Get<float>();
        uint32_t partialsPerRow = width_ / FP32_PER_REPEAT;
        uint32_t pitch = (partialsPerRow + UB_BLOCK_FLOATS - 1) / UB_BLOCK_FLOATS * UB_BLOCK_FLOATS;
        for (uint32_t i = 0; i < numRows; i++) {
            AscendC::WholeReduceMax<float>(
                maxs[i * pitch], gate[i * width_], (int32_t)FP32_PER_REPEAT,
                (int32_t)partialsPerRow, 1, 1, 8,
                AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<float>(
            rowMax, maxs, (int32_t)partialsPerRow, (int32_t)numRows, 1, 1,
            (int32_t)(pitch / UB_BLOCK_FLOATS),
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
"""


def main():
    base, out = sys.argv[1], sys.argv[2]
    lines = open(base).read().split("\n")

    start = next((i for i, l in enumerate(lines) if "AscendC::ReduceMax<float>" in l), None)
    if start is None:
        raise SystemExit("no ReduceMax call found; base kernel drifted")
    # walk back to the opening 'for'
    while start > 0 and "for (uint32_t i = 0" not in lines[start]:
        start -= 1
    end = next((i for i, l in enumerate(lines)
                if i > start and "ORDER_ONLY_VALUE" in l), None)
    if end is None:
        raise SystemExit("no compaction WholeReduceMax found; base kernel drifted")
    while end < len(lines) and "PipeBarrier" not in lines[end]:
        end += 1

    region = "\n".join(lines[start:end + 1])
    for token in ("ReduceMax<float>", "WholeReduceMax<float>", "rowMaxBuffer_"):
        if token not in region:
            raise SystemExit(f"region missing {token}; refusing to patch")

    out_lines = lines[:start] + [NEW.rstrip("\n")] + lines[end + 1:]

    buf_old = "pipe_->InitBuffer(maxBuffer_, MAX_TOKENS_PER_TILE * UB_BLOCK_FLOATS * sizeof(float));"
    buf_new = "pipe_->InitBuffer(maxBuffer_, 2 * MAX_TOKENS_PER_TILE * UB_BLOCK_FLOATS * sizeof(float));"
    hits = [i for i, l in enumerate(out_lines) if buf_old in l]
    if len(hits) != 1:
        raise SystemExit(f"expected 1 maxBuffer_ InitBuffer, found {len(hits)}")
    out_lines[hits[0]] = out_lines[hits[0]].replace(buf_old, buf_new)

    open(out, "w").write("\n".join(out_lines))
    print(f"patched lines {start + 1}..{end + 1}, widened maxBuffer_ to 256 floats")


main()
