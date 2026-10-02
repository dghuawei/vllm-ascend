#!/usr/bin/env python3
"""Third variant: TWO delta buffers (the new layout) but PER-ROW Casts (the old
call pattern). Splits the two-delta change into its two halves so we can see
which one moved the int8 LSB statistics at W%64 != 0.

  base        : one interleaved buffer + per-row Casts
  new_rowcast : two buffers            + per-row Casts   <- this file
  new         : two buffers            + one Cast per half

usage: make_rowcast_variant.py <new.cpp> <out new_rowcast.cpp>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

NEW = """            Cast(activated, deltaLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            Add(gate, gate, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(activated, deltaLocal[numElements], AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();"""

OLD = """            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * width_], AscendC::RoundMode::CAST_NONE,
                     width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(gate, gate, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[numElements + i * width_],
                     AscendC::RoundMode::CAST_NONE, width_);
            }
            AscendC::PipeBarrier<PIPE_V>();"""

if s.count(NEW) != 1:
    raise SystemExit(f"anchor found {s.count(NEW)} times, expected 1")
open(dst, "w").write(s.replace(NEW, OLD))
print(f"wrote {dst}")
