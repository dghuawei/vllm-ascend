#!/usr/bin/env python3
"""Generate experiment variants of add_lora_swiglu_quant.cpp.

Every transform is an exact string replacement and asserts it matched, so if
the base kernel drifts this fails loudly instead of silently producing a
different experiment.

  A  -- add the delta in bf16 over the whole tile, before casting to fp32.
        Removes 2 casts, one Add and an entire per-row loop. Also matches the
        eager torch reference, which rounds the sum to bf16.
  B  -- A, plus run the whole working chain in half instead of float.
        Halves vector density and the fp32 working set; drops the final
        fp32->half cast entirely.

usage: make_variant.py <base.cpp> <out.cpp> <A|B> [buffer_num] [tile_elements]
"""
import sys

def sub(s, old, new, what):
    if old not in s:
        raise SystemExit(f"transform '{what}' did not match; base kernel drifted")
    return s.replace(old, new, 1)


DEINTERLEAVE_AND_ADD = """        AscendC::LocalTensor<scalar_t> gateUpLocal = inQueueGateUp_.DeQue<scalar_t>();
        for (uint32_t i = 0; i < numRows; i++) {
            Cast(gate[i * width_], gateUpLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            Cast(up[i * width_], gateUpLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE, width_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inQueueGateUp_.FreeTensor(gateUpLocal);

        if (hasDelta_ != 0) {
            AscendC::LocalTensor<scalar_t> deltaLocal = inQueueDelta_.DeQue<scalar_t>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(gate, gate, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE,
                     width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(up, up, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueDelta_.FreeTensor(deltaLocal);
        }
"""

BF16_ADD = """        AscendC::LocalTensor<scalar_t> gateUpLocal = inQueueGateUp_.DeQue<scalar_t>();
        if (hasDelta_ != 0) {
            AscendC::LocalTensor<scalar_t> deltaLocal = inQueueDelta_.DeQue<scalar_t>();
            Add(gateUpLocal, gateUpLocal, deltaLocal, numRows * 2 * width_);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueDelta_.FreeTensor(deltaLocal);
        }
        for (uint32_t i = 0; i < numRows; i++) {
            Cast(gate[i * width_], gateUpLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            Cast(up[i * width_], gateUpLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE, width_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inQueueGateUp_.FreeTensor(gateUpLocal);
"""

QUANT_TAIL_F32 = """        AscendC::LocalTensor<half> halfLocal = up.ReinterpretCast<half>();
        Cast(halfLocal, activated, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY_.AllocTensor<int8_t>();
        Cast(yLocal, halfLocal, AscendC::RoundMode::CAST_RINT, numElements);
"""

QUANT_TAIL_HALF = """        AscendC::LocalTensor<int8_t> yLocal = outQueueY_.AllocTensor<int8_t>();
        Cast(yLocal, activated, AscendC::RoundMode::CAST_RINT, numElements);
"""


ROWS_FROM_TILE = """        numRowsPerTile_ = TILE_ELEMENTS / width_;
"""

ROWS_FROM_BUDGET = """        uint32_t bytesPerColumn =
            BUFFER_NUM * (3 * sizeof(scalar_t) + sizeof(int8_t) +
                          (hasDelta_ != 0 ? 2 * sizeof(scalar_t) : 0)) +
            3 * sizeof(float);
        numRowsPerTile_ = UB_TILE_BUDGET_BYTES / (width_ * bytesPerColumn);
"""


def main():
    base, out, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    bn = sys.argv[4] if len(sys.argv) > 4 else None
    tile = sys.argv[5] if len(sys.argv) > 5 else None
    s = open(base).read()

    if variant == "BUDGET":
        # derive rows per tile from the measured UB ceiling instead of a fixed
        # element count, so every width AND the no-delta path use the full UB
        s = sub(s, "constexpr uint32_t TILE_ELEMENTS = 5120;",
                "constexpr uint32_t UB_TILE_BUDGET_BYTES = 184 * 1024;", "budget constant")
        s = sub(s, ROWS_FROM_TILE, ROWS_FROM_BUDGET, "rows from budget")

    if variant in ("A", "B"):
        s = sub(s, DEINTERLEAVE_AND_ADD, BF16_ADD, "bf16 delta add")

    if variant == "B":
        # working buffers: float -> half
        for name in ("gateBuffer_", "upBuffer_", "actBuffer_"):
            s = sub(s, f"pipe_->InitBuffer({name}, tileElements * sizeof(float));",
                    f"pipe_->InitBuffer({name}, tileElements * sizeof(half));", f"{name} half")
        s = sub(s, "pipe_->InitBuffer(maxBuffer_, MAX_TOKENS_PER_TILE * UB_BLOCK_FLOATS * sizeof(float));",
                "pipe_->InitBuffer(maxBuffer_, MAX_TOKENS_PER_TILE * UB_BLOCK_HALFS * sizeof(half));",
                "maxBuffer half")
        s = sub(s, "constexpr uint32_t UB_BLOCK_FLOATS = 8;",
                "constexpr uint32_t UB_BLOCK_FLOATS = 8;\nconstexpr uint32_t UB_BLOCK_HALFS = 16;",
                "UB_BLOCK_HALFS")
        for name in ("gate", "up", "activated"):
            s = sub(s, f"AscendC::LocalTensor<float> {name} = ",
                    f"AscendC::LocalTensor<half> {name} = ", f"{name} tensor half")
        s = sub(s, "AscendC::LocalTensor<float> maxs = maxBuffer_.Get<float>();",
                "AscendC::LocalTensor<half> maxs = maxBuffer_.Get<half>();", "maxs half")
        for name in ("gateBuffer_", "upBuffer_", "actBuffer_"):
            s = sub(s, f"{name}.Get<float>()", f"{name}.Get<half>()", f"{name} Get half")
        s = sub(s, "AscendC::SwiGLU<float, false>", "AscendC::SwiGLU<half, false>", "SwiGLU half")
        s = sub(s, """            AscendC::ReduceMax<float>(maxs[i * UB_BLOCK_FLOATS], gate[i * width_], gate[i * width_],
                                      width_);""",
                """            AscendC::ReduceMax<half>(maxs[i * UB_BLOCK_HALFS], gate[i * width_], gate[i * width_],
                                     width_);""", "ReduceMax half")
        s = sub(s, "scaleValues_[i] = maxs.GetValue(i * UB_BLOCK_FLOATS) / INT8_MAX_VALUE;",
                "scaleValues_[i] = (float)maxs.GetValue(i * UB_BLOCK_HALFS) / INT8_MAX_VALUE;",
                "scale from half")
        s = sub(s, "Muls(activated[i * width_], activated[i * width_], reciprocal, width_);",
                "Muls(activated[i * width_], activated[i * width_], (half)reciprocal, width_);",
                "Muls half scalar")
        s = sub(s, QUANT_TAIL_F32, QUANT_TAIL_HALF, "quant tail half")

    if bn:
        import re
        s = re.sub(r"^constexpr int32_t BUFFER_NUM = .*$",
                   f"constexpr int32_t BUFFER_NUM = {bn};", s, flags=re.M)
    if tile:
        import re
        s = re.sub(r"^constexpr uint32_t TILE_ELEMENTS = .*$",
                   f"constexpr uint32_t TILE_ELEMENTS = {tile};", s, flags=re.M)

    open(out, "w").write(s)


main()
