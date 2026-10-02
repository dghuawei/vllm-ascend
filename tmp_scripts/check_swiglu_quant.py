"""Correctness sweep for the tiled add_lora_swiglu_quant kernel.

The tiling (R rows per iteration, R = 5120 // W) introduced a remainder path
and moved the absmax/scale handling, so this hammers the cases that structure
can break: row counts that are not a multiple of R, fewer rows than cores,
the no-delta branch, all-zero rows (the divide-by-zero guard), and both
dtypes.

Reference is the eager chain (gate_up + delta -> npu_swiglu -> dynamic_quant).
The fused kernel is deliberately NOT bit-identical (it accumulates and runs
silu in fp32 where the eager chain rounds to the input dtype twice), so the
comparison is a tolerance on act and a +-1 level allowance on int8 with a cap
on how many elements may differ.
"""

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op(), "custom ops disabled in this build"

DEV = "npu"
FAILED = []


def check(name, T, W, dtype, delta=True, fill=None):
    if fill is None:
        gate_up = torch.randn(T, 2 * W, device=DEV, dtype=dtype)
        d = torch.randn(T, 2 * W, device=DEV, dtype=dtype) * 0.1 if delta else None
    else:
        gate_up = torch.full((T, 2 * W), fill, device=DEV, dtype=dtype)
        d = torch.zeros(T, 2 * W, device=DEV, dtype=dtype) if delta else None

    act_f, y_f, s_f = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, d)

    ref_in = gate_up if d is None else gate_up + d
    act_e = torch_npu.npu_swiglu(ref_in, dim=-1)
    y_e, s_e = torch_npu.npu_dynamic_quant(act_e)

    # act: relative tolerance against the fp32-vs-dtype rounding difference
    scale_ref = act_e.float().abs().max().clamp(min=1e-3)
    act_err = (act_f.float() - act_e.float()).abs().max().item() / scale_ref.item()
    # int8: allow +-1 level everywhere, and count worse offenders
    ydiff = (y_f.int() - y_e.int()).abs()
    y_max = ydiff.max().item()
    y_bad = (ydiff > 1).float().mean().item()
    s_err = ((s_f.float() - s_e.float()).abs()
             / s_e.float().clamp(min=1e-6)).max().item()

    ok = act_err < 0.02 and y_bad < 0.01 and s_err < 0.02
    # shapes/dtypes must be exact regardless
    ok = ok and act_f.shape == (T, W) and y_f.shape == (T, W) and s_f.shape == (T,)
    ok = ok and y_f.dtype == torch.int8 and s_f.dtype == torch.float32
    ok = ok and act_f.dtype == dtype

    status = "ok " if ok else "FAIL"
    if not ok:
        FAILED.append(name)
    print(f"{status} {name:<44} T={T:<6} W={W:<5} {str(dtype).split('.')[-1]:<9}"
          f" act_rel={act_err:.4g} y_max={y_max} y_bad={y_bad:.4g} s_rel={s_err:.3g}")


def main():
    for dtype in (torch.bfloat16, torch.float16):
        # R = 5120 // W, so pick row counts that straddle the tile boundary
        for W in (1408, 2048, 4096, 5120):
            R = max(1, 5120 // W)
            for T in (1, 2, R - 1 if R > 1 else 1, R, R + 1, 3 * R + 1, 997):
                if T < 1:
                    continue
                check(f"tile-remainder R={R}", T, W, dtype)

        check("no-delta", 4096, 2048, dtype, delta=False)
        check("no-delta odd rows", 997, 1408, dtype, delta=False)
        check("all-zero rows (div guard)", 64, 2048, dtype, fill=0.0)
        check("fewer rows than cores", 3, 4096, dtype)
        check("single row", 1, 5120, dtype)
        check("large", 32768, 2048, dtype)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        raise SystemExit(1)
    print("all correctness checks passed")


if __name__ == "__main__":
    main()
