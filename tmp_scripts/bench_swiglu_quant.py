"""Is the fused w13 epilogue actually bandwidth-bound?

The fused op demonstrably fires (profiler shows one add_lora_swiglu_quant
instead of add -> swiglu -> quant) yet costs the same wall time. Either the
traffic saving is smaller than claimed, or the kernel is not bandwidth-bound
so traffic does not set the runtime.

This measures both:
  * a copy roofline on the same device (what HBM-bound code achieves here)
  * the baseline chain:  gate_up + delta -> npu_swiglu -> npu_dynamic_quant
  * the fused kernel

and prints achieved GB/s for each against its own byte count. If the fused
kernel lands far below the roofline while the baseline ops sit near it, the
bottleneck is sync/latency inside the kernel, not HBM traffic.

Byte counts (E = T*W elements, 2 B for bf16):
  baseline add    : read 4E + read 4E + write 4E = 12E
  baseline swiglu : read 4E + write 2E           =  6E
  baseline quant  : read 2E + write 1E           =  3E   -> 21E total
  fused           : read 4E + 4E, write 2E + 1E  = 11E
"""

import time

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op(), "custom ops disabled in this build"

DEV = "npu"
DTYPE = torch.bfloat16
REPEAT = 50
WARMUP = 10


def timeit(fn, repeat=REPEAT, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6  # us


def roofline():
    """Large contiguous copy: read N + write N, as close to pure HBM as we get."""
    n = 256 * 1024 * 1024 // 2  # 256 MB of bf16
    src = torch.randn(n, device=DEV, dtype=DTYPE)
    dst = torch.empty_like(src)
    t = timeit(lambda: dst.copy_(src), repeat=20, warmup=5)
    gbs = (2 * n * 2) / (t * 1e-6) / 1e9
    print(f"copy roofline: {t:.1f} us for 256MB  ->  {gbs:.0f} GB/s (read+write)\n")
    return gbs


def main():
    if not hasattr(torch.ops._C_ascend, "add_lora_swiglu_quant"):
        raise SystemExit("op missing -- kernels not rebuilt")

    peak = roofline()

    hdr = (f"{'T':>7} {'W':>6} | {'add':>8} {'swiglu':>8} {'quant':>8} {'base':>8} "
           f"{'fused':>8} | {'speedup':>7} | {'base GB/s':>9} {'fus GB/s':>9} {'%peak':>6}")
    print(hdr)
    print("-" * len(hdr))

    for T in (512, 2048, 8192, 32768):
        for W in (1408, 2048, 4096):
            gate_up = torch.randn(T, 2 * W, device=DEV, dtype=DTYPE)
            delta = torch.randn(T, 2 * W, device=DEV, dtype=DTYPE) * 0.1

            # correctness vs the chain it replaces
            act_f, y_f, s_f = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta)
            summed = gate_up + delta
            act_e = torch_npu.npu_swiglu(summed, dim=-1)
            y_e, s_e = torch_npu.npu_dynamic_quant(act_e)
            act_err = (act_f.float() - act_e.float()).abs().max().item()
            y_err = (y_f.int() - y_e.int()).abs().max().item()

            def base():
                s = gate_up + delta
                a = torch_npu.npu_swiglu(s, dim=-1)
                return torch_npu.npu_dynamic_quant(a)

            t_add = timeit(lambda: gate_up + delta)
            t_swiglu = timeit(lambda: torch_npu.npu_swiglu(summed, dim=-1))
            t_quant = timeit(lambda: torch_npu.npu_dynamic_quant(act_e))
            t_base = timeit(base)
            t_fused = timeit(
                lambda: torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta))

            E = T * W
            gbs_base = (21 * E) / (t_base * 1e-6) / 1e9
            gbs_fused = (11 * E) / (t_fused * 1e-6) / 1e9

            print(f"{T:>7} {W:>6} | {t_add:>8.1f} {t_swiglu:>8.1f} {t_quant:>8.1f} "
                  f"{t_base:>8.1f} {t_fused:>8.1f} | {t_base / t_fused:>7.2f} | "
                  f"{gbs_base:>9.0f} {gbs_fused:>9.0f} {100 * gbs_fused / peak:>5.0f}%"
                  f"   err act={act_err:.3g} y={y_err}")

    print()
    print("us/call. 'base' = add+swiglu+quant chained (what the profiler shows).")
    print("If fused %peak is far below base GB/s vs peak, the kernel is sync-bound:")
    print("cutting traffic 21E->11E cannot help until the per-token stalls go.")


if __name__ == "__main__":
    main()
