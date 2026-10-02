#!/usr/bin/env python3
"""Script 3 of 3 -- UNFUSED: the eager op chain the fused kernel replaces.

  delta=1 (prefill):  gate_up + delta -> npu_swiglu -> npu_dynamic_quant
  delta=0 (decode):                      npu_swiglu -> npu_dynamic_quant

Imports only torch + torch_npu, never vllm_ascend, so it needs no custom-op
build and cannot disturb a checkout.

Two modes, because wall clock and kernel time are different questions:

  wall   python-side latency of the eager chain. At small T this is almost
         entirely dispatcher overhead and says nothing about the kernels.
  prof   run under msprof with a small iteration count, bracketing each
         measured window with a marker op. device_time.py then sums the
         hardware Task Duration of the ops inside the window, which is the
         dense per-call device cost vLLM sees in compiled mode.

Byte counts, E = T*W output elements, bf16 = 2 B:
  add 12E + swiglu 6E + quant 3E = 21E with delta, 9E without.
"""
import sys
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"
DTYPE = torch.bfloat16

DECODE_T = [8, 16, 32, 63, 128]
PREFILL_T = [2048, 4096, 16384, 32768]
WIDTHS = [768, 1024, 1408, 2048]
WARMUP = 10


def iters_for(t):
    return 2000 if t <= 128 else (200 if t <= 4096 else 50)


def prof_iters_for(t):
    return 50 if t <= 128 else (30 if t <= 4096 else 20)


def shapes():
    return ([(t, w, 0) for w in WIDTHS for t in DECODE_T]
            + [(t, w, 1) for w in WIDTHS for t in PREFILL_T])


def make_chain(gate_up, delta, has_delta):
    if has_delta:
        def chain():
            s = gate_up + delta
            a = torch_npu.npu_swiglu(s, dim=-1)
            return torch_npu.npu_dynamic_quant(a)
    else:
        def chain():
            a = torch_npu.npu_swiglu(gate_up, dim=-1)
            return torch_npu.npu_dynamic_quant(a)
    return chain


def timeit(fn, repeat, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    t_submit = time.perf_counter()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeat * 1e6, (t_submit - t0) / repeat * 1e6


def run_wall():
    for T, W, has_delta in shapes():
        it = iters_for(T)
        gate_up = torch.randn(T, 2 * W, device=DEV, dtype=DTYPE)
        delta = torch.randn(T, 2 * W, device=DEV, dtype=DTYPE) * 0.1 if has_delta else None
        chain = make_chain(gate_up, delta, has_delta)

        us, submit_us = timeit(chain, it)

        summed = (gate_up + delta) if has_delta else gate_up
        act = torch_npu.npu_swiglu(summed, dim=-1)
        t_add = timeit(lambda: gate_up + delta, it)[0] if has_delta else 0.0
        t_swiglu = timeit(lambda: torch_npu.npu_swiglu(summed, dim=-1), it)[0]
        t_quant = timeit(lambda: torch_npu.npu_dynamic_quant(act), it)[0]

        E = T * W
        gbs = (E * (21.0 if has_delta else 9.0) + T * 4.0) / (us * 1e-6) / 1e9
        print(f"## variant=unfused T={T} W={W} delta={has_delta} iters={it}")
        print(f"T={T} W={W} delta={has_delta} aiv=40 | {us:.2f} us | {gbs:.1f} GB/s | "
              f"submit {submit_us:.2f} us"
              + ("  <<< HOST BOUND, wall time is the launcher" if submit_us > 0.9 * us else ""))
        print(f"   breakdown: add={t_add:.2f} swiglu={t_swiglu:.2f} quant={t_quant:.2f} "
              f"sum={t_add + t_swiglu + t_quant:.2f}", flush=True)

        del gate_up, delta, summed, act
        torch.npu.empty_cache()


def run_prof():
    marker_src = torch.ones(4, device=DEV, dtype=torch.float32)

    def marker():
        torch.cumsum(marker_src, 0)

    for T, W, has_delta in shapes():
        it = prof_iters_for(T)
        ops_per_call = 3 if has_delta else 2
        gate_up = torch.randn(T, 2 * W, device=DEV, dtype=DTYPE)
        delta = torch.randn(T, 2 * W, device=DEV, dtype=DTYPE) * 0.1 if has_delta else None
        chain = make_chain(gate_up, delta, has_delta)

        for _ in range(WARMUP):
            chain()
        torch.npu.synchronize()

        marker()
        for _ in range(it):
            chain()
        marker()
        torch.npu.synchronize()

        print(f"MANIFEST T={T} W={W} delta={has_delta} iters={it} "
              f"launches={ops_per_call * it} measured_from=0", flush=True)

        del gate_up, delta
        torch.npu.empty_cache()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "wall"
    torch.npu.set_device(0)
    if mode == "wall":
        run_wall()
    elif mode == "prof":
        run_prof()
    else:
        raise SystemExit("mode must be wall or prof")


if __name__ == "__main__":
    main()
