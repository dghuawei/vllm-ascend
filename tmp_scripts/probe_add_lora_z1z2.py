"""Standalone probe: does torch.ops._C_ascend.add_lora actually launch the
fused z1/z2 kernels on THIS node, and is the result correct?

Bypasses vllm entirely (no server, no scheduler, no MoE), with a shape that is
unambiguously eligible by csrc/torch_binding.cpp:add_lora_eligible and far
under every decode size cap. So:

  * numbers correct AND z1/z2 in the profile -> the build is fine; the missing
    kernels in the server profile are a gating/size-cap issue upstream.
  * numbers correct but NO z1/z2 in the profile -> it silently fell back to
    bgmv (watch stderr for the "not eligible" warning) OR the profiler does
    not record raw <<<>>> launches on this node.
  * numbers WRONG / zero delta -> the raw kernel launch failed (SOC-version
    mismatch in the ascendc binary); the launch return code is not checked in
    add_lora_fused_impl, so that failure is otherwise invisible.

Run on both nodes and diff the output:
    python tmp_scripts/probe_add_lora_z1z2.py
    python tmp_scripts/probe_add_lora_z1z2.py --profile /tmp/z1z2_probe
"""

import argparse
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)


def environment_report() -> None:
    print("=== environment ===")
    print("torch            :", torch.__version__)
    print("soc_version (rt) :", torch_npu.npu.get_soc_version(), "(220-225=A2, 250-255=A3)")
    try:
        import vllm_ascend

        pkg_dir = os.path.dirname(vllm_ascend.__file__)
        print("vllm_ascend      :", pkg_dir)
        print("version          :", getattr(vllm_ascend, "__version__", "?"))
        from vllm_ascend import _build_info  # type: ignore

        for attr in ("__soc_version__", "__device_type__", "__version__", "__commit__"):
            if hasattr(_build_info, attr):
                print(f"_build_info{attr:<10}:", getattr(_build_info, attr))
    except Exception as e:  # pragma: no cover
        print("vllm_ascend import failed:", e)
        return

    # the custom op namespace is registered by the C extension, which
    # vllm_ascend imports lazily (utils.py:427) -- do it explicitly
    try:
        import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401
    except Exception as e:
        print("vllm_ascend_C import failed:", e)

    so = os.path.join(pkg_dir, "libvllm_ascend_kernels.so")
    print("kernels .so      :", so, "exists" if os.path.exists(so) else "MISSING")
    if os.path.exists(so):
        print("  size/mtime     :", os.path.getsize(so), os.path.getmtime(so))
        try:
            out = subprocess.run(
                ["nm", "-D", "--defined-only", so], capture_output=True, text=True, timeout=60
            ).stdout
            syms = sorted({ln.split()[-1] for ln in out.splitlines() if "add_lora_z" in ln})
            print("  z1/z2 symbols  :", syms or "NONE FOUND")
        except Exception as e:
            print("  nm failed      :", e)
    print("  add_lora op    :", hasattr(torch.ops._C_ascend, "add_lora"))


def build_case(device, dtype):
    """One eligible decode-shaped case: B=8, H1=128, R=16, two 64-wide slices."""
    torch.manual_seed(0)
    B, H1, R, max_loras = 8, 128, 16, 2
    slices = [64, 64]
    scale = 0.5

    x = torch.randn(B, H1, device=device, dtype=dtype)
    y = torch.randn(B, sum(slices), device=device, dtype=dtype)
    lora_a = [torch.randn(max_loras, 1, R, H1, device=device, dtype=dtype) * 0.05 for _ in slices]
    lora_b = [torch.randn(max_loras, 1, h2, R, device=device, dtype=dtype) * 0.05 for h2 in slices]
    # one row per slot plus a -1 (no-lora) row, which both paths must skip
    idx = torch.tensor([0, 1, -1, 0, 1, 0, 1, -1], device=device, dtype=torch.int64)
    return dict(
        x=x, y=y, lora_a=lora_a, lora_b=lora_b, idx=idx, slices=slices, scale=scale,
        B=B, R=R,
    )


def reference(case):
    """fp32 reference for y += ((x @ A^T) @ B^T) * scale, per row's slot."""
    x = case["x"].float()
    out = case["y"].float().clone()
    off = 0
    for s, h2 in enumerate(case["slices"]):
        A = case["lora_a"][s].float()  # [slots, 1, R, H1]
        Bw = case["lora_b"][s].float()  # [slots, 1, h2, R]
        for t in range(case["B"]):
            slot = int(case["idx"][t].item())
            if slot < 0:
                continue
            z1 = A[slot, 0] @ x[t]              # [R]
            out[t, off : off + h2] += (Bw[slot, 0] @ z1) * case["scale"]
        off += h2
    return out


def call_op(case):
    y = case["y"].clone()
    seq_len = torch.tensor([case["B"]], dtype=torch.int32)
    lora_indices = torch.tensor([0], dtype=torch.int32)
    torch.ops._C_ascend.add_lora(
        y,
        case["x"],
        case["lora_a"],
        case["lora_b"],
        lora_indices,
        seq_len,
        case["idx"],
        case["slices"],
        0,                 # offset_start (must be 0 for the fused kernel)
        case["scale"],
        True,              # add_inputs
        torch.tensor(False, dtype=torch.bool),  # use_gmm  -> decode branch
        torch.tensor(False, dtype=torch.bool),  # no_lora
        torch.tensor(True, dtype=torch.bool),   # use_add_lora -> want fused
    )
    torch.npu.synchronize()
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", metavar="DIR", default=None,
                    help="also capture a torch_npu profile into DIR")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    environment_report()

    device = "npu:0"
    torch.npu.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    case = build_case(device, dtype)

    print("\n=== running add_lora (eligible decode shape) ===")
    print(f"B={case['B']} H1={case['x'].shape[1]} R={case['R']} slices={case['slices']} dtype={dtype}")
    got = call_op(case)
    want = reference(case)
    err = (got.float() - want).abs().max().item()
    delta = (got.float() - case["y"].float()).abs().max().item()
    print(f"max|err| vs fp32 reference : {err:.4e}")
    print(f"max|delta| applied to y    : {delta:.4e}")
    if delta == 0.0:
        print("VERDICT: y was NOT modified -- the launch did nothing (failed raw launch?)")
    elif err < 5e-2:
        print("VERDICT: numerically correct (fused kernel or bgmv fallback both pass here)")
    else:
        print("VERDICT: WRONG RESULT")

    if args.profile:
        os.makedirs(args.profile, exist_ok=True)
        exp = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=exp,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile),
        ) as prof:
            for _ in range(5):
                call_op(case)
                prof.step()
        print(f"\nprofile written to {args.profile}; now grep it:")
        print(f"  grep -rioE 'add_lora_z[12][a-z_0-9]*|bgmv_(shrink|expand)[a-z_0-9]*' "
              f"{args.profile}/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv | sort | uniq -c")
    return 0


if __name__ == "__main__":
    sys.exit(main())
