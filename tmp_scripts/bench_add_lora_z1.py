"""B4-vs-B3 microbenchmark for the fused add_lora z1/z2 kernels.

Why this shape SWEEP and not one "production shape": raw <<<>>> launches carry
no Input Shapes in kernel_details.csv, and inverting the host-side grid planner
against the observed Block Num is under-determined (tmp_scripts/solve_z1_grid.py
shows dozens of (batch, nSlices, R) solutions per block count). So instead we
bracket the production range and report the per-shape b3/b4 ratio:

  * ratio flat across the sweep  -> a chip-level property (bandwidth / AIV count)
  * ratio blows up at particular (batch, H1) -> a real mechanism to chase

The within-chip control is the point: at every shape we also time the bgmv
fallback on the SAME data. bgmv is a different kernel doing the same math, so
  z1 2x slower AND bgmv 2x slower  -> the chip
  z1 2x slower BUT bgmv ~1x        -> our kernel

Production context (DeepSeek-V4-Flash, w8a8, TP8/EP8, from the b3 traces):
hidden 4096, moe intermediate 2048, E=32 local experts, R=16, L=3 adapter slots,
decode batch 48 (8 seqs x 6 spec tokens) for dense layers and routed-row counts
of 30..504 for the MoE layers.

b3 reference numbers to compare against (same traces):
  add_lora_z1_bfloat16_t_0  14104 calls  avg 69.42 us  med 21.26  min 5.36
  add_lora_z2_bfloat16_t_2  14104 calls  avg 19.40 us  med 13.72  min 5.62
  bgmv_shrink_bfloat16_t_8   1548 calls  avg 88.02 us  med 87.04
  bgmv_expand_bfloat16_t_6   1548 calls  avg 38.69 us  med 38.10

Usage:
    python tmp_scripts/bench_add_lora_z1.py --out /tmp/z1bench
    python tmp_scripts/bench_add_lora_z1.py --out /tmp/z1bench --quick
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  (lazy in utils.py:427; op ns needs it)

# ---------------------------------------------------------------------------
# environment / chip identity -- this is half the experiment
# ---------------------------------------------------------------------------

ACL_HEADER_GLOBS = [
    "/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_base.h",
    "/usr/local/Ascend/ascend-toolkit/latest/*/include/acl/acl_base.h",
    "/usr/local/Ascend/*/ascend-toolkit/latest/include/acl/acl_base.h",
]


def _vector_core_enum():
    """ACL_DEVICE_INFO_VECTOR_CORE_NUM's value, read out of the CANN header so
    we do not hardcode an enum that differs between CANN releases."""
    for pat in ACL_HEADER_GLOBS:
        for h in glob.glob(pat):
            try:
                txt = open(h, errors="ignore").read()
            except OSError:
                continue
            if "ACL_DEVICE_INFO_VECTOR_CORE_NUM" not in txt:
                continue
            # the enum block is a plain list; count position from its start
            start = txt.index("aclDeviceInfo")
            block = txt[start:txt.index("}", start)]
            names = []
            for ln in block.splitlines():
                ln = ln.strip().rstrip(",")
                if ln.startswith("ACL_DEVICE_INFO"):
                    names.append(ln.split("=")[0].strip())
            if "ACL_DEVICE_INFO_VECTOR_CORE_NUM" in names:
                return names.index("ACL_DEVICE_INFO_VECTOR_CORE_NUM"), h, names
    return None, None, None


def aiv_count():
    """What add_lora_fused_inplace itself queries (torch_binding.cpp:857)."""
    import ctypes

    val, hdr, names = _vector_core_enum()
    if val is None:
        return None, "acl_base.h not found"
    try:
        lib = ctypes.CDLL("libascendcl.so")
    except OSError as e:
        return None, f"libascendcl.so: {e}"
    out = ctypes.c_int64(0)
    rc = lib.aclGetDeviceCapability(ctypes.c_int(0), ctypes.c_int(val), ctypes.byref(out))
    return (out.value if rc == 0 else None), f"enum={val} rc={rc} hdr={hdr}"


def environment_report():
    print("=== chip / stack identity ===")
    print("hostname         :", os.uname().nodename)
    print("torch            :", torch.__version__)
    print("torch_npu        :", getattr(torch_npu, "__version__", "?"))
    print("soc_version (rt) :", torch_npu.npu.get_soc_version(), "(220-225=A2)")
    try:
        print("device name      :", torch_npu.npu.get_device_name(0))
    except Exception as e:
        print("device name      : ?", e)
    aiv, how = aiv_count()
    print("AIV cores        :", aiv, f"({how})")
    try:
        p = torch_npu.npu.get_device_properties(0)
        print("device props     :", p)
    except Exception as e:
        print("device props     : ?", e)
    for v in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        print(f"{v:<17}:", os.environ.get(v, "-"))
    try:
        ver = subprocess.run(
            ["bash", "-lc",
             "cat $ASCEND_HOME_PATH/version.cfg 2>/dev/null || "
             "cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        print("CANN version.cfg :", ver.replace("\n", " | ") or "-")
    except Exception as e:
        print("CANN version.cfg : ?", e)
    try:
        import vllm_ascend as va
        from vllm_ascend import _build_info
        print("vllm_ascend      :", os.path.dirname(va.__file__))
        for a in ("__soc_version__", "__device_type__"):
            print(f"_build_info.{a:<16}:", getattr(_build_info, a, "?"))
        so = os.path.join(os.path.dirname(va.__file__), "libvllm_ascend_kernels.so")
        print("kernels .so      :", so, os.path.getsize(so) if os.path.exists(so) else "MISSING")
    except Exception as e:
        print("vllm_ascend      : ?", e)
    print()
    return aiv


# ---------------------------------------------------------------------------
# the shapes
# ---------------------------------------------------------------------------
# Eligibility (csrc/torch_binding.cpp:add_lora_eligible) requires H1 % 16 == 0,
# every output_slice % 16 == 0, y_width % 16 == 0, offset_start == 0,
# nSlices <= 4, R in {16, 32, 64}.  Decode size caps (torch_binding.cpp:823-826):
# T <= 256 for a single slice, T <= 1024 once there are >= 2 slices.
L_SLOTS = 3   # max_loras in the production config
RANK = 16

def shape_matrix(quick):
    cases = []
    # dense-layer decode: the real one is batch 48 (8 seqs x 6 spec tokens)
    for batch in ([48] if quick else [8, 48, 96, 256]):
        cases.append(dict(batch=batch, H1=4096, R=RANK, slices=[256, 256], tag="dense-qkv"))
        cases.append(dict(batch=batch, H1=4096, R=RANK, slices=[4096], tag="dense-wide1"))
        cases.append(dict(batch=batch, H1=256, R=RANK, slices=[4096], tag="dense-o"))
    # MoE-layer decode: batch is the routed row count (tokens x top_k), which is
    # why these get large; >= 2 slices to stay under the 1024 cap
    for batch in ([288] if quick else [144, 288, 504, 1024]):
        cases.append(dict(batch=batch, H1=4096, R=RANK, slices=[2048, 2048], tag="moe-w13"))
        cases.append(dict(batch=batch, H1=2048, R=RANK, slices=[2048, 2048], tag="moe-w2x2"))
    # rank sensitivity: rankBlock/groups logic keys off R vs aivNum
    if not quick:
        for R in (32, 64):
            cases.append(dict(batch=48, H1=4096, R=R, slices=[256, 256], tag=f"rank{R}"))
    return cases


def build_case(c, device, dtype):
    torch.manual_seed(0)
    batch, H1, R, slices = c["batch"], c["H1"], c["R"], c["slices"]
    x = torch.randn(batch, H1, device=device, dtype=dtype)
    y = torch.randn(batch, sum(slices), device=device, dtype=dtype)
    lora_a = [torch.randn(L_SLOTS, 1, R, H1, device=device, dtype=dtype).mul_(0.05)
              for _ in slices]
    lora_b = [torch.randn(L_SLOTS, 1, h2, R, device=device, dtype=dtype).mul_(0.05)
              for h2 in slices]
    # realistic slot mix including -1 (no-lora) rows, which both paths must skip
    idx = torch.randint(-1, L_SLOTS, (batch,), device=device, dtype=torch.int64)
    return dict(c, x=x, y=y, lora_a=lora_a, lora_b=lora_b, idx=idx,
                seq_len=torch.tensor([batch], dtype=torch.int32),
                lora_indices=torch.tensor([0], dtype=torch.int32),
                t_false=torch.tensor(False, dtype=torch.bool),
                t_true=torch.tensor(True, dtype=torch.bool))


def call(case, fused):
    torch.ops._C_ascend.add_lora(
        case["y"], case["x"], case["lora_a"], case["lora_b"],
        case["lora_indices"], case["seq_len"], case["idx"],
        case["slices"], 0, 0.5, True,
        case["t_false"],                               # use_gmm -> decode branch
        case["t_false"],                               # no_lora
        case["t_true"] if fused else case["t_false"],  # use_add_lora
    )


def event_time_us(case, fused, iters, warmup=10):
    for _ in range(warmup):
        call(case, fused)
    torch.npu.synchronize()
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call(case, fused)
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters   # ms -> us per call


def profile_kernels(case, fused, outdir, iters):
    """Authoritative per-kernel device times, straight from kernel_details.csv."""
    os.makedirs(outdir, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(outdir),
    ) as prof:
        for _ in range(iters):
            call(case, fused)
            prof.step()
    agg = {}
    for p in glob.glob(os.path.join(outdir, "**", "kernel_details.csv"), recursive=True):
        for rec in csv.DictReader(open(p)):
            n = rec["Name"]
            key = None
            for pat in ("add_lora_z1", "add_lora_z2", "bgmv_shrink", "bgmv_expand"):
                if pat in n:
                    key = pat
            if key is None:
                continue
            agg.setdefault(key, {"calls": 0, "us": 0.0, "blocks": rec["Block Num"]})
            agg[key]["calls"] += 1
            agg[key]["us"] += float(rec["Duration(us)"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for profiles + results.csv")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--prof-iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    aiv = environment_report()

    device = "npu:0"
    torch.npu.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    rows = []
    hdr = (f"{'tag':<12} {'batch':>6} {'H1':>6} {'R':>3} {'slices':>12} "
           f"{'fused_us':>9} {'bgmv_us':>9} {'speedup':>8} "
           f"{'z1_us':>8} {'z2_us':>8} {'z1_blk':>7}")
    print("=== sweep ===")
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(shape_matrix(args.quick)):
        case = build_case(c, device, dtype)
        try:
            t_fused = event_time_us(case, True, args.iters)
            t_bgmv = event_time_us(case, False, args.iters)
        except Exception as e:
            print(f"{c['tag']:<12} {c['batch']:>6} FAILED: {e}")
            continue
        z1 = z2 = float("nan")
        z1blk = "-"
        if not args.no_profile:
            d = os.path.join(args.out, f"prof_{i:02d}_{c['tag']}_{c['batch']}")
            agg = profile_kernels(case, True, d, args.prof_iters)
            if "add_lora_z1" in agg:
                z1 = agg["add_lora_z1"]["us"] / agg["add_lora_z1"]["calls"]
                z1blk = agg["add_lora_z1"]["blocks"]
            if "add_lora_z2" in agg:
                z2 = agg["add_lora_z2"]["us"] / agg["add_lora_z2"]["calls"]
            if "add_lora_z1" not in agg:
                z1blk = "NO-Z1!"   # fell back; the comparison would be meaningless
        print(f"{c['tag']:<12} {c['batch']:>6} {c['H1']:>6} {c['R']:>3} "
              f"{str(c['slices']):>12} {t_fused:9.2f} {t_bgmv:9.2f} "
              f"{t_bgmv / t_fused:8.2f} {z1:8.2f} {z2:8.2f} {z1blk:>7}")
        rows.append(dict(host=os.uname().nodename, soc=torch_npu.npu.get_soc_version(),
                         aiv=aiv, tag=c["tag"], batch=c["batch"], H1=c["H1"], R=c["R"],
                         slices="x".join(map(str, c["slices"])),
                         fused_us=round(t_fused, 3), bgmv_us=round(t_bgmv, 3),
                         z1_us=round(z1, 3), z2_us=round(z2, 3), z1_blocks=z1blk))

    out_csv = os.path.join(args.out, "results.csv")
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
