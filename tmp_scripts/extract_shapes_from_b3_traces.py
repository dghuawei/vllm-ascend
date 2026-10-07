#!/usr/bin/env python3
"""Extract GroupedMatmul* / add_lora_z* production shapes+durations from b3
kernel_details.csv traces (vllm-ascend LoRA runs captured 2026-10-01/02 on
researchagent-node1, 910B3).

Usage: python3 extract_shapes_from_b3_traces.py <trace_root> [...]
Writes a per-trace summary to stdout.
"""
import csv, sys, os, glob
from collections import defaultdict

PATTERNS = ("GroupedMatmul", "add_lora", "bgmv", "Swiglu", "SwiGlu",
            "DynamicQuant", "QuantBatchMatmul")

def interesting(name):
    return any(p.lower() in name.lower() for p in PATTERNS)

def run(path):
    print("=" * 100)
    print("TRACE:", path)
    rows = defaultdict(list)          # (name, inshapes) -> [dur]
    typ  = {}
    core = {}
    n = 0
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for rec in r:
            n += 1
            name = rec["Name"]
            if not interesting(name):
                continue
            key = (rec["Type"], rec["Input Shapes"], rec["Input Data Types"])
            rows[key].append(float(rec["Duration(us)"]))
            typ[key] = rec["Type"]
            core[key] = rec["Accelerator Core"]
    print(f"total kernel rows: {n}")
    out = []
    for key, durs in rows.items():
        durs.sort()
        out.append((sum(durs), len(durs), key, durs))
    out.sort(reverse=True)
    print(f"{'total_us':>12} {'calls':>7} {'avg_us':>9} {'med_us':>9} {'min_us':>9} {'max_us':>9}  core            type / shapes")
    for total, cnt, key, durs in out:
        t, shapes, dtypes = key
        med = durs[len(durs)//2]
        print(f"{total:12.1f} {cnt:7d} {total/cnt:9.2f} {med:9.2f} {durs[0]:9.2f} {durs[-1]:9.2f}  {core[key]:<14}  {t}")
        print(f"{'':>12} shapes: {shapes}")
        print(f"{'':>12} dtypes: {dtypes}")
    return out

if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:
        for p in sorted(glob.glob(os.path.join(a, "**", "kernel_details.csv"), recursive=True)):
            run(p)
