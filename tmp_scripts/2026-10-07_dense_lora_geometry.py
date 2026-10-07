#!/usr/bin/env python3
"""Pin the DENSE LoRA geometry (H1, output_slices, R) that add_lora_z1/z2 see.

z1/z2 are raw <<<>>> launches, so the trace carries no shapes for them. But in
the SAME trace the prefill steps run the identical geometry through
GroupedMatmul, which IS an aclnn op with Input Shapes. The dense/attention LoRA
gmms are the 1-group ones (`M,K x 1,K,N`); the MoE ones have 32 groups. So:

    shrink  M,H1   x 1,H1,(nSlices*R)
    expand  M,R    x 1,R,h2_s          one call per slice s

Reading those off gives H1, R and every h2_s per LoRA'd dense layer -- exactly
the tuple add_lora_fused_impl is launched with at decode, where only M changes.

Usage: python3 tmp_scripts/2026-10-07_dense_lora_geometry.py [trace_dir]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

DEFAULT = os.path.expanduser(
    "~/work/huawei/prof/lora_fusedswiglu")


def find_csvs(root):
    out = []
    for dirpath, _d, files in os.walk(root):
        if "kernel_details.csv" in files:
            out.append(os.path.join(dirpath, "kernel_details.csv"))
    return sorted(out)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    agg = defaultdict(list)
    for path in find_csvs(root):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Name") or "").lower()
                if "groupedmatmul" not in name:
                    continue
                shapes = (row.get("Input Shapes") or "").strip('" ')
                dtypes = (row.get("Input Data Types") or "").strip('" ')
                parts = shapes.split(";")
                if len(parts) < 2:
                    continue
                b = parts[1].split(",")
                # 1 group == the dense/attention path; 32 groups == MoE experts
                if len(b) != 3 or b[0] != "1":
                    continue
                try:
                    agg[(shapes, dtypes)].append(float(row["Duration(us)"]))
                except (TypeError, ValueError, KeyError):
                    pass

    print(f"# dense (1-group) LoRA GroupedMatmul in {root}\n")
    rows = sorted(agg.items(), key=lambda kv: -sum(kv[1]))
    for (shapes, dtypes), ds in rows:
        a, b = shapes.split(";")[:2]
        M, K = a.split(",")
        _, K2, N = b.split(",")
        kind = "shrink" if int(K) > int(N) else "expand"
        print(f"  {kind:6s} M={M:>6} K={K2:>6} N={N:>6}  n={len(ds):5d} "
              f"avg={st.mean(ds):8.2f}us   [{dtypes}]")
    if not rows:
        print("  (none)")


if __name__ == "__main__":
    main()
