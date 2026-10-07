#!/usr/bin/env python3
"""Extract production shapes/dtypes for the three kernels being optimised.

Reads every ASCEND_PROFILER_OUTPUT/kernel_details.csv under ~/work/huawei/prof
and reports, per kernel family, the distinct (Input Shapes, Input Data Types,
Input Formats, Block Num) keys with duration stats.

  * ScatterNdUpdateV2   -- aclnn op: `Input Shapes` IS populated, so the real
                           production shapes come straight out of the trace.
  * add_lora_z1/z2, add_lora_swiglu_quant -- raw <<<>>> launches: the trace
                           carries no shapes (N/A). Only counts/durations are
                           meaningful here; their shapes come from the config.

Matching is by SUBSTRING on both Name and Type, because aclnn ops appear as
`aclnnScatterNdUpdateV2_ScatterNdUpdateV2AiCore_ScatterNdUpdateV2` while raw
launches appear as bare `add_lora_z1_bfloat16_t_0`.

Usage:  python3 tmp_scripts/2026-10-07_extract_three_kernel_shapes.py [prof_root]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/work/huawei/prof")

FAMILIES = {
    "scatter_nd_update_v2": "scatterndupdatev2",
    "add_lora_z1": "add_lora_z1",
    "add_lora_z2": "add_lora_z2",
    "add_lora_swiglu_quant": "add_lora_swiglu_quant",
}


def find_csvs(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "kernel_details.csv" in filenames:
            out.append(os.path.join(dirpath, "kernel_details.csv"))
    return sorted(out)


def main():
    csvs = find_csvs(ROOT)
    print(f"# scanned {len(csvs)} kernel_details.csv under {ROOT}\n")

    agg = defaultdict(lambda: defaultdict(list))
    traces = defaultdict(set)

    for path in csvs:
        trace = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Name") or "")
                typ = (row.get("Type") or "")
                hay = (name + "|" + typ).lower()
                for fam, needle in FAMILIES.items():
                    if needle in hay:
                        key = (
                            (row.get("Input Shapes") or "N/A").strip('" '),
                            (row.get("Input Data Types") or "N/A").strip('" '),
                            (row.get("Input Formats") or "N/A").strip('" '),
                            (row.get("Block Num") or "?").strip(),
                        )
                        try:
                            agg[fam][key].append(float(row["Duration(us)"]))
                        except (TypeError, ValueError, KeyError):
                            pass
                        traces[fam].add(trace)
                        break

    for fam in FAMILIES:
        if fam not in agg:
            print(f"## {fam}: NOT PRESENT in any trace\n")
            continue
        total = sum(len(v) for v in agg[fam].values())
        print(f"## {fam}: {total} calls across {len(traces[fam])} traces")
        rows = sorted(agg[fam].items(), key=lambda kv: -sum(kv[1]))
        for (shapes, dtypes, fmts, blk), ds in rows[:30]:
            ds = sorted(ds)
            print(f"  n={len(ds):6d} blk={blk:>4} tot={sum(ds)/1000:8.1f}ms "
                  f"avg={st.mean(ds):8.2f} med={st.median(ds):8.2f} "
                  f"min={ds[0]:7.2f} max={ds[-1]:8.2f} us")
            if shapes != "N/A":
                print(f"         shapes = {shapes}")
                print(f"         dtypes = {dtypes}")
                print(f"         formats= {fmts}")
        if len(rows) > 30:
            print(f"  ... {len(rows)-30} more distinct keys")
        print()


if __name__ == "__main__":
    main()
