#!/usr/bin/env python3
"""Summarise msprof op_summary PipeUtilization rows for the swiglu kernel."""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
name_key = None
for k in (rows[0].keys() if rows else []):
    if k.strip().lower() in ("op name", "op_name"):
        name_key = k
        break

sel = [r for r in rows if name_key and "swiglu" in (r[name_key] or "").lower()]
if not sel:
    print("   (no swiglu rows) op names seen:",
          sorted({(r[name_key] or "?") for r in rows})[:10] if name_key else "?")
    sys.exit()

want = [k for k in sel[0].keys()
        if any(s in k.lower() for s in
               ("task duration", "aiv_", "aic_", "ratio", "vec_", "mte",
                "scalar", "cube", "bound"))]

print(f"   n={len(sel)} rows")
means = {}
for k in want:
    nums = []
    for r in sel:
        try:
            nums.append(float(r[k]))
        except (TypeError, ValueError):
            pass
    if not nums:
        continue
    means[k] = sum(nums) / len(nums)
    print(f"     {k.strip():<42s} mean={means[k]:10.4f}  max={max(nums):10.4f}")


def is_ratio(k):
    low = k.strip().lower()
    return low.endswith("ratio") and not any(
        s in low for s in ("time", "duration", "cycle"))


summed = [k for k in means if is_ratio(k)]
if summed:
    total = sum(means[k] for k in summed)
    print(f"     {'SUM OF RATIOS':<42s}      {total:10.4f}")
    print(f"       summed: {', '.join(sorted(k.strip() for k in summed))}")
