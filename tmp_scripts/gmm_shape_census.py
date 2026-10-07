#!/usr/bin/env python3
"""Census of every GroupedMatmul* invocation in a b3 kernel_details.csv,
grouped by (type, shapes), sorted by M (the row count) so prefill vs decode
is obvious.  Prints a compact table meant to seed a microbenchmark."""
import csv, sys, glob, os
from collections import defaultdict

def m_of(shapes):
    s = shapes.strip('"')
    first = s.split(';')[0]
    try:
        return int(first.split(',')[0])
    except Exception:
        return -1

def run(path):
    agg = defaultdict(list)
    meta = {}
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            t = rec["Type"]
            if not t.startswith("GroupedMatmul"):
                continue
            key = (t, rec["Input Shapes"].strip('"'), rec["Input Data Types"])
            agg[key].append(float(rec["Duration(us)"]))
            meta[key] = rec["Accelerator Core"]
    rows = []
    for key, d in agg.items():
        d.sort()
        rows.append((m_of(key[1]), key, d))
    rows.sort()
    print(f"{'M':>8} {'calls':>6} {'avg_us':>9} {'med_us':>9} {'min_us':>9} {'max_us':>9}  type  shapes")
    for m, key, d in rows:
        t, shapes, dt = key
        print(f"{m:8d} {len(d):6d} {sum(d)/len(d):9.2f} {d[len(d)//2]:9.2f} {d[0]:9.2f} {d[-1]:9.2f}  {t}  {shapes}")
        print(f"{'':>8}   dtypes: {dt}")

for a in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(a, "**", "kernel_details.csv"), recursive=True)):
        print("=" * 110); print(p)
        run(p)
