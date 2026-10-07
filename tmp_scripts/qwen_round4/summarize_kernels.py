#!/usr/bin/env python3
"""Aggregate kernel_details.csv from a 6L round-4 capture into a kernel-family
table (sum device us, call count, avg, p50) per rank, plus a cross-rank mean.

Usage: python3 summarize_kernels.py RANK_DIR_GLOB [--top 30] [--out out.json]
Finds */ASCEND_PROFILER_OUTPUT/kernel_details.csv under the given root.
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load_csv(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        try:
            ki = header.index("Name")
            ti = header.index("Duration(us)")
        except ValueError:
            # fall back: common column layouts
            ki = 1
            ti = len(header) - 1
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(ki, ti):
                continue
            try:
                rows.append((parts[ki], float(parts[ti])))
            except ValueError:
                continue
    return rows


def fam_table(rows):
    fam = {}
    for name, dur in rows:
        fam.setdefault(name, []).append(dur)
    out = []
    for name, ds in fam.items():
        ds_sorted = sorted(ds)
        out.append(
            {
                "kernel": name,
                "calls": len(ds),
                "sum_us": round(sum(ds), 1),
                "avg_us": round(sum(ds) / len(ds), 2),
                "p50_us": round(ds_sorted[len(ds_sorted) // 2], 2),
                "p90_us": round(ds_sorted[int(0.9 * (len(ds) - 1))], 2),
            }
        )
    out.sort(key=lambda r: -r["sum_us"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.root, "**/kernel_details.csv"), recursive=True))
    if not csvs:
        sys.exit(f"no kernel_details.csv under {args.root}")
    print(f"{len(csvs)} rank csvs")
    per_rank = {}
    for c in csvs:
        rank = c.split("_rank")[1].split("_")[0] if "_rank" in c else c
        rows = load_csv(c)
        per_rank[rank] = fam_table(rows)
        total = sum(r["sum_us"] for r in per_rank[rank])
        print(f"  rank{rank}: {len(rows)} kernel launches, total device {total/1e6:.3f} s")

    # cross-rank mean table keyed by kernel name
    names = set()
    for t in per_rank.values():
        names.update(r["kernel"] for r in t)
    nrank = len(per_rank)
    mean_rows = []
    for n in names:
        sums, calls = [], []
        for t in per_rank.values():
            hit = next((r for r in t if r["kernel"] == n), None)
            sums.append(hit["sum_us"] if hit else 0.0)
            calls.append(hit["calls"] if hit else 0)
        mean_rows.append(
            {
                "kernel": n,
                "mean_sum_us": round(sum(sums) / nrank, 1),
                "rank_spread_us": [round(min(sums), 1), round(max(sums), 1)],
                "mean_calls": round(sum(calls) / nrank, 1),
            }
        )
    mean_rows.sort(key=lambda r: -r["mean_sum_us"])

    print(f"\nTOP {args.top} kernel families by mean per-rank device time:")
    print(f"{'kernel':60s} {'mean_sum_ms':>11s} {'min-max_ms':>18s} {'mean_calls':>10s}")
    for r in mean_rows[: args.top]:
        lo, hi = r["rank_spread_us"]
        print(f"{r['kernel'][:60]:60s} {r['mean_sum_us']/1e3:11.2f} {lo/1e3:8.2f}-{hi/1e3:9.2f} {r['mean_calls']:10.1f}")

    if args.out:
        json.dump({"per_rank": per_rank, "cross_rank_mean": mean_rows}, open(args.out, "w"))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
