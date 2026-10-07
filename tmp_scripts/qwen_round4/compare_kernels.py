#!/usr/bin/env python3
"""Compare per-kernel behavior between two ASCEND_PROFILER_OUTPUT traces.

Reference: 43L full-model capture  (~/work/huawei/prof/new_kernel/new_kernel)
Test:      6L truncated capture    (~/work/huawei/prof/b3trunc/<rank0 dir>)

For every kernel name present in either trace, report calls, total us, and
per-call duration (avg / p50 / p90) in each trace, plus ratios. The
question being answered is whether per-CALL durations match (kernels "behave
the same") while call counts differ roughly proportional to layer count x
steps in window.

Usage: compare_kernels.py REF_CSV TEST_CSV [--top 45] [--out json]
"""
import argparse
import csv
import json
import sys


def load(path):
    fam = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        ki = header.index("Name")
        ti = header.index("Duration(us)")
        for line in f:
            parts = next(csv.reader([line]))
            if len(parts) <= max(ki, ti):
                continue
            try:
                d = float(parts[ti])
            except ValueError:
                continue
            fam.setdefault(parts[ki], []).append(d)
    return fam


def stats(ds):
    ds = sorted(ds)
    n = len(ds)
    return {
        "calls": n,
        "sum_us": sum(ds),
        "avg_us": sum(ds) / n,
        "p50_us": ds[n // 2],
        "p90_us": ds[int(0.9 * (n - 1))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_csv")
    ap.add_argument("test_csv")
    ap.add_argument("--top", type=int, default=45)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grep", default=None, help="regex-ish substring filter on kernel name")
    args = ap.parse_args()

    ref = load(args.ref_csv)
    test = load(args.test_csv)
    names = set(ref) | set(test)

    rows = []
    for n in names:
        r = stats(ref[n]) if n in ref else None
        t = stats(test[n]) if n in test else None
        rows.append({"kernel": n, "ref": r, "test": t})
    rows.sort(key=lambda x: -(x["ref"]["sum_us"] if x["ref"] else 0))

    ref_tot = sum(sum(v) for v in ref.values())
    test_tot = sum(sum(v) for v in test.values())
    ref_n = sum(len(v) for v in ref.values())
    test_n = sum(len(v) for v in test.values())
    print(f"REF  (43L): {ref_n} launches, total device {ref_tot/1e6:.2f} s")
    print(f"TEST ( 6L): {test_n} launches, total device {test_tot/1e6:.2f} s")
    print(f"layer ratio 6/43 = {6/43:.4f}")

    hdr = (f"{'kernel':52s} {'calls43L':>9s} {'calls6L':>8s} {'x_call':>6s} "
           f"{'avg43L':>8s} {'avg6L':>8s} {'avgR':>5s} {'p50 43L':>8s} {'p50 6L':>7s} "
           f"{'p90 43L':>8s} {'p90 6L':>7s} {'sum43L_ms':>10s} {'sum6L_ms':>9s}")
    print("\n" + hdr)
    shown = 0
    out_rows = []
    for row in rows:
        n = row["kernel"]
        if args.grep and args.grep not in n:
            continue
        if not args.grep and shown >= args.top:
            continue
        r, t = row["ref"], row["test"]
        cr = r["calls"] if r else 0
        ct = t["calls"] if t else 0
        calls_ratio = (ct / cr) if cr else float("nan")
        avg_ratio = (t["avg_us"] / r["avg_us"]) if (r and t) else float("nan")
        fmt = lambda v, w, p: (f"{v:{w}.{p}f}" if v == v else "    n/a")
        line = (f"{n[:52]:52s} {cr:9d} {ct:8d} {calls_ratio:6.3f} "
                + (f"{r['avg_us']:8.2f}" if r else f"{'n/a':>8s}")
                + (f"{t['avg_us']:8.2f}" if t else f"{'n/a':>8s}")
                + (f"{avg_ratio:5.2f}" if avg_ratio == avg_ratio else "  n/a")
                + (f"{r['p50_us']:8.2f}" if r else f"{'n/a':>8s}")
                + (f"{t['p50_us']:7.2f}" if t else f"{'n/a':>7s}")
                + (f"{r['p90_us']:8.2f}" if r else f"{'n/a':>8s}")
                + (f"{t['p90_us']:7.2f}" if t else f"{'n/a':>7s}")
                + (f"{r['sum_us']/1e3:10.1f}" if r else f"{'n/a':>10s}")
                + (f"{t['sum_us']/1e3:9.1f}" if t else f"{'n/a':>9s}"))
        print(line)
        if args.grep or shown < args.top:
            shown += 1
        out_rows.append(
            {
                "kernel": n,
                "ref_calls": cr,
                "test_calls": ct,
                "calls_ratio": calls_ratio,
                "ref_avg_us": r["avg_us"] if r else None,
                "test_avg_us": t["avg_us"] if t else None,
                "avg_ratio": avg_ratio,
                "ref_p50_us": r["p50_us"] if r else None,
                "test_p50_us": t["p50_us"] if t else None,
                "ref_p90_us": r["p90_us"] if r else None,
                "test_p90_us": t["p90_us"] if t else None,
                "ref_sum_us": r["sum_us"] if r else None,
                "test_sum_us": t["sum_us"] if t else None,
            }
        )

    only_test = [n for n in test if n not in ref]
    only_ref = [n for n in ref if n not in test]
    print(f"\nkernels only in 6L trace ({len(only_test)}): {sorted(only_test)[:15]}")
    print(f"kernels only in 43L trace ({len(only_ref)}): {sorted(only_ref)[:15]}")

    if args.out:
        json.dump(out_rows, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
