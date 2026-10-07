#!/usr/bin/env python3
"""Compare kernel behavior 43L-ref vs 6L-test keyed on NORMALIZED Type column.

kernel_details.csv "Type" holds op/raw-kernel labels that differ by stack
version (hash-suffixed micro-kernel names vs clean op names), so exact-name
joins produce false absences. This normalizes Type into families:
  - strip 16+hex hashes and what follows them
  - strip _high_performance/_high_precision/_normal tails, _kernelN
  - strip trailing _<digits>
  - map to the longest underscore-boundary prefix that is a known base
    (bases = clean Type set of the 6L trace + hcom_* + a few ref bases)

Usage: compare_families.py REF_CSV TEST_CSV [--out json] [--top 40]
"""
import argparse
import csv
import json
import re
import sys

csv.field_size_limit(10**7)

HASH = re.compile(r"_[0-9a-f]{16,}")


def norm(name, bases):
    if name.startswith("hcom_"):
        m = re.match(r"hcom_(allGather|allReduce|allGatherV|allReduce|broadcast|reduceScatter)", name)
        return "hcom_" + (m.group(1) if m else name[5:].rstrip("_"))
    n = HASH.split(name)[0]
    n = re.sub(r"_(high_performance|high_precision|normal)\b.*$", "", n)
    n = re.sub(r"_kernel\d+$", "", n)
    while True:
        n2 = re.sub(r"_\d+$", "", n)
        if n2 == n:
            break
        n = n2
    # longest underscore-boundary prefix in bases
    best = None
    parts = n.split("_")
    for k in range(len(parts), 0, -1):
        cand = "_".join(parts[:k])
        if cand in bases:
            best = cand
            break
    return best if best else n


def load(path, bases=None):
    fam = {}
    with open(path) as f:
        r = csv.DictReader(f)
        rows = [(row["Type"], float(row["Duration(us)"])) for row in r]
    if bases is None:
        bases = set(t for t, _ in rows)
    for t, d in rows:
        fam.setdefault(norm(t, bases), []).append(d)
    return fam


def stats(ds):
    ds = sorted(ds)
    n = len(ds)
    return {"calls": n, "sum_us": sum(ds), "avg": sum(ds) / n,
            "p50": ds[n // 2], "p90": ds[int(0.9 * (n - 1))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_csv")
    ap.add_argument("test_csv")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    test = load(args.test_csv)                 # bases from clean 6L trace
    bases = set(test) | {"hcom_allReduce", "hcom_allGather",
                         "bgmv_shrink_bfloat16_t", "bgmv_expand_bfloat16_t",
                         "GroupedMatmulSwigluQuant", "DequantSwigluQuant",
                         "xpand_dflash_and_dspark_inputs_kernel_single_grid",
                         "rejection_greedy_sample_triton"}
    ref = load(args.ref_csv, bases)

    rt = sum(sum(v) for v in ref.values())
    tt = sum(sum(v) for v in test.values())
    rn = sum(len(v) for v in ref.values())
    tn = sum(len(v) for v in test.values())
    print(f"REF 43L rank0: {rn} launches, {rt/1e6:.2f} s device")
    print(f"TEST 6L rank0: {tn} launches, {tt/1e6:.2f} s device   "
          f"(launch ratio {tn/rn:.3f}, layer ratio 6/43={6/43:.3f})")

    names = sorted(set(ref) | set(test),
                   key=lambda n: -(sum(ref.get(n, [0])) and sum(ref[n])))
    hdr = (f"{'family':44s} {'n43':>7s} {'n6':>6s} {'x':>5s} "
           f"{'avg43':>8s} {'avg6':>8s} {'p50_43':>8s} {'p50_6':>7s} "
           f"{'p90_43':>9s} {'p90_6':>8s} {'ms43':>8s} {'ms6':>7s}")
    print("\n" + hdr)
    out = []
    for n in names:
        r, t = ref.get(n), test.get(n)
        if not r and not t:
            continue
        if not r or not t:
            one = r or t
            if sum(one) < 30e3:   # hide sub-30ms one-sided crumbs in table
                out.append({"family": n, "ref": r, "test": t})
                continue
        f = lambda v, w, p=2: (f"{v:{w}.{p}f}" if v is not None else f"{'-':>{w}s}")
        rs = stats(r) if r else None
        ts = stats(t) if t else None
        x = (ts["calls"] / rs["calls"]) if (rs and ts) else float("nan")
        line = (f"{n[:44]:44s} " +
                f"{(rs['calls'] if rs else 0):7d} {(ts['calls'] if ts else 0):6d} " +
                (f"{x:5.2f}" if x == x else f"{'-':>5s}") + " " +
                f"{f(rs['avg'] if rs else None, 8)} {f(ts['avg'] if ts else None, 8)} " +
                f"{f(rs['p50'] if rs else None, 8)} {f(ts['p50'] if ts else None, 7)} " +
                f"{f(rs['p90'] if rs else None, 9)} {f(ts['p90'] if ts else None, 8)} " +
                f"{f((rs['sum_us']/1e3) if rs else None, 8, 1)} {f((ts['sum_us']/1e3) if ts else None, 7, 1)}")
        print(line)
        out.append({"family": n, "ref": rs, "test": ts, "calls_ratio": x})

    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
