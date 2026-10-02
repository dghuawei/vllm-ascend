#!/usr/bin/env python3
"""Turn an msprof op_summary into per-shape DEVICE time.

Wall-clock timing of a launch loop measures the launcher whenever the host
cannot issue faster than the device executes -- which is exactly the decode
regime (T<=128, kernel in single-digit us). This reads the hardware Task
Duration instead, and reports the dense device cost of one call: the sum of
every op in one iteration, which is what vLLM sees in compiled mode.

usage:
  device_time.py --summary op_summary.csv --log run.log --variant NAME
                 [--filter swiglu] [--marker cumsum] [--bytes-fused|--bytes-unfused]

Two ways to find each shape's measured window:
  --filter  keep only rows whose op name matches, then consume the exact
            launch counts the MANIFEST lines declare (C++ harness: our kernel
            is the only thing on the stream).
  --marker  the runner brackets each measured window with a distinctive op;
            rows strictly between marker 2k and 2k+1 are shape k's window
            (python harness: setup ops share names with measured ops).

Output is in the same shape the wall-clock harness prints, so
tabulate_matrix.py reads it with no changes.
"""
import argparse
import csv
import re
import sys

MANIFEST = re.compile(
    r"^MANIFEST T=(\d+) W=(\d+) delta=(\d+) iters=(\d+) launches=(\d+) measured_from=(\d+)")


def pick(keys, *cands):
    for c in cands:
        for k in keys:
            if c in k.strip().lower():
                return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--filter")
    ap.add_argument("--marker")
    ap.add_argument("--unfused", action="store_true",
                    help="use the 21E/9E unfused byte count instead of 11E/7E")
    a = ap.parse_args()

    shapes = []
    for line in open(a.log, errors="replace"):
        m = MANIFEST.match(line)
        if m:
            T, W, d, iters, launches, mfrom = (int(x) for x in m.groups())
            shapes.append(dict(T=T, W=W, d=d, iters=iters, launches=launches, mfrom=mfrom))
    if not shapes:
        raise SystemExit("no MANIFEST lines in " + a.log)

    rows = list(csv.DictReader(open(a.summary)))
    if not rows:
        raise SystemExit("empty op_summary")
    keys = list(rows[0].keys())
    k_name = pick(keys, "op name", "op_name")
    k_dur = pick(keys, "task duration", "duration")
    k_start = pick(keys, "task start time", "start time")
    if not (k_name and k_dur):
        raise SystemExit(f"cannot find name/duration columns in {keys}")

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return None

    rows = [r for r in rows if num(r, k_dur) is not None]
    if k_start:
        rows.sort(key=lambda r: num(r, k_start) if num(r, k_start) is not None else 0.0)

    windows = []
    if a.marker:
        marks = [i for i, r in enumerate(rows) if a.marker.lower() in (r[k_name] or "").lower()]
        if not marks:
            raise SystemExit(f"no op name matches marker '{a.marker}'")
        # One marker call can decompose into several ops (cumsum emits a MemSet
        # then a Cumsum), so collapse consecutive marker rows into one cluster.
        clusters = [[marks[0]]]
        for i in marks[1:]:
            if i == clusters[-1][-1] + 1:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        if len(clusters) != 2 * len(shapes):
            raise SystemExit(f"found {len(clusters)} marker clusters from {len(marks)} marker "
                             f"ops, need {2 * len(shapes)} for {len(shapes)} shapes")
        for k in range(len(shapes)):
            lo = clusters[2 * k][-1] + 1
            hi = clusters[2 * k + 1][0]
            windows.append([r for r in rows[lo:hi]
                            if a.marker.lower() not in (r[k_name] or "").lower()])
    else:
        sel = [r for r in rows
               if not a.filter or a.filter.lower() in (r[k_name] or "").lower()]
        want = sum(s["launches"] for s in shapes)
        if len(sel) != want:
            print(f"   !! op row count {len(sel)} != expected {want}; "
                  f"segmentation is unreliable", file=sys.stderr)
        pos = 0
        for s in shapes:
            blk = sel[pos:pos + s["launches"]]
            pos += s["launches"]
            windows.append(blk[s["mfrom"]:])

    for s, win in zip(shapes, windows):
        if not win:
            continue
        total = sum(num(r, k_dur) or 0.0 for r in win)
        us = total / s["iters"]
        E = s["T"] * s["W"]
        if a.unfused:
            b = E * (21.0 if s["d"] else 9.0)
        else:
            b = E * (11.0 if s["d"] else 7.0)
        b += s["T"] * 4.0
        print(f"## variant={a.variant} T={s['T']} W={s['W']} delta={s['d']} iters={s['iters']}")
        print(f"T={s['T']} W={s['W']} delta={s['d']} aiv=40 | {us:.2f} us | "
              f"{b / (us * 1e-6) / 1e9:.1f} GB/s")
        names = sorted({(r[k_name] or "?").strip() for r in win})
        print(f"   device: {len(win)} ops over {s['iters']} iters "
              f"({len(win) / s['iters']:.1f}/iter) {names if len(names) <= 4 else names[:4]}")


main()
