#!/usr/bin/env python3
"""Join the variant logs into one table.

usage: tabulate_matrix.py [--variants a,b,c] <log> [<log> ...]

--variants selects and orders the columns; the first is the reference the
ratios are taken against. Without it, every variant found is shown in the
order first seen. Logs carry both a wall-clock variant and a '<name>_dev'
device-time variant, so the two are usually tabulated separately.
Reads the '## variant=... T=... W=... delta=...' markers and the following
'T=.. W=.. delta=.. aiv=.. | X us | Y GB/s' line from each log.
"""
import re
import sys
from collections import defaultdict

HDR = re.compile(r"^## variant=(\S+) T=(\d+) W=(\d+) delta=(\d+)")
RES = re.compile(r"^T=(\d+) W=(\d+) delta=(\d+) aiv=(\d+) \| ([\d.]+) us \| ([\d.]+) GB/s")

args = sys.argv[1:]
want = None
if args and args[0] == "--variants":
    want = args[1].split(",")
    args = args[2:]

us = defaultdict(dict)
gbs = defaultdict(dict)
variants = []

for path in args:
    cur = None
    for line in open(path, errors="replace"):
        m = HDR.match(line)
        if m:
            cur = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            if m.group(1) not in variants:
                variants.append(m.group(1))
            continue
        m = RES.match(line)
        if m and cur:
            key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            us[key][cur[0]] = float(m.group(5))
            gbs[key][cur[0]] = float(m.group(6))
            cur = None

if not us:
    raise SystemExit("no results parsed")

if want:
    missing = [v for v in want if v not in variants]
    if missing:
        print(f"(not present in these logs: {', '.join(missing)})")
    variants = [v for v in want if v in variants]
    if not variants:
        raise SystemExit("none of the requested variants are present")

ref = variants[0]
w = 11
print(f"{'T':>6} {'W':>5} {'d':>2} | " + " ".join(f"{v:>{w}}" for v in variants)
      + " | " + " ".join(f"{v + '/' + ref:>13}" for v in variants[1:]))
print("-" * (6 + 6 + 3 + 3 + (w + 1) * len(variants) + 3 + 14 * (len(variants) - 1)))

for key in sorted(us, key=lambda k: (k[2], k[1], k[0])):
    T, W, d = key
    row = us[key]
    cells = " ".join(f"{row[v]:>{w}.1f}" if v in row else f"{'-':>{w}}" for v in variants)
    rel = " ".join(
        f"{row[ref] / row[v]:>13.3f}" if v in row and ref in row and row[v] else f"{'-':>13}"
        for v in variants[1:])
    print(f"{T:>6} {W:>5} {d:>2} | {cells} | {rel}")

print()
print(f"us per call. ratio > 1 means that variant is FASTER than {ref}.")
