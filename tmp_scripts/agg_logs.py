"""Aggregate every evalscope performance_summary.txt under scripts/logs/.
Rows = runs (experiment / timestamp / model), cols = median (p50) metrics."""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/vllm-workspace/scripts/logs"
WANT = ["Latency (s)", "TTFT (ms)", "TPOT (ms)", "Output Tokens", "Input Tokens"]


def parse(path):
    out = {}
    for line in open(path, errors="replace"):
        if "│" not in line:
            continue
        cells = [c.strip() for c in line.split("│")]
        if "Avg Output Rate" in line and len(cells) > 2:
            m = re.search(r"([\d.]+)", cells[2])
            if m:
                out["rate"] = float(m.group(1))
        if "Total Test Time" in line and len(cells) > 2:
            m = re.search(r"([\d.]+)", cells[2])
            if m:
                out["test_s"] = float(m.group(1))
        # Per-request rows: | conc | rate | metric | avg | p50 | p99 | max |
        for w in WANT:
            if len(cells) >= 8 and cells[3] == w:
                try:
                    out[w] = float(cells[5])  # p50
                except ValueError:
                    pass
    return out


rows = []
for exp in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, exp)
    if not os.path.isdir(d):
        continue
    for dirpath, _, files in os.walk(d):
        if "performance_summary.txt" not in files:
            continue
        model = os.path.basename(dirpath)
        ts = os.path.basename(os.path.dirname(dirpath))
        r = parse(os.path.join(dirpath, "performance_summary.txt"))
        if r:
            r["exp"], r["model"], r["ts"] = exp, model, ts
            rows.append(r)

if not rows:
    raise SystemExit("no performance_summary.txt found under " + ROOT)

hdr = f"{'experiment':<62} {'model':<14} {'TTFT':>8} {'TPOT':>7} {'Lat':>8} {'out':>6} {'in':>6} {'rate':>7}"
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x["exp"], x["model"])):
    exp = r["exp"][:62]
    print(f"{exp:<62} {r['model'][:14]:<14} "
          f"{r.get('TTFT (ms)', float('nan')):>8.0f} "
          f"{r.get('TPOT (ms)', float('nan')):>7.2f} "
          f"{r.get('Latency (s)', float('nan')):>8.1f} "
          f"{r.get('Output Tokens', float('nan')):>6.0f} "
          f"{r.get('Input Tokens', float('nan')):>6.0f} "
          f"{r.get('rate', float('nan')):>7.1f}")
print("\np50 (median) per request; rate = Avg Output Rate tok/s (whole run).")
