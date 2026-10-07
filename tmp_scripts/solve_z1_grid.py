#!/usr/bin/env python3
"""Invert the host-side grid planning in csrc/kernels/add_lora_fused.cpp
(PlanGrid + the rankBlock logic) to find which (batch, nSlices, R, aivNum)
produce the Block Num values observed for add_lora_z1 in the b3 trace."""

def plan_z1(batch, nSlices, R, aiv):
    targetGroups = max(1, (aiv + batch - 1) // batch)
    targetGroups = min(targetGroups, R)
    rankBlock = (R + targetGroups - 1) // targetGroups
    rankBlock = ((rankBlock + 7) // 8) * 8
    rankBlock = min(rankBlock, R)
    groups = (R + rankBlock - 1) // rankBlock
    units = nSlices * batch * groups
    upc = max(1, (units + aiv - 1) // aiv)
    grid = (units + upc - 1) // upc
    return grid, units, groups, rankBlock

TARGETS = (28, 34)
for aiv in (32, 40, 48, 64):
    print(f"--- aivNum = {aiv} ---")
    for tgt in TARGETS:
        hits = []
        for R in (16, 32, 64):
            for nS in (1, 2, 3, 4):
                for batch in range(1, 4097):
                    g, units, groups, rb = plan_z1(batch, nS, R, aiv)
                    if g == tgt:
                        hits.append((batch, nS, R, groups, rb, units))
        # keep it readable: only the small-batch solutions
        small = [h for h in hits if h[0] <= 600]
        print(f"  grid={tgt}: {len(hits)} solutions; batch<=600 ->")
        for h in small[:14]:
            print(f"      batch={h[0]:5d} nSlices={h[1]} R={h[2]:2d} groups={h[3]} rankBlock={h[4]:2d} units={h[5]}")
