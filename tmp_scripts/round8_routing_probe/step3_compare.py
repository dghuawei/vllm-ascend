"""Step 3: differential comparison of both branches vs ground truth.

Scenarios: DeepSeek-V4-Flash-like (E=256, top_k=8), decode shapes
(nt=48/64/128), 3 adapters (rank-16 is irrelevant to the index math -- the
folded index does not depend on rank), AllGather-EP (-1 inactive pairs)
and TP-only/non-EP variants, plus the clamp-overflow probe.

Run:  python3 step3_compare.py
"""

import numpy as np

from step1_extract import (combined_addlora, combined_addlora_ref,
                           combined_origin, recover_origin)
from step2_scenario import build_scenario


def classify(impl_comb, gt_comb, gt_avail, stack_rows, impl_pair_per_row=None):
    """Count diff classes + return list of (row, detail)."""
    diffs = []
    counters = dict(wrong_delta=0, dropped=0, applied_on_nothing=0,
                    oob_read=0, tail_garbage=0)
    n = impl_comb.size
    for r in range(n):
        ic, gc = int(impl_comb[r]), int(gt_comb[r])
        if ic == gc:
            continue
        kind = None
        if r < gt_avail:
            if gc >= 0 and ic >= 0:
                kind = "wrong_delta"          # different LoRA applied to real row
            elif gc >= 0 and ic < 0:
                kind = "dropped"              # real row's delta silently lost
            else:  # gc < 0: GT says no delta (no adapter / unknown token)
                kind = "applied_on_nothing"   # phantom delta on real row
        else:
            kind = "tail_garbage"             # row has no real pair at all
        if ic >= stack_rows and ic >= 0:
            counters["oob_read"] += 1         # bgmv reads a_flat[ic] unchecked
        counters[kind] += 1
        pair = ""
        if impl_pair_per_row is not None:
            e, s = impl_pair_per_row[r]
            pair = f" origin_pair=(expert_id={e},slot={s})"
        diffs.append((r, ic, gc, f"{kind}{pair}"))
    return counters, diffs


def origin_pair_per_row(sc, tie):
    """(expert_per_row, lora_per_row) exactly as origin/test feeds the apply."""
    return recover_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                          sc["top_k"], tie=tie)


def run_one(name, sc, stack_rows):
    ep, first, E_loc = sc["ep"], sc["first"], sc["E_local"]
    a = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                         sc["adapter_enabled"], first, E_loc, sc["top_k"])
    a_ref = combined_addlora_ref(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                                 sc["adapter_enabled"], first, E_loc, sc["top_k"])
    assert np.array_equal(a, a_ref), f"{name}: kernel-semantics != torch-ref"
    o_st = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                           sc["adapter_enabled"], E_loc, sc["top_k"], tie="stable")
    o_un = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                           sc["adapter_enabled"], E_loc, sc["top_k"], tie="unstable")
    gt = sc["gt_combined"]
    avail = sc["available"]

    co, do = classify(o_st, gt, avail, stack_rows)
    cu, _ = classify(o_un, gt, avail, stack_rows)
    ca, da = classify(a, gt, avail, stack_rows)
    # branch-vs-branch
    ab = np.nonzero(o_st != a)[0]
    ab_u = np.nonzero(o_un != a)[0]
    exp_p, exp_s = origin_pair_per_row(sc, "stable")

    print(f"\n== {name}: nt={sc['nt']} top_k={sc['top_k']} E={sc['E']} "
          f"ep={ep} rank={sc['ep_rank']} (first={first}, E_local={E_loc}) "
          f"active={avail}/{sc['num_pairs']}")
    print(f"   origin(stable) vs GT : {co}")
    print(f"   origin(unstable) vs GT: {cu}")
    print(f"   addlora       vs GT : {ca}")
    print(f"   origin(stable) != addlora on {ab.size} rows; "
          f"origin(unstable) != addlora on {ab_u.size} rows")
    if ab.size:
        e = exp_p.reshape(-1) if exp_p.ndim else exp_p
        rows = ab[:6]
        for r in rows:
            print(f"     row {r:5d}: origin comb={o_st[r]:5d} "
                  f"(expert_id={int(e[r])}, slot={int(exp_s[r])}) "
                  f"addlora comb={a[r]:5d} gt={gt[r]:5d} "
                  f"gt(exp={int(sc['gt_expert_local'][r])},slot={int(sc['gt_slot'][r])})")
    return dict(name=name, ab=int(ab.size), ab_u=int(ab_u.size), co=co, cu=cu, ca=ca)


def main():
    results = []
    # ---- S1: non-EP AllGather (origin/test's only AG-LoRA config) ----
    for nt in (48, 64, 128):
        sc = build_scenario(seed=nt, nt=nt)
        results.append(run_one(f"S1 non-EP nt={nt}", sc,
                               stack_rows=sc["max_loras"] * sc["E_local"]))
    sc = build_scenario(seed=7, nt=64, disable_adapter=1)
    results.append(run_one("S1b non-EP, adapter#1 disabled", sc,
                           3 * sc["E_local"]))

    # ---- S2: AllGather-EP, ep=4, rank 0 (expert remap is identity) ----
    for nt in (48, 64, 128):
        sc = build_scenario(seed=nt, nt=nt, ep=4, ep_rank=0)
        results.append(run_one(f"S2 EP ep4 rank0 nt={nt}", sc,
                               3 * sc["E_local"]))

    # ---- S3: AllGather-EP, ep=4, rank 1 (first=64: origin never remaps) ----
    for nt in (48, 64, 128):
        sc = build_scenario(seed=nt, nt=nt, ep=4, ep_rank=1)
        results.append(run_one(f"S3 EP ep4 rank1 nt={nt}", sc,
                               3 * sc["E_local"]))

    # ---- S4: EP ep=8 rank 3, longer decode ----
    sc = build_scenario(seed=99, nt=128, ep=8, ep_rank=3)
    results.append(run_one("S4 EP ep8 rank3 nt=128", sc, 3 * sc["E_local"]))

    # ---- S5: clamp overflow -- lora tensor smaller than dispatched tokens ----
    sc = build_scenario(seed=5, nt=48, tli_size=24)
    results.append(run_one("S5 clamp overflow (tli covers 24/48 tokens)", sc,
                           3 * sc["E_local"]))

    print("\n================ SUMMARY ================")
    for r in results:
        print(f"{r['name']:45s} origin!=addlora: {r['ab']:5d}/{r['ab_u']:5d} "
              f"(stable/unstable)  origin_bad={sum(r['co'].values())} "
              f"addlora_bad={sum(r['ca'].values())}")


if __name__ == "__main__":
    main()
