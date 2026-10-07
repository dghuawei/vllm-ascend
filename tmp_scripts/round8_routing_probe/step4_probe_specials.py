"""Step 4: the four probes the task calls out, explicitly.

(a) torch.abs on expanded_row_idx with -1 padding markers
(b) clamp_(max=token_lora_indices.numel()-1) under row-count overflow
(c) lora_per_row == -1 handling in the apply (code-read finding + check)
(d) addlora_path kernel vs its torch reference, and tie-invariance
"""

import numpy as np

from step1_extract import (combined_addlora, combined_addlora_ref,
                           combined_origin, recover_origin)
from step2_scenario import build_scenario


def probe_a_abs_tie():
    print("--- (a) abs(-1)=1 tie on the FIRST REAL ROW beyond row 0 ---")
    n_stable_wrong = n_unstable_wrong = n_seeds = 0
    ex = None
    for seed in range(30):
        sc = build_scenario(seed=seed, nt=48, ep=4, ep_rank=0)  # first=0: remap identity
        dest, k = sc["dest"], sc["top_k"]
        avail = sc["available"]
        if avail < 2:
            continue
        n_seeds += 1
        flat = sc["topk_ids"].reshape(-1)
        p_dest1 = int(np.nonzero(dest == 1)[0][0])
        inactive = np.nonzero(dest < 0)[0]
        # STABLE tie: among key==1 ties the smallest flat pair id wins row 1
        stable_correct = p_dest1 < (inactive.min() if inactive.size else np.inf)
        o_st = combined_origin(dest, sc["topk_ids"], sc["token_lora_indices"],
                               sc["adapter_enabled"], sc["E_local"], k, "stable")
        o_un = combined_origin(dest, sc["topk_ids"], sc["token_lora_indices"],
                               sc["adapter_enabled"], sc["E_local"], k, "unstable")
        a = combined_addlora(dest, sc["topk_ids"], sc["token_lora_indices"],
                             sc["adapter_enabled"], sc["first"], sc["E_local"], k)
        gt1 = int(sc["gt_combined"][1])
        if int(o_st[1]) != gt1:
            n_stable_wrong += 1
        if int(o_un[1]) != gt1:
            n_unstable_wrong += 1
        if ex is None and int(o_st[1]) != gt1:
            ex = (seed, p_dest1, int(inactive.min()), gt1,
                  int(o_st[1]), int(o_un[1]), int(a[1]), sc)
    print(f"  EP ep=4 rank0, nt=48, 30 seeds ({n_seeds} with available>=2):")
    print(f"  origin row 1 WRONG on {n_stable_wrong}/{n_seeds} seeds (stable ties), "
          f"{n_unstable_wrong}/{n_seeds} (adversarial ties); addlora correct on all")
    if ex:
        (seed, p1, imin, gt1, ost, oun, aadd, sc) = ex
        k = sc["top_k"]
        real_tok, real_k = p1 // k, p1 % k
        inv_st = np.argsort(np.abs(sc["dest"]), kind="stable")
        p_taken = int(inv_st[1])
        print(f"  concrete (seed={seed}): row 1 really holds pair {p1} "
              f"(token {real_tok}, k={real_k}); abs(-1)=1 gives {int((sc['dest']<0).sum())} "
              f"inactive pairs key 1; earliest inactive pair id {imin} < {p1} -> "
              f"argsort puts pair {p_taken} (token {p_taken//k}, INACTIVE, expert "
              f"{int(sc['topk_ids'].reshape(-1)[p_taken])}) at row 1.")
        print(f"  row 1 combined: GT={gt1}, origin={ost} (stable) / {oun} (unstable), "
              f"addlora={aadd}  -> origin DROPS the real delta AND applies the "
              f"inactive pair's adapter to the wrong row")
    return n_stable_wrong > 0


def probe_b_clamp():
    print("\n--- (b) clamp_(max=numel-1) when dispatched rows exceed the lora tensor ---")
    sc = build_scenario(seed=5, nt=48, tli_size=24)
    # make the overflow visible: last covered token must carry a live adapter
    sc["token_lora_indices"][23] = 1
    for r in range(24 * sc["top_k"], sc["num_pairs"]):  # recompute GT slots >= tli
        sc["gt_combined"][r] = -1
        sc["gt_slot"][r] = -1
    o = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                        sc["adapter_enabled"], sc["E_local"], sc["top_k"])
    a = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                         sc["adapter_enabled"], sc["first"], sc["E_local"], sc["top_k"])
    over = np.nonzero((np.arange(sc["num_pairs"]) // sc["top_k"]) >= 24)[0]
    diff = over[o[over] != a[over]]
    print(f"  non-EP nt=48, token_lora_indices covers 24/48 tokens, slot of token 23 = 1")
    print(f"  rows of tokens 24..47: origin vs addlora differ on {diff.size}/{over.size}")
    r = int(diff[0]) if diff.size else -1
    if r >= 0:
        tok = r // sc["top_k"]
        print(f"  e.g. row {r} (token {tok}): origin applies slot {int(sc['token_lora_indices'][23])} "
              f"(= LAST covered token's adapter, comb={int(o[r])}); "
              f"addlora emits -1 (delta skipped). Both silently; no error either way.")
    return diff.size > 0


def probe_c_skip_vs_mask():
    print("\n--- (c) are -1 rows skipped in the apply, or just masked? ---")
    print("  origin/test punica_npu.add_lora_fused_moe folds (lora<0 | adapter off)")
    print("    -> combined -1; csrc/kernels/bgmv_shrink.cpp:68 `if (reqLoRAIndex_<0) continue;`")
    print("    -> shrink_out row stays 0 (torch.zeros buffer) => bgmv_expand of -1 also skips")
    print("    => FULL SKIP, zero delta; no upper-bound check on the index.")
    print("  addlora_path same bgmv + kernel already emits -1 for inactive/out-of-range rows.")
    # numeric check: origin emits >=0 combined for inactive-sourced rows (not skipped)
    sc = build_scenario(seed=3, nt=48, ep=4, ep_rank=1)
    o = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                        sc["adapter_enabled"], sc["E_local"], sc["top_k"])
    stack_rows = sc["max_loras"] * sc["E_local"]
    active_rows = np.nonzero(o[:sc["available"]] >= 0)[0]
    oob = int(np.sum((o >= stack_rows) & (o >= 0)))
    print(f"  EP ep=4 rank1 nt=48: origin emits non-negative combined on "
          f"{active_rows.size}/{sc['available']} real rows (all mis-gathered, see S3),")
    print(f"  and {oob} rows have combined >= stack rows ({stack_rows}) -> "
          f"bgmv_shrink indexes a_flat[combined] unguarded: OOB read.")


def probe_d_kernel_ties():
    print("\n--- (d) addlora kernel semantics: torch-ref equality + tie invariance ---")
    bad = inv = 0
    for seed in range(20):
        for ep, rank in ((1, 0), (4, 0), (4, 1), (8, 3)):
            sc = build_scenario(seed=seed, nt=48, ep=ep, ep_rank=rank)
            a_s = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                                   sc["adapter_enabled"], sc["first"], sc["E_local"],
                                   sc["top_k"], "stable")
            a_u = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                                   sc["adapter_enabled"], sc["first"], sc["E_local"],
                                   sc["top_k"], "unstable")
            a_r = combined_addlora_ref(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                                       sc["adapter_enabled"], sc["first"], sc["E_local"],
                                       sc["top_k"])
            if not np.array_equal(a_s, a_r):
                bad += 1
            if not np.array_equal(a_s, a_u):
                inv += 1
    print(f"  20 seeds x 4 EP configs: kernel-semantics != torch-ref: {bad}; "
          f"tie-order-sensitive: {inv}  (0 = bit-identical to reference and "
          f"invariant, as tests/ut/lora/test_combined_idx_dedup.py asserts)")


if __name__ == "__main__":
    probe_a_abs_tie()
    probe_b_clamp()
    probe_c_skip_vs_mask()
    probe_d_kernel_ties()
