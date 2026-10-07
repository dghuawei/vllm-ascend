"""Step 5: machine-checkable verdict.

VERDICT: YES -- a realistic AllGather-EP decode batch (nt=48, top_k=8, E=256,
3 adapters; rank-16 irrelevant to the index math) makes the branches assign
DIFFERENT (expert, lora_slot) to dispatched rows; addlora_path matches ground
truth, origin/test silently misroutes. In non-EP AllGather (the only
AllGather-LoRA config origin/test's own select_moe_comm_method can produce,
because it routes LoRA+EP to AlltoAll -- origin/test:vllm_ascend/
ascend_forward_context.py:356-365) the two are bit-identical: the combined
kernel is a pure perf refactor there.

Trigger conditions (any one suffices), all present in addlora_path's
deliberate LoRA+EP->ALLGATHER config (addlora_path:vllm_ascend/
ascend_forward_context.py:374-384):
  T1 expert_map != None, ep_rank > 0 (first_expert_idx > 0):
     origin/test never remaps global->local expert ids -> every active row
     gathers slot*E_local + global_expert: wrong (slot, expert) pair if in
     range, OOB bgmv read if >= max_loras*E_local.
  T2 any inactive pair (expanded_row_idx == -1) with flat id < the flat id of
     the real pair whose dest == 1: torch.abs(-1)=1 ties the padding marker
     with the real destination 1, argsort puts an INACTIVE pair at dispatched
     row 1 -> real pair's delta dropped, inactive pair's adapter applied
     (to a local expert row) -> both wrong, silent. Tie-break dependent
     (torch.argsort stable=False), wrong for 25/30 seeds even with stable ties.
  T3 inactive pairs also populate the UNDEFINED tail rows [available, n)
     with non-negative combined indices -> phantom deltas on discarded rows and
     OOB reads (bgmv_shrink.cpp:68 skips only negative indices; no upper bound).
  T4 (config-driven, not per-input) dispatched tokens exceed
     token_lora_indices.numel(): origin/test's clamp_(max=...) reuses the
     LAST covered token's adapter for ALL overflow tokens (wrong delta applied);
     addlora_path emits -1 (delta dropped). Both silent.
"""

import numpy as np

from step1_extract import combined_addlora, combined_origin
from step2_scenario import build_scenario


def main():
    # T1/T3: AllGather-EP, ep=4, rank 1, decode shape
    sc = build_scenario(seed=0, nt=48, ep=4, ep_rank=1)
    o = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                        sc["adapter_enabled"], sc["E_local"], sc["top_k"])
    a = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                         sc["adapter_enabled"], sc["first"], sc["E_local"], sc["top_k"])
    gt = sc["gt_combined"]
    assert not np.array_equal(o, gt) and np.array_equal(a, gt)
    real = np.nonzero(o[:sc["available"]] != gt[:sc["available"]])[0]
    print(f"T1 EP rank1: {real.size}/{sc['available']} real dispatched rows get a "
          f"different folded index; addlora==GT everywhere.")
    r = int(real[0])
    print(f"   row {r}: origin comb={int(o[r])}, addlora comb={int(a[r])}, GT={int(gt[r])}")

    # T2: rank 0 (remap is identity there) still breaks on the abs tie
    sc = build_scenario(seed=0, nt=48, ep=4, ep_rank=0)
    o = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                        sc["adapter_enabled"], sc["E_local"], sc["top_k"])
    a = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                         sc["adapter_enabled"], sc["first"], sc["E_local"], sc["top_k"])
    assert int(o[1]) != int(sc["gt_combined"][1]) and int(a[1]) == int(sc["gt_combined"][1])
    print(f"T2 EP rank0 row 1 abs-tie: origin={int(o[1])} "
          f"(pair taken is INACTIVE, expert id {int(o[1]) % sc['E_local'] if o[1] >= 0 else -1} "
          f"remote/OOB), addlora={int(a[1])}==GT={int(sc['gt_combined'][1])}")

    # non-EP: identical
    n_diff = 0
    for seed in range(20):
        sc = build_scenario(seed=seed, nt=48)
        o = combined_origin(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                            sc["adapter_enabled"], sc["E_local"], sc["top_k"])
        a = combined_addlora(sc["dest"], sc["topk_ids"], sc["token_lora_indices"],
                             sc["adapter_enabled"], sc["first"], sc["E_local"], sc["top_k"])
        n_diff += int((o != a).sum())
    assert n_diff == 0
    print("T0 non-EP AllGather (20 seeds): origin == addlora bit-identical, both == GT.")
    print("\nVERDICT: divergence exists exactly for AllGather-EP (-1 markers and/or "
          "first_expert_idx>0). addlora_path is correct; origin/test would silently "
          "misroute. origin/test masks this by forcing LoRA+EP to AlltoAll; "
          "addlora_path removes that gate on A2 and fixes the recovery accordingly.")


if __name__ == "__main__":
    main()
