"""Step 2: realistic scenario builder + ground truth.

Models the npu_moe_init_routing_v2 contract used by BOTH branches' AllGather
dispatcher (origin/test:vllm_ascend/ops/fused_moe/token_dispatcher.py:424-434
passes active_expert_range=[first_expert_idx, last_expert_idx]):

  - flat pair p = token*top_k + k, expert e = topk_ids.reshape(-1)[p]
  - EP: pairs whose expert is outside the rank's contiguous local range
    [first, first+E_local) are INACTIVE -> expanded_row_idx[p] = -1
  - active pairs are sorted by expert into compacted rows [0, available)
  - rows [available, num_pairs) of the static tensors are UNDEFINED
    (the base GMM computes nothing there; npu_moe_token_unpermute drops them)

DeepSeek-V4-Flash: no local config.json found (name appears only in
.github/workflows/misc/model_dataset_list.json); E=256 taken from the
DeepSeek-V3-family default n_routed_experts=256 (task fallback).

Ground truth per dispatched row r:
  r < available: row r provably holds pair p = order[r]
      true_expert_local = topk[p] - first
      true_slot         = token_lora_indices[p // top_k]
      true_combined     = slot*E_local + expert_local  iff slot>=0 and
                          adapter_enabled[slot], else -1
  r >= available: row holds NO real pair -> correct behaviour is a zero
      delta, i.e. combined == -1 (anything >= 0 is applied to a garbage row;
      bgmv_shrink.cpp has no upper bound, so >= stack_rows is also OOB read).
"""

import numpy as np


def build_scenario(seed, nt, top_k=8, E=256, ep=1, ep_rank=0, max_loras=3,
                   p_no_adapter=0.30, disable_adapter=None, tli_size=None,
                   num_pairs=None):
    """Returns dict with everything step1/step3 need.

    tli_size: token_lora_indices.numel() -- punica sizes it to
      max_num_batched_tokens; pass < nt to emulate the AllGather row-count
      overflow the clamp guards (probe question b).
    """
    rng = np.random.default_rng(seed)
    # per-token top_k DISTINCT experts (real router behaviour)
    topk = np.stack([rng.choice(E, size=top_k, replace=False)
                     for _ in range(nt)]).astype(np.int64)
    num_pairs = num_pairs if num_pairs is not None else nt * top_k

    E_local = E // ep
    first = ep_rank * E_local

    token_lora = np.where(rng.random(nt) < p_no_adapter, -1,
                          rng.integers(0, max_loras, nt)).astype(np.int64)
    tli_size = tli_size if tli_size is not None else nt
    token_lora_indices = token_lora[:tli_size].copy()  # punica tensor (may be smaller)

    adapter_enabled = np.ones(max_loras, dtype=np.int32)
    if disable_adapter is not None:
        adapter_enabled[disable_adapter] = 0

    flat_e = topk.reshape(-1)
    n = min(num_pairs, topk.size)
    flat_e = flat_e[:n]
    active = (flat_e >= first) & (flat_e < first + E_local)
    dest = np.full(n, -1, dtype=np.int64)
    # expert-sorted, compacted destinations for active pairs (stable by pair id)
    act_pairs = np.nonzero(active)[0]
    order = act_pairs[np.argsort(flat_e[act_pairs], kind="stable")]
    dest[order] = np.arange(order.size, dtype=np.int64)
    available = int(order.size)

    # ---------------- ground truth ----------------
    gt_expert_local = np.full(n, -1, dtype=np.int64)   # per row, local expert id
    gt_slot = np.full(n, -1, dtype=np.int64)           # per row, adapter slot
    gt_combined = np.full(n, -1, dtype=np.int64)       # per row, folded index
    for r in range(available):
        p = int(order[r])
        e_loc = int(flat_e[p]) - first
        tok = p // top_k
        slot = int(token_lora[tok]) if tok < tli_size else -1  # true mapping:
        #   a real dispatched row whose token is NOT covered by the (too small)
        #   lora tensor has no known slot -> treated as unknown (-1) for BOTH
        #   branches; probe (b) then asks what each impl *actually* applies.
        gt_expert_local[r] = e_loc
        gt_slot[r] = slot
        if slot >= 0 and adapter_enabled[slot] != 0:
            gt_combined[r] = slot * E_local + e_loc
    # rows >= available: -1 (must contribute nothing)

    return dict(nt=nt, top_k=top_k, E=E, ep=ep, ep_rank=ep_rank,
                E_local=E_local, first=first, max_loras=max_loras,
                topk_ids=topk, dest=dest, available=available,
                token_lora=token_lora, token_lora_indices=token_lora_indices,
                adapter_enabled=adapter_enabled,
                gt_expert_local=gt_expert_local, gt_slot=gt_slot,
                gt_combined=gt_combined, order=order, num_pairs=n)
