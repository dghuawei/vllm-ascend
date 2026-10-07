"""Step 1: RETYPED index arithmetic of both branches (numpy, CPU-only).

No torch on this Mac; numpy index semantics are identical for the ops used
(abs, argsort, floor-div, clamp, gather, where).

Sources (read via `git show <ref>:<path>`; nothing imported):

ORIGIN/TEST
  origin/test:vllm_ascend/lora/fused_moe.py:190-222
      _recover_moe_lora_routing_allgather  -> recover_origin()
  origin/test:vllm_ascend/lora/punica_npu.py (add_lora_fused_moe, ~l.398-410)
      combined_idx fold                    -> fold_origin()
  origin/test:csrc/kernels/bgmv_shrink.cpp:66-71
      `if (reqLoRAIndex_ < 0) continue;`  -> a row is skipped iff the FOLDED
      combined index is < 0; there is NO upper-bound check, so a combined
      index >= stack rows is a silent OOB weight read.

ADDLORA_PATH
  addlora_path:vllm_ascend/lora/fused_moe.py:272-303
      _build_combined_lora_idx_allgather (driver -> torch.ops kernel)
  addlora_path:csrc/torch_binding.cpp:924-932 (keys for the kernel):
      keys = dest.to(fp32); keys.masked_fill_(keys < 0, float(n)); inv = argsort(keys)
      -> ALL inactive pairs share ONE trailing tie key n (safe only because
         the kernel emits -1 for every inactive pair).
  addlora_path:csrc/kernels/combined_lora_idx.cpp:6-27 (per-row semantics):
      p = inv[r]; token = p / top_k
      slot   = token < num_valid_tokens ? lora_indices[token] : -1
      expert = topk_ids[p] - first_expert_idx
      enabled = dest[p] >= 0 && slot >= 0 && 0 <= expert < num_experts
                && adapter_enabled[slot] != 0
      out[r] = enabled ? slot * num_experts + expert : -1
      -> combined_addlora()
  addlora_path:vllm_ascend/lora/fused_moe.py:305-368
      _build_combined_lora_idx_allgather_torch (reference impl, uses DISTINCT
      inactive keys n+arange instead of the single tie key n; asserted
      bit-identical to the kernel by tests/ut/lora/test_combined_idx_dedup.py)
      -> combined_addlora_ref()

Key structural differences (what the probe must expose):
  D1 abs(-1)=1 collides with the REAL destination 1 (origin/test) vs
     inactive keys pushed to n / n+arange (addlora_path).
  D2 origin/test has NO expert-range guard and NO EP local remap
     (expert ids stay global); addlora_path subtracts first_expert_idx and
     emits -1 for out-of-range / inactive rows.
  D3 origin/test folds (expert, lora) and never looks at dest again: a row
     sourced from an inactive pair still gets a non-negative combined index
     whenever its lora slot is active -> delta applied to garbage/remote rows.
  D4 clamp: origin/test clamps orig_token to token_lora_indices.size-1
     (overflow tokens silently get the LAST token's slot); addlora_path maps
     token >= num_valid to slot -1 (overflow rows silently skipped).
"""

import numpy as np


# --------------------------------------------------------------------------
# origin/test
# --------------------------------------------------------------------------
def recover_origin(expanded_row_idx, topk_ids, token_lora_indices, top_k,
                   tie="stable"):
    """Retyped from origin/test:vllm_ascend/lora/fused_moe.py:190-222.

        top_k = lora_context.top_k
        expanded = torch.abs(expanded_row_idx)
        inv_perm = torch.argsort(expanded)
        expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.long)
        orig_token = inv_perm // top_k
        token_lora_indices = lora_context.punica_wrapper.token_lora_indices
        orig_token = orig_token.clamp_(max=token_lora_indices.size - 1)
        lora_per_row = token_lora_indices[orig_token]

    torch.argsort default is stable=False; tie order is implementation
    defined. `tie` emulates the two extremes:
      "stable"     -> equal keys keep original pair order (best case)
      "unstable"   -> equal keys reversed (worst case a real sort may hit)
    """
    expanded = np.abs(expanded_row_idx.astype(np.int64))          # abs: -1 -> 1
    if tie == "stable":
        order = np.argsort(expanded, kind="stable")
    elif tie == "unstable":                                        # adversarial
        order = np.lexsort((-np.arange(expanded.size), expanded))
    else:
        raise ValueError(tie)
    inv_perm = order.astype(np.int64)                              # torch.argsort
    expert_per_row = topk_ids.reshape(-1)[inv_perm]
    orig_token = inv_perm // top_k                                 # floor div
    # clamp_(max=...) in place; NOTE: no min clamp, negatives impossible post-abs
    orig_token = np.minimum(orig_token, token_lora_indices.size - 1)
    lora_per_row = token_lora_indices[orig_token]
    return expert_per_row, lora_per_row


def fold_origin(expert_per_row, lora_per_row, adapter_enabled, num_experts):
    """Retyped from origin/test:vllm_ascend/lora/punica_npu.py add_lora_fused_moe:

        expert_idx = expert_ids.view(-1).to(torch.long)
        num_experts = lora_a_stacked[0].shape[1]      # = LOCAL experts under EP
        lora_idx_safe = token_lora_mapping.clamp(min=0)
        enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
        combined_idx = torch.where(enabled, lora_idx_safe * num_experts + expert_idx,
                                   torch.full_like(token_lora_mapping, -1))

    NO bound check on expert_idx, NO dest>=0 check, NO EP remap.
    bgmv_shrink.cpp:68 skips a row iff combined_idx[row] < 0; anything >= 0
    indexes a_flat[combined] unchecked (OOB read if >= stack rows).
    """
    lora_idx_safe = np.maximum(lora_per_row, 0)
    enabled = (lora_per_row >= 0) & (adapter_enabled[lora_idx_safe] != 0)
    return np.where(enabled,
                    lora_idx_safe * num_experts + expert_per_row,
                    np.full_like(lora_per_row, -1, dtype=np.int64))


def combined_origin(expanded_row_idx, topk_ids, token_lora_indices,
                    adapter_enabled, num_experts, top_k, tie="stable"):
    expert_per_row, lora_per_row = recover_origin(
        expanded_row_idx, topk_ids, token_lora_indices, top_k, tie=tie)
    return fold_origin(expert_per_row, lora_per_row, adapter_enabled, num_experts)


# --------------------------------------------------------------------------
# addlora_path
# --------------------------------------------------------------------------
def combined_addlora(expanded_row_idx, topk_ids, lora_indices, adapter_enabled,
                     first_expert_idx, num_experts, top_k, tie="stable"):
    """Retyped kernel semantics: torch_binding.cpp:924-932 (keys) +
    csrc/kernels/combined_lora_idx.cpp per-row body (header lines 6-27).

    Keys push ALL inactive pairs onto the single tie value n; per-row output
    does not depend on the tie order because every inactive pair emits -1.
    """
    dest = expanded_row_idx.reshape(-1).astype(np.int64)
    n = dest.size
    keys = dest.astype(np.float64)                 # aten fp32 (values exact here)
    keys[keys < 0] = float(n)                      # masked_fill_(keys<0, n)
    if tie == "stable":
        inv = np.argsort(keys, kind="stable")
    else:
        inv = np.lexsort((-np.arange(n), keys))
    num_valid = min(n // top_k, lora_indices.size)   # torch_binding.cpp:934-935
    topkf = topk_ids.reshape(-1).astype(np.int64)
    out = np.full(n, -1, dtype=np.int64)
    for ri in range(n):
        p = int(inv[ri])
        token = p // top_k
        slot = int(lora_indices[token]) if token < num_valid else -1   # D4
        expert = int(topkf[p]) - int(first_expert_idx)                # EP local
        enabled = (dest[p] >= 0                       # inactive pair -> -1
                   and slot >= 0
                   and 0 <= expert < num_experts      # expert-range guard
                   and adapter_enabled[slot] != 0)
        out[ri] = slot * num_experts + expert if enabled else -1
    return out


def combined_addlora_ref(expanded_row_idx, topk_ids, lora_indices, adapter_enabled,
                         first_expert_idx, num_experts, top_k):
    """Retyped _build_combined_lora_idx_allgather_torch
    (addlora_path:vllm_ascend/lora/fused_moe.py:305-368): DISTINCT inactive
    keys (n + arange), per-pair build, one final gather by inv.
    Must equal combined_addlora() (mirrors tests/ut/lora/test_combined_idx_dedup.py).
    """
    dest = expanded_row_idx.reshape(-1).astype(np.int64)
    n = dest.size
    active = dest >= 0
    keys = np.where(active,
                    dest,
                    n + np.arange(n, dtype=np.int64))              # :323-326
    inv = np.argsort(keys, kind="stable")
    num_tokens = min(n // top_k, lora_indices.size)             # :337
    # slot per pair: expand (NOT repeat_interleave) + pad with -1 (:339-349)
    slot_pairs = np.full(n, -1, dtype=np.int64)
    slot_pairs[:num_tokens * top_k] = np.repeat(
        lora_indices[:num_tokens], top_k)
    expert_pairs = topk_ids.reshape(-1).astype(np.int64) - int(first_expert_idx)  # :351-357
    slot_safe = np.maximum(slot_pairs, 0)                          # :359
    enabled = active & (slot_pairs >= 0) & (adapter_enabled[slot_safe] != 0)  # :360-363
    combined_pairs = np.where(enabled,
                              slot_safe * num_experts + expert_pairs, -1)      # :364-366
    # NOTE ref lacks the kernel's 0<=expert<num_experts guard; test inputs keep
    # topk ids valid so both agree. Gather by inv (:367).
    return combined_pairs[inv].astype(np.int64)
