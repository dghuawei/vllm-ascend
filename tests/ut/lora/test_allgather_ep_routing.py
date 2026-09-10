# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the EP-aware AllGather MoE LoRA routing recovery.

These tests build synthetic ``expanded_row_idx`` tensors that follow the
``npu_moe_init_routing_v2`` contract (row_idx_type=0):

- indexed by the ORIGINAL flat (token, k) pair,
- active pairs carry their compacted destination row in [0, available),
- inactive pairs (expert outside this rank's active range) are -1,
- tensor shapes are static (num_tokens * top_k rows; tail undefined).

They verify ``_recover_moe_lora_routing_allgather`` returns, per dispatched
row, the LOCAL expert id and LoRA slot, and that rows owned by other ranks
(including the undefined tail) are disabled via the -1 slot sentinel.
The non-EP case must stay byte-identical to the legacy recovery.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.lora.fused_moe import _recover_moe_lora_routing_allgather


def _make_context(*, top_k, use_ep, local_num_experts, token_lora_indices):
    return SimpleNamespace(
        top_k=top_k,
        use_ep=use_ep,
        local_num_experts=local_num_experts,
        punica_wrapper=SimpleNamespace(token_lora_indices=token_lora_indices),
    )


def _simulate_dispatch(topk_ids, first_expert, last_expert):
    """Simulate npu_moe_init_routing_v2 (row_idx_type=0, count mode).

    Active pairs are sorted by expert id (stable in pair order) into the
    compacted head of the output; inactive pairs become -1.
    Returns (dest, sorted_pair_order) where dest[p] is pair p's destination
    row (or -1) and sorted_pair_order lists the active pairs by destination.
    """
    flat_experts = topk_ids.reshape(-1).tolist()
    active_pairs = [p for p, e in enumerate(flat_experts) if first_expert <= e < last_expert]
    by_expert = sorted(active_pairs, key=lambda p: flat_experts[p])
    dest = [-1] * len(flat_experts)
    for row, p in enumerate(by_expert):
        dest[p] = row
    return torch.tensor(dest, dtype=torch.int32), by_expert


@pytest.fixture
def ep_rank_patch(monkeypatch):
    """Patch the lazy get_ep_group import target used by the recovery."""

    def _patch(rank_in_group):
        class _FakeGroup:
            pass

        fake = _FakeGroup()
        fake.rank_in_group = rank_in_group

        import vllm.distributed.parallel_state as ps

        monkeypatch.setattr(ps, "get_ep_group", lambda: fake)

    return _patch


@pytest.mark.parametrize("num_ranks", [2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ep_recovery_matches_bruteforce(num_ranks, seed, ep_rank_patch):
    torch.manual_seed(seed)
    num_tokens, top_k, experts_per_rank = 12, 2, 8
    num_experts = num_ranks * experts_per_rank
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    token_lora_indices = torch.randint(-1, 2, (64,), dtype=torch.long)

    for rank in range(num_ranks):
        first, last = rank * experts_per_rank, (rank + 1) * experts_per_rank
        ep_rank_patch(rank)
        ctx = _make_context(
            top_k=top_k,
            use_ep=True,
            local_num_experts=experts_per_rank,
            token_lora_indices=token_lora_indices,
        )
        dest, order = _simulate_dispatch(topk_ids, first, last)
        expert_per_row, lora_per_row = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)

        num_rows = num_tokens * top_k
        assert expert_per_row.shape == (num_rows,)
        assert lora_per_row.shape == (num_rows,)

        flat_experts = topk_ids.reshape(-1).tolist()
        for row, p in enumerate(order):
            # active row: local expert id and the token's lora slot
            assert int(expert_per_row[row]) == flat_experts[p] - first
            assert int(lora_per_row[row]) == int(token_lora_indices[p // top_k])
        for row in range(len(order), num_rows):
            # other-rank rows and the undefined tail must be disabled
            assert int(lora_per_row[row]) == -1
            assert 0 <= int(expert_per_row[row]) < experts_per_rank  # clamped, never OOB


@pytest.mark.parametrize("seed", [0, 1])
def test_no_ep_recovery_is_legacy_equivalent(seed, ep_rank_patch):
    torch.manual_seed(seed)
    num_tokens, top_k, num_experts = 9, 2, 16
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    token_lora_indices = torch.randint(-1, 2, (32,), dtype=torch.long)

    ctx = _make_context(
        top_k=top_k,
        use_ep=False,
        local_num_experts=num_experts,
        token_lora_indices=token_lora_indices,
    )
    dest, order = _simulate_dispatch(topk_ids, 0, num_experts)
    expert_per_row, lora_per_row = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)

    # legacy algorithm: abs + argsort over a complete permutation
    inv_perm = torch.argsort(torch.abs(dest))
    legacy_expert = topk_ids.reshape(-1)[inv_perm].to(torch.long)
    legacy_lora = token_lora_indices[(inv_perm // top_k).clamp_(max=31)]

    assert torch.equal(expert_per_row, legacy_expert)
    assert torch.equal(lora_per_row, legacy_lora)
    # every row is active without EP
    assert len(order) == num_tokens * top_k


def test_ep_all_rows_other_rank_stays_disabled(ep_rank_patch):
    """Rank whose local range receives zero pairs must disable every row."""
    num_tokens, top_k = 6, 2
    topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int64)  # all expert 0
    token_lora_indices = torch.zeros((16,), dtype=torch.long)  # all lora slot 0
    ep_rank_patch(3)  # rank 3 owns experts [24, 32): none active
    ctx = _make_context(
        top_k=top_k,
        use_ep=True,
        local_num_experts=8,
        token_lora_indices=token_lora_indices,
    )
    dest, order = _simulate_dispatch(topk_ids, 24, 32)
    assert not order  # nothing dispatched to this rank
    expert_per_row, lora_per_row = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)
    assert torch.all(lora_per_row == -1)
    assert torch.all(expert_per_row >= 0) and torch.all(expert_per_row < 8)


def test_prefers_split_lora_indices_when_present(ep_rank_patch):
    """With flash-comm off, prepare splits the batch per TP rank and stores
    the matching shard slots in split_lora_indices (padded with -1); the
    recovery must index those, not the global token_lora_indices."""
    num_tokens, top_k = 8, 2
    topk_ids = torch.randint(0, 16, (num_tokens, top_k), dtype=torch.int64)
    global_indices = torch.randint(0, 2, (32,), dtype=torch.long)
    # rank 1 shard: tokens 4..8 of the global batch, distinct slots
    shard_indices = torch.randint(0, 2, (num_tokens,), dtype=torch.long)
    ep_rank_patch(0)
    ctx = _make_context(
        top_k=top_k,
        use_ep=False,
        local_num_experts=16,
        token_lora_indices=global_indices,
    )
    ctx.split_lora_indices = shard_indices
    dest, order = _simulate_dispatch(topk_ids, 0, 16)
    expert_per_row, lora_per_row = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)
    for row, p in enumerate(order):
        assert int(lora_per_row[row]) == int(shard_indices[p // top_k])


def test_ep_with_split_indices_uses_shard_slots(ep_rank_patch):
    """EP + token-split prepare: shard slots still resolve per dispatched row,
    non-local rows disabled via -1."""
    num_tokens, top_k, experts_per_rank = 8, 2, 8
    topk_ids = torch.randint(0, 16, (num_tokens, top_k), dtype=torch.int64)
    shard_indices = torch.randint(0, 2, (num_tokens,), dtype=torch.long)
    ep_rank_patch(1)
    ctx = _make_context(
        top_k=top_k,
        use_ep=True,
        local_num_experts=experts_per_rank,
        token_lora_indices=torch.zeros((32,), dtype=torch.long),
    )
    ctx.split_lora_indices = shard_indices
    first, last = experts_per_rank, 2 * experts_per_rank
    dest, order = _simulate_dispatch(topk_ids, first, last)
    expert_per_row, lora_per_row = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)
    flat_experts = topk_ids.reshape(-1).tolist()
    for row, p in enumerate(order):
        assert int(expert_per_row[row]) == flat_experts[p] - first
        assert int(lora_per_row[row]) == int(shard_indices[p // top_k])
    for row in range(len(order), num_tokens * top_k):
        assert int(lora_per_row[row]) == -1


def test_shapes_are_value_independent(ep_rank_patch):
    """Graph-capturability: outputs must depend only on input shapes."""
    ep_rank_patch(0)
    token_lora_indices = torch.zeros((16,), dtype=torch.long)
    ctx = _make_context(
        top_k=2,
        use_ep=True,
        local_num_experts=8,
        token_lora_indices=token_lora_indices,
    )
    shapes = set()
    for fill in (0, 5, 15):  # different routing patterns, same shape
        topk_ids = torch.full((8, 2), fill, dtype=torch.int64)
        dest, _ = _simulate_dispatch(topk_ids, 0, 16)
        expert_per_row, lora_per_row = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)
        shapes.add((tuple(expert_per_row.shape), tuple(lora_per_row.shape)))
    assert len(shapes) == 1


def _legacy_chain(ctx, dest, topk_ids):
    """Reference: the historical recovery + build composition."""
    import vllm_ascend.ops.fused_moe.token_dispatcher  # import order
    from vllm_ascend.lora.fused_moe import (
        _recover_moe_lora_routing_allgather,
    )
    from vllm_ascend.lora.punica_npu import build_combined_lora_idx

    expert, slot = _recover_moe_lora_routing_allgather(ctx, dest, topk_ids)
    return build_combined_lora_idx(slot, expert, ctx.adapter_enabled, ctx.num_experts)


@pytest.mark.parametrize("num_ranks", [2, 4])
@pytest.mark.parametrize("use_ep", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_merged_combined_idx_matches_legacy_chain(num_ranks, use_ep, seed, ep_rank_patch):
    from vllm_ascend.lora.fused_moe import _build_combined_lora_idx_allgather

    torch.manual_seed(seed)
    experts_per_rank = 8
    num_experts = num_ranks * experts_per_rank
    rank = 1 if use_ep else 0
    ep_rank_patch(rank)
    num_tokens, top_k = 24, 2
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    token_lora_indices = torch.randint(-1, 2, (64,), dtype=torch.long)
    # include a disabled adapter on some runs
    adapter_enabled = torch.ones(3, dtype=torch.int32)
    if seed % 2 == 0:
        adapter_enabled[1] = 0

    first, last = rank * experts_per_rank, (rank + 1) * experts_per_rank
    ctx = _make_context(
        top_k=top_k,
        use_ep=use_ep,
        local_num_experts=experts_per_rank,
        token_lora_indices=token_lora_indices,
    )
    ctx.adapter_enabled = adapter_enabled
    ctx.num_experts = experts_per_rank if use_ep else num_experts
    # the merged builder reads the stacks for num_experts
    ctx.w13_lora_a_stacked = (torch.empty(2, ctx.num_experts, 4, 8),)

    flat = topk_ids.reshape(-1)
    act = (flat >= first) & (flat < last) if use_ep else torch.ones_like(flat, dtype=torch.bool)
    order = act.nonzero().squeeze(1)
    dest = torch.full((flat.numel(),), -1, dtype=torch.int32)
    dest[order] = torch.randperm(order.numel(), dtype=torch.int32)

    merged = _build_combined_lora_idx_allgather(ctx, dest, topk_ids)
    legacy = _legacy_chain(ctx, dest, topk_ids)
    assert torch.equal(merged, legacy), f"mismatch (max diff at {(merged != legacy).nonflatten().nonzero()[:5] if hasattr((merged != legacy), 'nonflatten') else ''})"
