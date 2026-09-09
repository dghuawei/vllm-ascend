# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-layer combined_idx dedup between the w13 and w2 MoE
LoRA applies: the identical gather index must be built once per layer,
shared by both applies, cleared after w2, and any stale stash must fall
back to a correct rebuild."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401  (loads torch.ops._C_ascend)
import vllm_ascend.lora.punica_npu as punica_mod
from vllm_ascend.lora.fused_moe import (
    moe_lora_apply_w13,
    moe_lora_apply_w2,
    reset_lora_indices,
)

DEV = "npu:0"
W, E, R, H, I = 2, 4, 8, 32, 16  # loras, local experts, rank, hidden, intermediate


def _make_context(rows, seed=0, stale_idx=None):
    torch.manual_seed(seed)
    stacks_w13 = (torch.zeros(W, E, R, H), torch.zeros(W, E, 2 * I, R))
    stacks_w2 = (torch.zeros(W, E, R, I), torch.zeros(W, E, H, R))
    ctx = SimpleNamespace(
        punica_wrapper=Mock(),
        adapter_enabled=torch.ones(W + 1, dtype=torch.int32),
        w13_lora_a_stacked=stacks_w13,
        w13_lora_b_stacked=(stacks_w13[1],),
        w2_lora_a_stacked=stacks_w2,
        w2_lora_b_stacked=(stacks_w2[1],),
        fully_sharded=False,
        top_k=2,
    )
    if stale_idx is not None:
        ctx.combined_lora_idx = stale_idx
    return ctx


def _routing(rows):
    mapping = torch.randint(-1, W, (rows,), dtype=torch.long)
    experts = torch.randint(0, E, (rows,), dtype=torch.long)
    return experts, mapping


def test_w13_and_w2_share_one_combined_idx():
    rows = 12
    ctx = _make_context(rows)
    routing = _routing(rows)
    captured = []

    def record(y, x, **kw):
        captured.append(kw.get("combined_idx"))

    ctx.punica_wrapper.add_lora_fused_moe.side_effect = record
    with patch.object(punica_mod, "build_combined_lora_idx", wraps=punica_mod.build_combined_lora_idx) as b:
        moe_lora_apply_w13(ctx, gate_up_out=Mock(), hidden_states=Mock(), lora_routing=routing)
        moe_lora_apply_w2(ctx, down_out=Mock(), silu_out=Mock(), lora_routing=routing)
        assert b.call_count == 1, "combined_idx must be built exactly once per layer"

    assert captured[0] is not None and captured[1] is not None
    assert captured[0] is captured[1], "w13 and w2 must receive the same tensor object"
    # expected content matches the historical formula
    experts, mapping = routing
    safe = mapping.clamp(min=0)
    enabled = (mapping >= 0) & ctx.adapter_enabled[safe].bool()
    ref = torch.where(enabled, safe * E + experts, torch.full_like(mapping, -1))
    assert torch.equal(captured[0], ref)
    # cleared after w2
    assert not hasattr(ctx, "combined_lora_idx")


def test_stale_stash_falls_back_to_rebuild():
    rows = 10
    ctx = _make_context(rows, stale_idx=torch.full((rows + 7,), -1, dtype=torch.long))
    routing = _routing(rows)
    captured = []

    def record(y, x, **kw):
        captured.append(kw.get("combined_idx"))

    ctx.punica_wrapper.add_lora_fused_moe.side_effect = record
    # w13 not called: stale stash must not be consumed
    moe_lora_apply_w2(ctx, down_out=Mock(), silu_out=Mock(), lora_routing=routing)
    assert captured[0] is None, "shape-mismatched stash must not reach w2 (None -> punica rebuilds)"
    ctx.punica_wrapper.add_lora_fused_moe.assert_called_once()


def test_empty_rank_guards_consistent():
    ctx = _make_context(0)
    routing = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    moe_lora_apply_w13(ctx, gate_up_out=Mock(), hidden_states=Mock(), lora_routing=routing)
    assert not hasattr(ctx, "combined_lora_idx"), "early-return must not set the stash"
    moe_lora_apply_w2(ctx, down_out=Mock(), silu_out=Mock(), lora_routing=routing)
    ctx.punica_wrapper.add_lora_fused_moe.assert_not_called()


def test_reset_clears_combined_field():
    ctx = SimpleNamespace(combined_lora_idx=torch.zeros(3), split_lora_indices=torch.zeros(3))
    reset_lora_indices(ctx)
    assert not hasattr(ctx, "combined_lora_idx")
    assert not hasattr(ctx, "split_lora_indices")


@pytest.mark.parametrize("rows", [8, 64])
def test_numeric_pair_matches_reference(rows):
    """End-to-end w13+w2 through the shared index on NPU vs einsum."""
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    torch.manual_seed(rows)
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._use_moe_gmm_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._no_lora_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._use_add_lora_cpu = torch.tensor(True, dtype=torch.bool)
    wrapper._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
    from vllm_ascend.lora.lora_ops import bgmv_expand_slice, bgmv_shrink

    wrapper.bgmv_shrink = bgmv_shrink
    wrapper.bgmv_expand_slice = bgmv_expand_slice

    x = (torch.randn(rows, H, device=DEV) * 0.3).bfloat16()
    silu = (torch.randn(rows, I, device=DEV) * 0.3).bfloat16()
    # per-slice stacks, mirroring the production packed layout (one A per
    # output slice, matching lora_b slices)
    a13 = [(torch.randn(W, E, R, H, device=DEV) * 0.05).bfloat16() for _ in range(2)]
    b13 = [(torch.randn(W, E, I, R, device=DEV) * 0.05).bfloat16() for _ in range(2)]
    a2 = [(torch.randn(W, E, R, I, device=DEV) * 0.05).bfloat16()]
    b2 = [(torch.randn(W, E, H, R, device=DEV) * 0.05).bfloat16()]
    mapping = torch.randint(-1, W, (rows,), device=DEV, dtype=torch.long)
    experts = torch.randint(0, E, (rows,), device=DEV, dtype=torch.long)

    ctx = SimpleNamespace(
        punica_wrapper=wrapper,
        adapter_enabled=torch.ones(W + 1, device=DEV, dtype=torch.int32),
        w13_lora_a_stacked=tuple(a13),
        w13_lora_b_stacked=tuple(b13),
        w2_lora_a_stacked=tuple(a2),
        w2_lora_b_stacked=tuple(b2),
        fully_sharded=False,
        top_k=2,
    )
    y13 = torch.zeros(rows, 2 * I, device=DEV, dtype=torch.bfloat16)
    y2 = torch.zeros(rows, H, device=DEV, dtype=torch.bfloat16)
    routing = (experts, mapping)

    moe_lora_apply_w13(ctx, gate_up_out=y13, hidden_states=x, lora_routing=routing)
    moe_lora_apply_w2(ctx, down_out=y2, silu_out=silu, lora_routing=routing)
    assert not hasattr(ctx, "combined_lora_idx")

    ref13 = torch.zeros(rows, 2 * I, device=DEV, dtype=torch.float32)
    ref2 = torch.zeros(rows, H, device=DEV, dtype=torch.float32)
    for r in range(rows):
        slot, e = int(mapping[r]), int(experts[r])
        if slot < 0:
            continue
        ref13[r, :I] += (x[r].float() @ a13[0][slot, e].float().T) @ b13[0][slot, e].float().T
        ref13[r, I:] += (x[r].float() @ a13[1][slot, e].float().T) @ b13[1][slot, e].float().T
        ref2[r] += (silu[r].float() @ a2[0][slot, e].float().T) @ b2[0][slot, e].float().T

    err13 = (y13.float() - ref13).abs().max().item()
    err2 = (y2.float() - ref2).abs().max().item()
    print(f"rows={rows}: w13 err={err13:.2e} w2 err={err2:.2e}")
    assert err13 < 3e-2, err13
    assert err2 < 3e-2, err2
