# SPDX-License-Identifier: Apache-2.0
"""Numeric tests for the fused add_lora path in add_lora_fused_moe.

Validates that the decode path (single opaque add_lora op with the in-tree
fused kernel) produces the same result as the legacy per-row bgmv loop and
the einsum reference, on DeepSeek-V4-Flash-shaped w13/w2 projections with
expert-dispatched rows, mixed adapters and -1 (no-adapter) rows.
"""

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401
import vllm_ascend.lora.punica_npu as punica_mod
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

DEV = "npu:0"
H, I, R, W, E_LOCAL = 7168, 2048, 16, 2, 8
W13_SLICES = (I, I)
W2_SLICES = (H,)


def _make_wrapper():
    """Bare wrapper with just the state add_lora_fused_moe touches."""
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._use_moe_gmm_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._no_lora_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._use_add_lora_cpu = torch.tensor(True, dtype=torch.bool)
    wrapper._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
    from vllm_ascend.lora.lora_ops import (
        bgmv_expand_slice,
        bgmv_shrink,
    )

    wrapper.bgmv_shrink = bgmv_shrink
    wrapper.bgmv_expand_slice = bgmv_expand_slice
    return wrapper


def _make_case(rows, seed, with_offset=0):
    torch.manual_seed(seed)
    x = (torch.randn(rows, H, device=DEV) * 0.3).bfloat16()
    a13 = [(torch.randn(W, E_LOCAL, R, H, device=DEV) * 0.05).bfloat16() for _ in W13_SLICES]
    b13 = [(torch.randn(W, E_LOCAL, s, R, device=DEV) * 0.05).bfloat16() for s in W13_SLICES]
    a2 = [(torch.randn(W, E_LOCAL, R, I, device=DEV) * 0.05).bfloat16() for _ in W2_SLICES]
    b2 = [(torch.randn(W, E_LOCAL, s, R, device=DEV) * 0.05).bfloat16() for s in W2_SLICES]
    # mixed adapters, ~1/4 rows without adapter (-1)
    token_lora_mapping = torch.randint(-1, W, (rows,), device=DEV, dtype=torch.long)
    adapter_enabled = torch.ones(W + 1, device=DEV, dtype=torch.int32)
    expert_ids = torch.randint(0, E_LOCAL, (rows,), device=DEV, dtype=torch.long)
    return x, a13, b13, a2, b2, token_lora_mapping, adapter_enabled, expert_ids


def _reference(x, a_stack, b_stack, mapping, enabled, expert_ids, y_width, offset, seed_y):
    torch.manual_seed(seed_y)
    y = (torch.randn(x.shape[0], y_width, device=DEV) * 0.2).bfloat16()
    ref = y.float().clone()
    for r in range(x.shape[0]):
        slot = int(mapping[r])
        if slot < 0:
            continue
        e = int(expert_ids[r])
        col = offset
        for s in range(len(a_stack)):
            out_s = b_stack[s].shape[-2]
            z = x[r].float() @ a_stack[s][slot, e].float().T
            ref[r, col : col + out_s] += z @ b_stack[s][slot, e].float().T
            col += out_s
    return y, ref


def _run(wrapper, x, a_stack, b_stack, mapping, enabled, eids, y, fused):
    punica_mod.ENABLE_ADD_LORA_KERNEL = fused
    try:
        wrapper.add_lora_fused_moe(
            y=y,
            x=x,
            lora_a_stacked=tuple(a_stack),
            lora_b_stacked=tuple(b_stack),
            expert_ids=eids,
            adapter_enabled=enabled,
            token_lora_mapping=mapping,
        )
    finally:
        punica_mod.ENABLE_ADD_LORA_KERNEL = True
    return y


@pytest.mark.parametrize("rows", [8, 64, 256])
def test_fused_matches_reference_and_legacy(rows):
    wrapper = _make_wrapper()
    x, a13, b13, a2, b2, mapping, enabled, eids = _make_case(rows, seed=rows)

    # w13 (2 slices, offset 0)
    y_f, ref = _reference(x, a13, b13, mapping, enabled, eids, 2 * I, 0, seed_y=100)
    y_l = y_f.clone()
    _run(wrapper, x, a13, b13, mapping, enabled, eids, y_f, fused=True)
    _run(wrapper, x, a13, b13, mapping, enabled, eids, y_l, fused=False)
    err_ref = (y_f.float() - ref).abs().max().item()
    err_leg = (y_f.float() - y_l.float()).abs().max().item()
    print(f"w13 rows={rows}: vs-ref {err_ref:.2e}, vs-legacy {err_leg:.2e}")
    assert err_ref < 3e-2, err_ref
    assert err_leg < 3e-2, err_leg

    # w2 (1 slice; its lora A consumes the silu output of dim I)
    x2 = (torch.randn(rows, I, device=DEV) * 0.3).bfloat16()
    y_f2, ref2 = _reference(x2, a2, b2, mapping, enabled, eids, H, 0, seed_y=200)
    y_l2 = y_f2.clone()
    _run(wrapper, x2, a2, b2, mapping, enabled, eids, y_f2, fused=True)
    _run(wrapper, x2, a2, b2, mapping, enabled, eids, y_l2, fused=False)
    err_ref2 = (y_f2.float() - ref2).abs().max().item()
    err_leg2 = (y_f2.float() - y_l2.float()).abs().max().item()
    print(f"w2  rows={rows}: vs-ref {err_ref2:.2e}, vs-legacy {err_leg2:.2e}")
    assert err_ref2 < 3e-2, err_ref2
    assert err_leg2 < 3e-2, err_leg2


def test_no_adapter_rows_untouched():
    wrapper = _make_wrapper()
    rows = 32
    x, a13, b13, _, _, mapping, enabled, eids = _make_case(rows, seed=5)
    mapping = torch.full((rows,), -1, device=DEV, dtype=torch.long)  # all base
    y, ref = _reference(x, a13, b13, mapping, enabled, eids, 2 * I, 0, seed_y=7)
    _run(wrapper, x, a13, b13, mapping, enabled, eids, y, fused=True)
    assert torch.equal(y, ref.bfloat16()) or (y.float() - ref).abs().max().item() < 1e-6


def test_large_rows_falls_back_inside_op():
    """rows > decode cap: the op's internal bgmv-pair fallback must still be
    correct (same math as the legacy loop)."""
    wrapper = _make_wrapper()
    rows = 512
    x, a13, b13, _, _, mapping, enabled, eids = _make_case(rows, seed=9)
    y_f, ref = _reference(x, a13, b13, mapping, enabled, eids, 2 * I, 0, seed_y=11)
    _run(wrapper, x, a13, b13, mapping, enabled, eids, y_f, fused=True)
    err = (y_f.float() - ref).abs().max().item()
    print(f"rows=512 (fallback): vs-ref {err:.2e}")
    assert err < 3e-2, err
