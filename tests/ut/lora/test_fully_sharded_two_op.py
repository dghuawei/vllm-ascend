# SPDX-License-Identifier: Apache-2.0
"""Numeric tests for the fully-sharded two-op MoE LoRA path.

add_lora_fused_moe with fully_sharded=True routes through
add_lora_shrink + add_lora_expand (grouped GEMM for prefill, bgmv for
decode) with a cross-rank collective interposed between the two ops:
  - w13-style (lora_a rank-sharded):  all_gather along the rank axis
  - w2-style  (lora_a unsharded):     all_reduce of partial sums

TP collectives are mocked to emulate a TP-world: the mock receives THIS
rank's shrink output and concatenates/sums the other ranks' partials,
computed analytically from the same per-rank weight shards. The final y
is checked against an unsharded fp32 reference, validating buffer
widths, the [T, L, R_local] lora-block view (L>1 reorder), the stacked
decode layout, and the expand placement.
"""

import unittest.mock as um

import pytest
import torch
import torch_npu  # noqa: F401

import vllm  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

DEV = "npu:0"
H, I, R, E, L, TP = 1024, 1024, 64, 4, 2, 8
R_LOC, I_LOC, H_LOC = R // TP, I // TP, H // TP
R2_LOC = 16 // TP  # narrow-shard case (forces gmm dispatch)


def _make_wrapper(gmm: bool) -> PunicaWrapperNPU:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._use_moe_gmm_cpu = torch.tensor(gmm, dtype=torch.bool)
    wrapper._use_gmm_shrink_cpu = torch.tensor(gmm, dtype=torch.bool)
    wrapper._use_gmm_expand_cpu = torch.tensor(gmm, dtype=torch.bool)
    wrapper._no_lora_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._use_add_lora_cpu = torch.tensor(False, dtype=torch.bool)
    return wrapper


def _routing(T: int, seed: int):
    """Rows sorted by expert (gmm requirement); mixed slots incl. -1."""
    torch.manual_seed(seed)
    slots = torch.randint(-1, L, (T,))
    experts = torch.sort(torch.randint(0, E, (T,))).values
    combined = torch.where(
        slots >= 0, slots * E + experts, torch.full_like(slots, -1)
    )
    group_list = torch.bincount(experts, minlength=E).to(torch.int64)
    return slots, experts, combined, group_list


def _shrink_ref(x, a, slots, experts, out_width_layout):
    """fp32 shrink of x against a [L, E, r, K] shard, on the given layout.

    out_width_layout: 'flat' -> [T, L*r] (gmm per-slice buffer),
                      'stacked' -> [1, T, r] (bgmv stacked buffer slice 0).
    """
    T = x.shape[0]
    if out_width_layout == "flat":
        out = torch.zeros(T, L * a.shape[2], dtype=torch.float32, device=DEV)
        for t in range(T):
            slot = int(slots[t])
            if slot < 0:
                continue
            out[t, slot * a.shape[2]:(slot + 1) * a.shape[2]] = (
                x[t].float() @ a[slot, int(experts[t])].float().T
            )
        return out
    out = torch.zeros(1, T, a.shape[2], dtype=torch.float32, device=DEV)
    for t in range(T):
        slot = int(slots[t])
        if slot < 0:
            continue
        out[0, t] = x[t].float() @ a[slot, int(experts[t])].float().T
    return out


def _expand_ref(y, shrink, b, slots, experts, slice_out, offset):
    """shrink is in the flat [T, L*R] layout; slice the slot block per row."""
    ref = y.float().clone()
    T = shrink.shape[0]
    for t in range(T):
        slot = int(slots[t])
        if slot < 0:
            continue
        z = shrink[t, slot * R:(slot + 1) * R]
        ref[t, offset:offset + slice_out[0]] += (
            z @ b[0][slot, int(experts[t])].float().T
        )
    return ref


@pytest.mark.parametrize("gmm", [True, False])
@pytest.mark.parametrize("w13", [True, False])
def test_two_op_fully_sharded(gmm: bool, w13: bool, expect_gmm: bool | None = None) -> None:
    """expect_gmm: the dispatch the mocks must emulate; defaults to gmm.
    The narrow-shard test passes gmm=False, expect_gmm=True to prove the
    BGMV_MIN_RANK guard overrides the decode flag."""
    if expect_gmm is None:
        expect_gmm = gmm
    torch.manual_seed(7)
    T = 48
    slots, experts, combined, group_list = _routing(T, 11)

    if w13:
        # A rank-sharded [L, E, R_LOC, H]; x replicated across ranks;
        # B column-sharded per slice [L, E, I_LOC, R].
        a_shards = [
            [(torch.randn(L, E, R_LOC, H, device=DEV) * 0.05).bfloat16()
             for _ in range(TP)]
            for _ in range(2)
        ]
        b = [(torch.randn(L, E, I_LOC, R, device=DEV) * 0.05).bfloat16()
             for _ in range(2)]
        x = (torch.randn(T, H, device=DEV) * 0.3).bfloat16()
        y = (torch.randn(T, 2 * I_LOC, device=DEV) * 0.2).bfloat16()
        a_stacked = tuple(s[0] for s in a_shards)  # rank-0 tensors for the op
        slice_out = [bb.shape[2] for bb in b]
        offset = 0
        partials = [
            [_shrink_ref(x, a_shards[s][r], slots, experts,
                         "flat" if expect_gmm else "stacked")
             for r in range(1, TP)]
            for s in range(2)
        ]

        def fake_gather(v, dim=-1):
            # gmm: per-slice [T, L, R_LOC] view; bgmv: stacked [S, T, R_LOC]
            if expect_gmm:
                assert v.shape == (T, L, R_LOC)
                s = fake_gather.slice_idx
                fake_gather.slice_idx += 1
                parts = [v] + [
                    p.view(T, L, R_LOC) for p in partials[s]
                ]
                return torch.cat(parts, dim=-1)
            assert v.shape == (2, T, R_LOC)
            parts = [v]
            for r in range(1, TP):
                stacked_r = torch.cat(
                    [partials[s][r - 1] for s in range(2)], dim=0
                )
                parts.append(stacked_r)
            return torch.cat(parts, dim=-1)

        def fake_reduce(v):
            raise AssertionError("w13 must all_gather, not all_reduce")

        # reference: unsharded A = rank-shards concatenated on the rank axis
        a_full = [torch.cat(s, dim=2).contiguous() for s in a_shards]
        shrink_full = [
            _shrink_ref(x, a_full[s], slots, experts, "flat")
            for s in range(2)
        ]
        ref = y.float().clone()
        for t in range(T):
            slot = int(slots[t])
            if slot < 0:
                continue
            col = 0
            for s in range(2):
                ref[t, col:col + slice_out[s]] += (
                    shrink_full[s][t, slot * R:(slot + 1) * R]
                    @ b[s][slot, int(experts[t])].float().T
                )
                col += slice_out[s]
    else:
        # A unsharded over the rank-local input slice [L, E, R, I_LOC];
        # per-rank x differs; B output-sharded [L, E, H_LOC, R].
        a = (torch.randn(L, E, R, I_LOC, device=DEV) * 0.05).bfloat16()
        b = [(torch.randn(L, E, H_LOC, R, device=DEV) * 0.05).bfloat16()]
        xs = [(torch.randn(T, I_LOC, device=DEV) * 0.3).bfloat16()
              for _ in range(TP)]
        x = xs[0]
        y = (torch.randn(T, H_LOC, device=DEV) * 0.2).bfloat16()
        a_stacked = (a,)
        slice_out = [b[0].shape[2]]
        offset = 0  # tp_rank 0
        partials = [
            _shrink_ref(xs[r], a, slots, experts, "flat" if expect_gmm else "stacked")
            for r in range(1, TP)
        ]

        def fake_gather(v, dim=-1):
            raise AssertionError("w2 must all_reduce, not all_gather")

        def fake_reduce(v):
            if expect_gmm:
                assert v.shape == (T, L * R)
                extra = torch.zeros_like(v)
                for p in partials:
                    extra += p
                return v + extra
            assert v.shape == (1, T, R)
            out = v.clone()
            for p in partials:
                out += p
            return out

        # sum of per-rank partials == shrink over the full input (A identical
        # across ranks; each rank sees a different input-slice)
        summed = None
        for xr in xs:
            p = _shrink_ref(xr, a, slots, experts, "flat")
            summed = p if summed is None else summed + p
        ref = _expand_ref(y, summed, b, slots, experts, slice_out, offset)

    wrapper = _make_wrapper(gmm)
    y_out = y.clone()
    fake_gather.slice_idx = 0
    with (
        um.patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
            side_effect=fake_gather,
        ),
        um.patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce",
            side_effect=fake_reduce,
        ),
        um.patch(
            "vllm_ascend.lora.punica_npu.get_tensor_model_parallel_world_size",
            return_value=TP,
        ),
    ):
        wrapper.add_lora_fused_moe(
            y=y_out,
            x=x,
            lora_a_stacked=a_stacked,
            lora_b_stacked=tuple(b),
            expert_ids=None,
            adapter_enabled=torch.ones(L + 1, device=DEV, dtype=torch.int32),
            fully_sharded=True,
            token_lora_mapping=combined.to(DEV),
            combined_idx=combined.to(DEV),
            group_list=group_list.to(DEV),
            group_list_type=1,
            offset=offset,
        )

    err = (y_out.float() - ref).abs().max().item()
    name = f"{'w13' if w13 else 'w2'}-{'gmm' if gmm else 'bgmv'}"
    print(f"{name}: T={T} err={err:.3e}")
    assert err < 5e-2, f"{name} max_err={err}"


def test_two_op_narrow_rank_shard_bgmv() -> None:
    """R/tp < 8 must work on the bgmv decode path (kernel CopyOutPad fix).

    bgmv_shrink's per-row store used DataCopy, which silently drops copies
    that are not 32B-aligned (rank < 8 fp32 elements); it now routes through
    DataCopyPad. This test uses R = 16, TP = 8 -> R_LOC = 2, the production
    fully-sharded shape for max_lora_rank 16, on the decode (bgmv) dispatch.
    The expand input is the gathered full rank (16), which bgmv_expand's
    reduce tree supports.
    """
    saved = {k: globals()[k] for k in ("R", "R_LOC", "I", "I_LOC", "H", "H_LOC")}
    try:
        globals().update(R=16, R_LOC=R2_LOC, I=1024, I_LOC=1024 // TP,
                         H=1024, H_LOC=1024 // TP)
        # decode-shaped batch with gmm disabled: bgmv must be correct
        test_two_op_fully_sharded(False, True, expect_gmm=False)
    finally:
        globals().update(saved)


@pytest.mark.parametrize("gmm", [True, False])
def test_w2_partial_expand_fold(gmm: bool) -> None:
    """partial_expand must skip the dedicated w2 all-reduce.

    Each rank expands its partial shrink (input-sharded full-rank A) into
    its tp_rank output slice of the [T, H] partial down_out; the MoE
    runner's final TP all-reduce completes the delta. Emulated here by
    summing TP per-rank outputs (each rank's base partial reused).
    """
    torch.manual_seed(23)
    T = 48
    R_, E_, L_ = R, E, L
    slots, experts, combined, group_list = _routing(T, 23)
    I2_LOC = 256  # w2 lora_a input shard (silu width per rank)

    a = (torch.randn(L_, E_, R_, I2_LOC, device=DEV) * 0.05).bfloat16()
    b_all = [
        (torch.randn(L_, E_, H // TP, R_, device=DEV) * 0.05).bfloat16()
        for _ in range(TP)
    ]
    xs = [(torch.randn(T, I2_LOC, device=DEV) * 0.3).bfloat16() for _ in range(TP)]
    y0 = (torch.randn(T, H, device=DEV) * 0.2).bfloat16()  # full-width partial

    def partial_z(xr):
        out = torch.zeros(T, R_, dtype=torch.float32, device=DEV)
        for t in range(T):
            s = int(slots[t])
            if s < 0:
                continue
            out[t] = xr[t].float() @ a[s, int(experts[t])].float().T
        return out

    def fake_gather(v, dim=-1):
        raise AssertionError("partial_expand must not all_gather")

    ar_called = {"n": 0}

    def fake_reduce(v):
        ar_called["n"] += 1
        raise AssertionError("partial_expand must not all_reduce")

    wrapper = _make_wrapper(gmm)
    y_out = y0.clone()
    with (
        um.patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
            side_effect=fake_gather,
        ),
        um.patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce",
            side_effect=fake_reduce,
        ),
        um.patch(
            "vllm_ascend.lora.punica_npu.get_tensor_model_parallel_world_size",
            return_value=TP,
        ),
    ):
        wrapper.add_lora_fused_moe(
            y=y_out,
            x=xs[0],
            lora_a_stacked=(a,),
            lora_b_stacked=(b_all[0],),
            expert_ids=None,
            adapter_enabled=torch.ones(L_ + 1, device=DEV, dtype=torch.int32),
            fully_sharded=True,
            partial_expand=True,
            token_lora_mapping=combined.to(DEV),
            combined_idx=combined.to(DEV),
            group_list=group_list.to(DEV),
            group_list_type=1,
            offset=0,  # tp_rank 0
        )

    # Emulate the final MoE all-reduce: each rank's y0-partial plus its own
    # block delta, summed over ranks.
    ref = y0.float() * TP
    for r in range(TP):
        z = partial_z(xs[r])
        blk = torch.zeros(T, H // TP, dtype=torch.float32, device=DEV)
        for t in range(T):
            s = int(slots[t])
            if s < 0:
                continue
            blk[t] = z[t] @ b_all[r][s, int(experts[t])].float().T
        ref[:, r * (H // TP):(r + 1) * (H // TP)] += blk

    # y_out is rank-0's partial: y0 + block-0 delta; ref - (TP-1)*y0 gives
    # exactly y0 + sum_r(block_r) -- rank 0's view after the final AR only
    # contains its own block delta, so strip the other ranks' blocks.
    expected = (ref - (TP - 1) * y0.float()).clone()
    expected[:, H // TP:] = y0.float()[:, H // TP:]
    err = (y_out.float() - expected).abs().max().item()
    print(f"w2-fold gmm={int(gmm)}: no-collective={ar_called['n'] == 0} "
          f"err={err:.3e}")
    assert err < 5e-2, err
