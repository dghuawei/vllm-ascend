# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dense torch.ops._C_ascend.add_lora dispatch
(fused in-tree kernel / gmm prefill / bgmv pair fallback) against a
torch reference."""

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401  (registers torch.ops._C_ascend)

H1 = 512
RANK = 16
SLICES = (768, 512)
NUM_SLOTS = 3
SCALE = 0.75
# bf16 unit roundoff is 2^-8; the widest reduce (H1=512 matmuls chained
# through fp32 accumulation) stays within a few ulp of the fp32 reference
DIFF_LIMIT = 0.05


def _make_case(num_tokens, dtype, seed, num_seqs=None):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    device = "npu"
    x = torch.randn(num_tokens, H1, generator=gen).to(dtype).to(device)
    y0 = torch.randn(num_tokens, sum(SLICES), generator=gen).to(dtype).to(device)
    lora_a = [
        (torch.randn(NUM_SLOTS, 1, RANK, H1, generator=gen) * 0.05).to(dtype).to(device)
        for _ in SLICES
    ]
    lora_b = [
        (torch.randn(NUM_SLOTS, 1, slice_size, RANK, generator=gen) * 0.05).to(dtype).to(device)
        for slice_size in SLICES
    ]
    if num_seqs is not None:
        # request-style layout: contiguous blocks, one adapter per block
        token_indices = torch.empty(num_tokens, dtype=torch.long)
        bounds = torch.linspace(0, num_tokens, num_seqs + 1).round().long().tolist()
        for seq_idx in range(num_seqs):
            token_indices[bounds[seq_idx]:bounds[seq_idx + 1]] = (
                -1 if (seq_idx % 4 == 3) else (seq_idx % NUM_SLOTS)
            )
    else:
        token_indices = torch.randint(0, NUM_SLOTS, (num_tokens,), generator=gen)
        no_lora_mask = torch.rand(num_tokens, generator=gen) < 0.25
        token_indices[no_lora_mask] = -1
    token_indices = token_indices.to(device)

    # per-sequence grouping metadata consumed by the gmm branch
    seq_start = 0
    seq_lens, group_slots = [], []
    for idx in range(1, num_tokens + 1):
        if idx == num_tokens or token_indices[idx] != token_indices[seq_start]:
            seq_lens.append(idx - seq_start)
            group_slots.append(token_indices[seq_start].item())
            seq_start = idx
    seq_len = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    lora_indices = torch.tensor(group_slots, dtype=torch.int64, device=device)
    return x, y0, lora_a, lora_b, token_indices, lora_indices, seq_len


def _reference(x, y0, lora_a, lora_b, token_indices, scale, add_inputs):
    out = y0.clone() if add_inputs else torch.zeros_like(y0)
    x_f32 = x.to(torch.float32)
    num_tokens = x.size(0)
    offset = 0
    for slice_idx, slice_size in enumerate(SLICES):
        a = lora_a[slice_idx].squeeze(1).to(torch.float32)
        b = lora_b[slice_idx].squeeze(1).to(torch.float32)
        z = torch.einsum("th,nrh->ntr", x_f32, a)
        res = torch.einsum("ntr,nsr->nts", z, b)
        rows = torch.arange(num_tokens, device=x.device)
        sel = res[token_indices.clamp_min(0), rows] * scale
        sel = torch.where(
            (token_indices < 0).unsqueeze(1), torch.zeros_like(sel), sel
        )
        out[:, offset:offset + slice_size] += sel.to(out.dtype)
        offset += slice_size
    return out


def _run_op(num_tokens, dtype, use_gmm, use_add_lora, add_inputs=True,
            no_lora=False, seed=1, num_seqs=None):
    x, y, lora_a, lora_b, token_indices, lora_indices, seq_len = _make_case(
        num_tokens, dtype, seed, num_seqs=num_seqs)
    torch.ops._C_ascend.add_lora(
        y, x, lora_a, lora_b, lora_indices, seq_len, token_indices,
        list(SLICES), 0, SCALE, add_inputs,
        torch.tensor(use_gmm), torch.tensor(no_lora), torch.tensor(use_add_lora),
    )
    ref = _reference(
        x, y if no_lora else _make_case(num_tokens, dtype, seed)[1],
        lora_a, lora_b, token_indices, SCALE, add_inputs,
    )
    if no_lora:
        ref = _make_case(num_tokens, dtype, seed)[1]
    torch.npu.synchronize()
    return y, ref


def _assert_close(y, ref):
    diff = (y.to(torch.float32) - ref.to(torch.float32)).abs()
    assert diff.max().item() <= DIFF_LIMIT, f"max abs diff {diff.max().item():.4f}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("num_tokens", "use_gmm", "use_add_lora", "num_seqs"),
    [
        (48, False, True, None),      # decode -> fused kernel
        (48, False, False, None),     # decode -> bgmv pair fallback
        (1024, False, True, None),    # decode, merged-slice window -> fused
        (512, True, True, 8),         # small prefill -> fused kernel
        (3072, True, True, 16),       # large prefill -> gmm
    ],
)
def test_add_lora_matches_reference(dtype, num_tokens, use_gmm, use_add_lora, num_seqs):
    y, ref = _run_op(num_tokens, dtype, use_gmm, use_add_lora, num_seqs=num_seqs)
    _assert_close(y, ref)


def test_add_lora_narrow_slice_matches_reference():
    # slices narrower than the rank take the aten expand path in the fallback
    global SLICES
    orig = SLICES
    SLICES = (768, 8)
    try:
        y, ref = _run_op(48, torch.bfloat16, False, True)
        _assert_close(y, ref)
        y, ref = _run_op(3072, torch.bfloat16, True, True, num_seqs=16)
        _assert_close(y, ref)
    finally:
        SLICES = orig


def test_add_lora_no_lora_flag_leaves_y_unchanged():
    y, ref = _run_op(48, torch.bfloat16, False, True, no_lora=True)
    assert torch.equal(y, ref)


def test_add_lora_add_inputs_false_overwrites():
    y, ref = _run_op(48, torch.bfloat16, False, True, add_inputs=False)
    _assert_close(y, ref)
    assert not torch.equal(y, torch.zeros_like(y))


def test_add_lora_tokens_without_lora_contribute_nothing():
    # every 4th block (-1) must not alter those rows beyond the reference
    y, ref = _run_op(2048, torch.bfloat16, True, True, num_seqs=12, seed=3)
    _assert_close(y, ref)
