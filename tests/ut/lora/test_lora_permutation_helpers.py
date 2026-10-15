# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the scatter-based permutation helpers on the AlltoAll
MoE LoRA path (replaces the int64 argsort that falls back to AiCpu)."""

import torch

from vllm_ascend.lora.fused_moe import _gather_by_row_permutation


def test_gather_by_row_permutation_matches_argsort():
    """The scatter-based permutation gather must equal values[argsort(m)]."""
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        n = torch.randint(8, 200, (1,), generator=g).item()
        values = torch.randint(-1, 3, (n,), generator=g)
        perm = torch.randperm(n, generator=g)
        # full permutation
        assert torch.equal(_gather_by_row_permutation(values, perm), values[torch.argsort(perm)])
        # -1 entries sort first, exactly like plain argsort (identical op
        # modulo the fp32 key cast).
        m = perm.clone()
        m[-3:] = -1
        out = _gather_by_row_permutation(values, m)
        ref = values[torch.argsort(m)]
        assert torch.equal(out, ref)


def test_gather_by_row_permutation_shapes_value_independent():
    for n in (8, 64, 257):
        values = torch.zeros(n, dtype=torch.long)
        m = torch.randperm(n)
        out = _gather_by_row_permutation(values, m)
        assert out.shape == (n,)
