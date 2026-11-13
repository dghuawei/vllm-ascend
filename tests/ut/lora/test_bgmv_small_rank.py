# SPDX-License-Identifier: Apache-2.0
"""Narrow-rank regression tests for the AscendC bgmv kernels.

bgmv_shrink's per-row CopyOut (fp32 Y) and bgmv_expand's per-row CopyInX
(fp32 X) move lora_rank * 4 bytes per token; for rank < 8 the DataCopy
length and GM row stride are not 32B-aligned and the transfer is silently
dropped. Both sites now route through DataCopyPad for unaligned ranks.

Known remaining limitation (documented, not fixed here): bgmv_expand's
Compute() reduce tree implements rank in {8, 16, 32, 64} only; narrower
expands silently skip the reduction. Narrow-rank *adapters* (rank < 8)
must therefore avoid the bgmv expand (the dense fused path already
rejects them via add_lora_eligible; MoE fully-sharded applies always
expand at full rank after the gather/reduce).
"""

import pytest
import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.lora.lora_ops import bgmv_expand_slice, bgmv_shrink

DEV = "npu:0"


@pytest.mark.parametrize("K", [1, 2, 4, 8, 16])
def test_bgmv_shrink_rank_sweep(K):
    """Shrink-only vs fp32 reference across rank widths (CopyOutPad fix)."""
    H, T, E = 512, 48, 4
    torch.manual_seed(5)
    experts = torch.sort(torch.randint(0, E, (T,))).values
    idx = experts.to(DEV)
    a = (torch.randn(E, K, H, device=DEV) * 0.05).bfloat16()
    x = (torch.randn(T, H, device=DEV) * 0.3).bfloat16()

    out = torch.zeros(T, K, dtype=torch.float32, device=DEV)
    bgmv_shrink(x, a, out, idx, 1.0)

    ref = torch.zeros(T, K, dtype=torch.float32, device=DEV)
    for t in range(T):
        ref[t] = x[t].float() @ a[int(experts[t])].float().T
    err = (out - ref).abs().max().item()
    print(f"shrink K={K}: err={err:.3e} max={ref.abs().max().item():.3f}")
    assert err < 3e-2, f"K={K} err={err}"
    assert ref.abs().max().item() > 1e-3  # non-degenerate weights


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.xfail(
    reason="bgmv_expand reduce tree only implements rank {8,16,32,64}; narrow ranks skip the reduction", strict=True
)
def test_bgmv_expand_narrow_rank_known_limitation(K):
    """Documents the narrow-expand limitation; passes once Compute() grows
    reduce branches for rank < 8."""
    R, N, T = K, 64, 48
    torch.manual_seed(6)
    b = (torch.randn(1, N, R, device=DEV) * 0.05).bfloat16()
    z = (torch.randn(T, R, device=DEV) * 0.2).float()
    y0 = (torch.randn(T, N, device=DEV) * 0.2).bfloat16()
    idx = torch.zeros(T, dtype=torch.long, device=DEV)
    y_out = y0.clone()
    bgmv_expand_slice(z, b, y_out, idx, 0, N, add_inputs=True)
    ref = y0.float().clone()
    for t in range(T):
        ref[t] += z[t] @ b[0].float().T
    err = (y_out.float() - ref).abs().max().item()
    print(f"expand K={K}: err={err:.3e}")
    assert err < 3e-2, f"K={K} err={err}"


