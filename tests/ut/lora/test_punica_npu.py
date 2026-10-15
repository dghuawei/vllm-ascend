# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PunicaWrapperNPU against the fused LoRA op semantics.

The python-level sgmv/bgmv prefill/decode dispatch lives inside the opaque
add_lora_shrink/add_lora_expand/add_lora ops (branched in C++ on the CPU
flag tensors), so these tests assert the flag plumbing and the python-side
branch structure instead of sgmv-vs-bgmv call selection.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import vllm_ascend.vllm_ascend_C  # noqa: F401  (registers torch.ops._C_ascend)

from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.lora import lora_ops
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.utils import AscendDeviceType


def _make_wrapper(
    *,
    is_prefill=False,
    no_lora=False,
    device_type=AscendDeviceType.A2,
    capturing=False,
    use_add_lora_kernel=False,
    use_moe_gmm=True,
) -> PunicaWrapperNPU:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = is_prefill
    wrapper.no_lora = no_lora
    wrapper.ascend_device_type = device_type
    wrapper._in_graph_mode = Mock(return_value=capturing)
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand = Mock()
    wrapper.bgmv_expand_slice = Mock()
    wrapper.sgmv_shrink = Mock()
    wrapper.sgmv_expand = Mock()
    wrapper.sgmv_expand_slice = Mock()
    # CPU flag tensors consumed by the opaque LoRA ops (see update_metadata).
    # use_moe_gmm=True keeps add_lora_fused_moe on the python per-slice bgmv
    # path; False sends eligible batches into the fused add_lora op.
    wrapper._use_gmm_shrink_cpu = torch.tensor(is_prefill, dtype=torch.bool)
    wrapper._use_gmm_expand_cpu = torch.tensor(is_prefill, dtype=torch.bool)
    wrapper._no_lora_cpu = torch.tensor(no_lora, dtype=torch.bool)
    wrapper._use_moe_gmm_cpu = torch.tensor(use_moe_gmm, dtype=torch.bool)
    wrapper._use_add_lora_cpu = torch.tensor(use_add_lora_kernel, dtype=torch.bool)
    wrapper._prefill_meta_ready = bool(is_prefill)
    # prefill_metadata is a read-only property computed from the fields above.
    # PunicaWrapperBase exposes these as read-only properties.
    wrapper.batch_size = 1
    wrapper.max_length = 1
    wrapper.token_nums = 1
    wrapper._seq_start_locs = torch.tensor([0])
    wrapper._seq_lengths = torch.tensor([1])
    wrapper._lora_indices_per_batch = torch.tensor([0])
    wrapper._token_lora_indices = torch.tensor([0, 1, 2, 3])
    wrapper._sampler_indices = torch.tensor([1, 0, 1, 0])
    wrapper.indices_len = [2, 2, 2, 2]
    return wrapper


@pytest.mark.parametrize(
    ("device_type", "max_lora_rank", "expect_torch_ops"),
    [
        (AscendDeviceType._310P, 8, True),
        (AscendDeviceType.A2, 128, True),
        (AscendDeviceType.A2, 16, False),
    ],
)
def test_punica_init_selects_kernel_backend(device_type, max_lora_rank, expect_torch_ops) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config

    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_current_hardware_profile",
            return_value=get_hardware_profile(device_type),
        ),
        patch(
            "vllm_ascend.lora.punica_npu.get_ascend_device_type",
            return_value=device_type,
        ),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes") as refresh,
        set_current_vllm_config(VllmConfig()),
    ):
        wrapper = PunicaWrapperNPU(
            8,
            2,
            torch.device("cpu"),
            lora_config=SimpleNamespace(
                max_lora_rank=max_lora_rank,
                max_loras=2,
                fully_sharded_loras=False,
            ),
        )
    refresh.assert_called_once()
    if expect_torch_ops:
        from vllm.lora.ops.torch_ops import bgmv_shrink

        assert wrapper.bgmv_shrink is bgmv_shrink
    else:
        assert wrapper.bgmv_shrink is lora_ops.bgmv_shrink


def test_add_shrink_calls_opaque_op_once() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    x = torch.ones(2, 8)
    y = (torch.zeros(2, 4), torch.zeros(2, 4))
    a = (torch.ones(1, 4, 8), torch.ones(1, 4, 8))
    with patch("torch.ops._C_ascend.add_lora_shrink") as op:
        wrapper.add_shrink(y, x, a, 0.25)
    assert op.call_count == 1
    # add_lora_shrink(y, x, lora_a, lora_indices, seq_len, token_lora_indices, scale, ...)
    assert len(op.call_args.args[0]) == 2
    assert op.call_args.args[6] == 0.25


def test_add_expand_calls_opaque_op_once() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    y = torch.zeros(2, 12)
    x = (torch.ones(2, 4), torch.ones(2, 4))
    b = (torch.ones(1, 4, 4), torch.ones(1, 8, 4))
    with patch("torch.ops._C_ascend.add_lora_expand") as op:
        wrapper.add_expand(y, x, b, (4, 8), offset_start=0, add_inputs=True)
    assert op.call_count == 1
    assert op.call_args.args[6] == [4, 8]
    assert op.call_args.args[7] == 0


def test_add_lora_embedding_skips_when_no_lora() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=True)
    wrapper.add_lora_embedding(torch.zeros(2, 8), torch.ones(2, 8, dtype=torch.float16), torch.ones(2, 8, 8))
    wrapper.sgmv_expand.assert_not_called()
    wrapper.bgmv_expand.assert_not_called()


def test_add_lora_embedding_casts_input_to_fp32() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 8, dtype=torch.float16)
    weights = torch.ones(2, 8, 8)
    wrapper.add_lora_embedding(y, x, weights)
    passed_x = wrapper.bgmv_expand.call_args.args[0]
    assert passed_x.dtype == torch.float32


def test_add_lora_embedding_prefill_uses_sgmv_expand() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=False)
    wrapper.add_lora_embedding(torch.zeros(2, 8), torch.ones(2, 8), torch.ones(2, 8, 8))
    wrapper.sgmv_expand.assert_called_once()
    wrapper.bgmv_expand.assert_not_called()


def test_add_lora_linear_routes_to_fused_add_lora() -> None:
    wrapper = _make_wrapper(is_prefill=False, use_add_lora_kernel=True)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 8)
    a = (torch.ones(1, 4, 8),)
    b = (torch.ones(1, 8, 4),)
    with patch("torch.ops._C_ascend.add_lora") as op:
        wrapper.add_lora_linear(y, x, a, b, 0.5, (8,))
    op.assert_called_once()


def test_add_lora_linear_buffer_path_uses_shrink_expand() -> None:
    wrapper = _make_wrapper(is_prefill=False, use_add_lora_kernel=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 8)
    a = (torch.ones(1, 4, 8),)
    b = (torch.ones(1, 8, 4),)
    buffer = (torch.zeros(2, 4),)
    with (
        patch("torch.ops._C_ascend.add_lora_shrink") as shrink,
        patch("torch.ops._C_ascend.add_lora_expand") as expand,
    ):
        wrapper.add_lora_linear(y, x, a, b, 0.5, (8,), buffer=buffer)
    shrink.assert_called_once()
    expand.assert_called_once()


def test_add_lora_fused_moe_builds_graph_safe_combined_index() -> None:
    wrapper = _make_wrapper(use_add_lora_kernel=False)
    max_loras, num_experts, rank, in_f, out_f = 2, 4, 8, 16, 32
    a = torch.ones(max_loras, num_experts, rank, in_f)
    b = torch.ones(max_loras, num_experts, out_f, rank)
    x = torch.ones(3, in_f)
    y = torch.zeros(3, out_f)
    mapping = torch.tensor([0, -1, 1])
    expert_ids = torch.tensor([1, 2, 3])
    adapter_enabled = torch.tensor([1, 1])
    wrapper.add_lora_fused_moe(
        y,
        x,
        (a,),
        (b,),
        expert_ids=expert_ids,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=mapping,
        offset=5,
    )
    combined = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(combined, torch.tensor([1, -1, 7]))
    assert wrapper.bgmv_expand_slice.call_args.args[4:6] == (5, 32)
    assert wrapper.bgmv_expand_slice.call_args.kwargs["add_inputs"] is True


def test_add_lora_fused_moe_masks_disabled_adapter_rows() -> None:
    wrapper = _make_wrapper(use_add_lora_kernel=False)
    a = torch.ones(2, 4, 2, 8)
    b = torch.ones(2, 4, 8, 2)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        (a,),
        (b,),
        expert_ids=torch.tensor([1, 2]),
        adapter_enabled=torch.tensor([0, 1]),
        token_lora_mapping=torch.tensor([0, 1]),
    )
    combined = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(combined, torch.tensor([-1, 6]))


def test_add_lora_fused_moe_rejects_unexpanded_rows() -> None:
    wrapper = _make_wrapper()
    with pytest.raises(AssertionError, match="top_k_num=1"):
        wrapper.add_lora_fused_moe(
            torch.zeros(1, 4),
            torch.ones(1, 4),
            (torch.ones(1, 1, 2, 4),),
            (torch.ones(1, 1, 4, 2),),
            expert_ids=torch.tensor([0]),
            adapter_enabled=torch.tensor([1]),
            token_lora_mapping=torch.tensor([0]),
            top_k_num=2,
        )


def test_add_lora_fused_moe_scales_shrink_buffer_by_routed_weight() -> None:
    wrapper = _make_wrapper()
    a = torch.ones(2, 4, 2, 8)
    b = torch.ones(2, 4, 8, 2)
    topk = torch.tensor([2.0, 4.0])
    # bgmv_shrink writes the rank-space delta into its y argument
    wrapper.bgmv_shrink.side_effect = lambda x, w, y, idx, scale: y.fill_(1.0)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        (a,),
        (b,),
        topk_weights=topk,
        expert_ids=torch.tensor([1, 2]),
        adapter_enabled=torch.tensor([1, 1]),
        token_lora_mapping=torch.tensor([0, 1]),
        mul_routed_weight=True,
    )
    passed_delta = wrapper.bgmv_expand_slice.call_args.args[0]
    assert torch.equal(passed_delta, torch.ones(2, 2) * topk.unsqueeze(1))


def test_add_lora_fused_moe_decode_routes_to_fused_add_lora_op() -> None:
    wrapper = _make_wrapper(use_moe_gmm=False)
    a = torch.ones(2, 4, 2, 8)
    b = torch.ones(2, 4, 8, 2)
    with patch("torch.ops._C_ascend.add_lora") as op:
        wrapper.add_lora_fused_moe(
            torch.zeros(2, 8),
            torch.ones(2, 8),
            (a,),
            (b,),
            expert_ids=torch.tensor([1, 2]),
            adapter_enabled=torch.tensor([1, 1]),
            token_lora_mapping=torch.tensor([0, 1]),
        )
    op.assert_called_once()
    wrapper.bgmv_shrink.assert_not_called()


def test_add_lora_logits_uses_sampler_indices() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 16)
    a = torch.ones(2, 4, 16)
    b = torch.ones(2, 8, 4)
    wrapper.add_lora_logits(y, x, a, b, scale=1.0)
    passed_indices = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(passed_indices, torch.tensor([1, 0]))
