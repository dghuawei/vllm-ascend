from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    has_lora,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def test_ascend_fused_moe_lora_initializes_skipped_upstream_fields() -> None:
    parallel_config = SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=False)
    shared_experts = torch.nn.Module()
    base_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=256,
            num_experts=256,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
        _shared_experts=shared_experts,
    )

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper.n_slices == 256 * 3


def test_moe_lora_apply_uses_adapter_enabled() -> None:
    punica_wrapper = Mock()
    # moe_lora_apply_w2 folds the return ("delta is not None" contract);
    # a fold-in-place dispatch returns None.
    punica_wrapper.add_lora_fused_moe.return_value = None
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked=(torch.empty(2, 4, 8, 32),),
        w13_lora_b_stacked=(torch.empty(2, 4, 16, 8),),
        w2_lora_a_stacked=(torch.empty(2, 4, 8, 16),),
        w2_lora_b_stacked=(torch.empty(2, 4, 32, 8),),
        adapter_enabled=torch.ones(3, dtype=torch.int32),
        fully_sharded=False,
        tp_rank=0,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out=torch.empty(2, 8),
        hidden_states=torch.empty(2, 32),
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out=torch.empty(2, 32),
        silu_out=torch.empty(2, 16),
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["adapter_enabled"] is context.adapter_enabled
    assert calls[1].kwargs["adapter_enabled"] is context.adapter_enabled
    assert calls[0].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["offset"] == 0


def test_moe_lora_apply_propagates_fully_sharded_metadata() -> None:
    punica_wrapper = Mock()
    # moe_lora_apply_w2 folds the return ("delta is not None" contract);
    # a fold-in-place dispatch returns None.
    punica_wrapper.add_lora_fused_moe.return_value = None
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked=(torch.empty(2, 4, 8, 32),),
        w13_lora_b_stacked=(torch.empty(2, 4, 16, 8),),
        w2_lora_a_stacked=(torch.empty(2, 4, 8, 16),),
        w2_lora_b_stacked=(torch.empty(1, 1, 16, 8),),
        adapter_enabled=torch.ones(3, dtype=torch.int32),
        fully_sharded=True,
        tp_rank=3,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out=torch.empty(2, 8),
        hidden_states=torch.empty(2, 32),
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out=torch.empty(2, 32),
        silu_out=torch.empty(2, 16),
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["offset"] == 48


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert torch.equal(expert_ids, torch.tensor([0, 0, 1, 1, 1, 1]))
    assert torch.equal(lora_slots, torch.tensor([0, -1, 0, -1, 1, 1]))


def test_all2all_routing_uses_local_experts_and_exchanged_adapters() -> None:
    context = SimpleNamespace(
        local_num_experts=3,
        exchanged_lora_indices=torch.tensor([1, -1, 0, 2]),
    )

    expert_ids, lora_slots = _recover_moe_lora_routing_all2all(
        context,
        group_list=torch.tensor([2, 0, 2]),
    )

    assert torch.equal(expert_ids, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(lora_slots, torch.tensor([1, -1, 0, 2]))


def test_has_lora_follows_batch_metadata() -> None:
    assert not has_lora(None)
    assert not has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))


@pytest.mark.parametrize(
    ("index_mapping", "expected_no_lora"),
    [((0, 0), True), ((0, 1), False), ((2, 0), False)],
)
def test_decode_metadata_refreshes_no_lora(index_mapping, expected_no_lora) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    # update_metadata reads the CPU no_lora tensor allocated in __init__ and
    # queries the process-wide gmm threshold via torch.ops._C_ascend; this
    # test bypasses __init__, so provision the tensor directly and set the
    # threshold explicitly. A decode-size batch must take the bgmv branch,
    # so keep the threshold out of reach.
    torch.ops._C_ascend.set_lora_gmm_threshold(1 << 30)
    wrapper._no_lora_cpu = torch.tensor(True, dtype=torch.bool)
    wrapper._moe_lora_id_cpu = torch.tensor(0, dtype=torch.long)
    mapping = SimpleNamespace(index_mapping=index_mapping)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [], 2, 100)
    assert wrapper.no_lora is expected_no_lora
