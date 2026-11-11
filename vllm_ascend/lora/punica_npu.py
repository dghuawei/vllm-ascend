# SPDX-License-Identifier: Apache-2.0

import os

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import logger
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend import envs
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.lora_ops import _LORA_WRAPPER_IDS, _LORA_WRAPPERS, lora_linear
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def build_combined_lora_idx_allgather(
    token_lora_mapping: torch.Tensor,
    expert_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Per-row gather index into the flattened [max_loras * num_experts, ...]
    LoRA weight stacks: ``combined_idx[row] = lora_id[row] * num_experts +
    expert_id[row]``, or -1 when the row has no active adapter. Identical to
    the block historically inlined in add_lora_fused_moe; extracted so the
    w13 and w2 applies of one MoE layer can share a single build.
    """
    expert_idx = expert_ids.view(-1).to(torch.long)
    lora_idx_safe = token_lora_mapping.clamp(min=0)
    enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
    return torch.where(
        enabled,
        lora_idx_safe * num_experts + expert_idx,
        torch.full_like(token_lora_mapping, -1),
    ).contiguous()


class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        refresh_all_lora_classes()
        self._max_num_batched_tokens = max_num_batched_tokens
        self._lora_shrink_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._lora_triton_workspaces: dict[int, torch.Tensor] = {}
        self.lora_config = kwargs.get("lora_config")
        ascend_device_type = get_ascend_device_type()
        self.ascend_device_type = ascend_device_type
        if not get_current_hardware_profile().supports(HardwareCapability.LORA_CUSTOM_OPS) or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        ):
            from vllm.lora.ops.torch_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        else:
            from vllm_ascend.lora.lora_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink

        self._single_lora_slot = (
            ascend_device_type in {AscendDeviceType.A2, AscendDeviceType.A3}
            and self.lora_config is not None
            and self.lora_config.max_loras == 1
            and not self.lora_config.fully_sharded_loras
        )
        self._single_lora_mask = None
        self._wrapper_id = next(_LORA_WRAPPER_IDS)
        _LORA_WRAPPERS[self._wrapper_id] = self
        if self._single_lora_slot:
            assert self.lora_config is not None
            lora_dtype = self.lora_config.lora_dtype
            if not isinstance(lora_dtype, torch.dtype):
                raise ValueError(f"LoRA dtype must be resolved before creating the Punica wrapper, got {lora_dtype!r}")
            self._single_lora_mask = torch.empty(
                (max_num_batched_tokens, 1),
                dtype=lora_dtype,
                device=device,
            )

        self._no_lora_cpu = torch.tensor(True, dtype=torch.bool)
        self._prefill_meta_ready = False

        # GroupGEMM is generally much faster for large batches, but kernel
        # launch overhead dominates for small ones, so the kernel choice
        # (gmm vs the in-tree small-batch kernel vs plain bgmv) is made
        # entirely in C++ from shape alone -- see lora_dispatch_kernel in
        # csrc/torch_binding.cpp. These two values are process constants,
        # resolved once here and pushed into C++ via one-time setter calls,
        # never passed as a per-call argument.
        self.gmm_threshold = (
            int(os.environ["LORA_GMM_THRESHOLD"]) if "LORA_GMM_THRESHOLD" in os.environ else max_batches
        )
        logger.warning(
            "LoRA gmm kernels enabled for batches with > %d tokens (LORA_GMM_THRESHOLD); "
            "batches at or below this size use the bgmv path.",
            self.gmm_threshold,
        )
        torch.ops._C_ascend.set_lora_gmm_threshold(self.gmm_threshold)
        torch.ops._C_ascend.set_lora_fast_kernel_enabled(bool(envs.VLLM_ASCEND_ENABLE_ADD_LORA_KERNEL))

    def _update_base_metadata(
        self,
        mapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
    ) -> None:
        super()._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
        if self._single_lora_mask is None:
            return

        token_count = self.indices_len[0]
        assert token_count is not None
        self._single_lora_mask[:token_count].copy_(self._token_lora_indices[:token_count].eq(0).unsqueeze(1))

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        # TODO: when all prefill requests will be served with GMM,
        # switch to self._update_base_metadata
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )

        token_num = len(mapping.index_mapping)
        # Consistent with what the dispatchers decide internally -- see
        # lora_gmm_dispatch in csrc/torch_binding.cpp. Not a prefill-vs-decode
        # phase distinction: this is purely "will the gmm branch be taken for
        # this token count," which the GMM branch needs prefill_metadata
        # (per-sequence seq_len/lora_indices) populated for, and the bgmv/fast
        # branches don't.
        gmm_enabled = bool(torch.ops._C_ascend.lora_gmm_dispatch(token_num))

        if gmm_enabled:
            # TODO: two metadata calls issue
            self._update_prefill_metadata(self.token_lora_indices)
            self.no_lora = bool(self.no_lora)
        else:
            self.no_lora = not any(mapping.index_mapping)

        self._no_lora_cpu.fill_(self.no_lora)

        self.is_prefill = bool(getattr(mapping, "is_prefill", True))
        self._prefill_meta_ready = gmm_enabled

    def add_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM for multiple slices of lora_a. The kernel (gmm / the
        in-tree small-batch kernel / plain bgmv) is chosen inside the
        opaque op from tensor shape alone.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output buffer, shape [n_slices, T, rank].
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """
        x = x.view(-1, x.shape[-1])
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora_dense_shrink(
            y,
            x,
            list(lora_a_stacked),
            lora_indices,
            seq_len,
            self.token_lora_indices,
            scale,
            self._no_lora_cpu,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b. The
        kernel (gmm / the in-tree small-batch kernel / plain bgmv) is
        chosen inside the opaque op from tensor shape alone.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input buffer, shape [n_slices, T, rank] --
                the buffer add_shrink wrote into (post-collective, if the
                caller ran one).
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (Tuple[int, ...]): Every slice's size
            offset_start (int): The starting position of y, defaults to 0
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora_dense_expand(
            y,
            x,
            list(lora_b_stacked),
            lora_indices,
            seq_len,
            self.token_lora_indices,
            list(output_slices),
            offset_start,
            add_inputs,
            self._no_lora_cpu,
        )
        y = y.view_as(y_org)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """
        if self.no_lora:
            return
        x = x.to(torch.float32)
        if self._prefill_meta_ready:
            self.sgmv_expand(x, lora_b_stacked, y, *self.prefill_metadata, add_inputs)
        else:
            self.bgmv_expand(x, lora_b_stacked, y, self.token_lora_indices, add_inputs)

    def add_lora(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Performs the full `y += (x @ lora_a) @ lora_b * scale` LoRA apply as
        a shrink followed by an expand. The kernel for each step (gmm / the
        in-tree small-batch kernel / plain bgmv) is chosen inside the
        opaque ops from tensor shape alone -- there is no separate
        "fused"/atomic op to pick between here.
        """
        y2d = y.view(-1, y.shape[-1])
        x2d = x.view(-1, x.shape[-1])
        rank = lora_b_stacked[0].size(-1) if lora_b_stacked else 0
        buf = torch.zeros(
            (len(lora_a_stacked), x2d.size(0), rank), dtype=torch.float32, device=x2d.device
        )
        self.add_shrink(buf, x2d, lora_a_stacked, scale, **kwargs)
        self.add_expand(
            y2d, buf, lora_b_stacked, output_slices, offset_start=offset_start, add_inputs=add_inputs, **kwargs
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        add_inputs: bool = True,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0) @ lora_a_stacked[
                    indices[i], layer_idx, :, :] @ lora_b_stacked[
                    indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        # `buffer` is part of the abstract interface but unused here: no
        # real caller ever passes it non-None, and there is no longer a
        # manual two-op path that would need an externally-supplied one.
        del buffer

        if self._single_lora_slot and not self.no_lora:
            # Single active adapter: the registry dispatch routes to a
            # plain matmul with the adapter mask (or the triton packed
            # path on A3), which beats the generic punica/fused dispatch.
            lora_linear(
                self._wrapper_id,
                y,
                x,
                list(lora_a_stacked),
                list(lora_b_stacked),
                scale,
                list(output_slices),
                kwargs.get("packed_lora_a"),
                kwargs.get("packed_lora_b"),
                kwargs.get("add_inputs", add_inputs),
            )
            return

        self.add_lora(y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, add_inputs=add_inputs, **kwargs)

    def _lora_linear_kernel(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        scale: float,
        output_slices: tuple[int, ...] | list[int],
    ) -> None:
        lora_a = tuple(lora_a_stacked)
        lora_b = tuple(lora_b_stacked)
        slices = tuple(output_slices)

        r = lora_b[0].size(-1)
        buffer = self._get_shrink_buffer(len(slices), x.size(0), r, x.device)
        self.add_shrink(buffer, x, lora_a, scale)
        self.add_expand(y, buffer, lora_b, slices, add_inputs=True)

    def _get_shrink_buffer(
        self,
        n_slices: int,
        num_tokens: int,
        rank: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (n_slices, rank)
        buf = self._lora_shrink_buffers.get(key)
        if buf is None or buf.shape[1] < num_tokens:
            alloc_rows = max(num_tokens, self._max_num_batched_tokens)
            buf = torch.zeros(
                (n_slices, alloc_rows, rank),
                dtype=torch.float32,
                device=device,
            )
            self._lora_shrink_buffers[key] = buf
        return buf[:, :num_tokens]

    def _lora_linear_matmul(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        scale: float,
        output_slices: tuple[int, ...] | list[int],
        packed_lora_a: torch.Tensor | None,
        packed_lora_b: torch.Tensor | None,
        add_inputs: bool,
    ) -> None:
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        assert self._single_lora_mask is not None
        adapter_mask = self._single_lora_mask[: x.size(0)]

        if len(lora_b_stacked) == 1:
            a_mat = lora_a_stacked[0][0, 0]
            b_mat = lora_b_stacked[0][0, 0, : output_slices[0]].transpose(0, 1)
        else:
            assert packed_lora_a is not None and packed_lora_b is not None
            a_mat = packed_lora_a[0, 0]
            b_mat = packed_lora_b[0, 0]

        shrink = torch.matmul(x, a_mat.transpose(0, 1)).mul_(adapter_mask)
        if scale != 1.0:
            shrink.mul_(scale)
        delta = torch.matmul(shrink, b_mat)
        if add_inputs:
            y.add_(delta)
        else:
            y.copy_(delta)

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        *,
        topk_weights: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_post_padded: torch.Tensor | None = None,
        max_lora_rank: int = 0,
        top_k_num: int = 1,
        shrink_config=None,
        expand_config=None,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        partial_expand: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
        group_list: torch.Tensor | None = None,
        group_list_type: int = 1,
        combined_idx: torch.Tensor | None = None,
        overwrite: bool = False,
        phase: str = "full",
        shrink_buffers: tuple[list[torch.Tensor], bool] | None = None,
    ) -> tuple[list[torch.Tensor], bool] | None:
        """
        Ascend-native fused MoE LoRA (v2): static-shape per-row gather via the
        same bgmv_shrink/bgmv_expand AscendC kernels (csrc/kernels/bgmv_*.cpp)
        used by the dense Linear LoRA layers, instead of grouping rows by a
        data-dependent ``torch.unique`` over active LoRA ids. The previous
        ``torch.unique``/``nonzero`` version produced output whose *shape*
        depended on tensor values, which ACL Graph capture cannot record
        (it failed with an `aclnnUnique2` error as soon as `enforce_eager`
        was turned off) -- every tensor below has a shape that depends only
        on input shapes, never on values, so this stays graph-capturable.

        Return contract: every path folds the delta into ``y`` in place and
        returns None -- the gmm-expand-cannot-accumulate-into-its-own-output
        case (fully-sharded, slices cover the full width of ``y``) is handled
        entirely inside add_lora_moe_expand now (a scratch buffer + explicit
        fold), so it never surfaces to Python. ``phase="shrink"`` returns
        ``(buffers, used_gmm)`` -- the intermediate shrink buffers plus which
        kernel layout they're in, needed to pass back into ``phase="expand"``.

        Rows are already one-token-per-row (top_k_num=1). Each row needs the
        LoRA slot for (lora_id, expert_id), so we fold both into a single
        gather index into a ``[max_loras * num_experts, ...]`` view of the
        existing per-(lora, expert) weight stacks:
            combined_idx[row] = lora_id[row] * num_experts + expert_id[row]
        or -1 when the row has no active adapter, mirroring the -1 sentinel
        ``PunicaWrapperBase.token_lora_indices`` already uses. bgmv_shrink/
        bgmv_expand skip any row whose index is negative (leaving the
        zero-initialized shrink buffer / unmodified ``y`` in place), so
        inactive rows get a zero delta for free -- no Python-level branching
        needed.

        overwrite=True writes the LoRA delta into ``y`` without accumulating
        (used by the aux-stream overlap path: ``y`` is then an isolated delta
        buffer the caller zeroes and folds into the GEMM output afterwards).
        Only supported on the decode fused/bgmv path.

        partial_expand: fully-sharded w2-style apply only (input-sharded A,
        full rank). Skips the dedicated shrink all-reduce and expands this
        rank's partial sum into its tp_rank output slice; the MoE runner's
        final TP all-reduce completes the delta
        (sum_r(z_r @ B_r^T) == z_full @ B_r^T after reduction). The caller
        must guarantee the final reduction covers the expanded tensor.

        phase: split execution of the two-op (group_list) branch so the aux
        stream can overlap the shrink with the base GEMM:
          "shrink"  -- run add_lora_shrink only and return the buffers (no
                       collective); the caller later passes them back with
          "expand"  -- run the cross-rank collective (unless folded) and
                       add_lora_expand into y.
        Only the group_list branch supports splitting (the fused single-op
        path cannot host the collective between shrink and expand anyway).
        """
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if token_lora_mapping is None and combined_idx is None:
            # Only needed for the index build; the prebuilt-combined path
            # does not touch it (and bare test wrappers may not carry it).
            token_lora_mapping = self.token_lora_indices

        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        if combined_idx is None:
            if expert_ids is None or token_lora_mapping is None:
                raise AssertionError(
                    "add_lora_fused_moe requires either combined_idx or (expert_ids, token_lora_mapping)"
                )
            # moe_lora_apply_w13/w2 share one routing per layer
            combined_idx = build_combined_lora_idx_allgather(
                token_lora_mapping,
                expert_ids,
                adapter_enabled,
                lora_a_stacked[0].shape[1],
            )

        if (
            not fully_sharded
            and not (mul_routed_weight and topk_weights is not None)
            and not bool(torch.ops._C_ascend.lora_gmm_dispatch(x2d.size(0)))
        ):
            # Single shrink+expand pass, no collective in between. Guarded
            # explicitly so the gmm branch is never internally chosen here
            # (checked with the same query update_metadata uses, for the
            # same num_tokens the ops will see): this path passes
            # combined_idx in place of group_list, which is only valid
            # because gmm -- the only branch that would actually read it as
            # per-expert row counts -- is guaranteed not to run. The general
            # two-op branch below handles any call where gmm could be
            # chosen, with the real group_list.
            output_slices_fused = [b.shape[-2] for b in lora_b_stacked]
            buffers, used_gmm = torch.ops._C_ascend.add_lora_moe_shrink(
                x2d, list(lora_a_stacked), combined_idx, combined_idx, 1.0,
                self._no_lora_cpu, False, group_list_type,
            )
            torch.ops._C_ascend.add_lora_moe_expand(
                y2d, buffers, list(lora_b_stacked), combined_idx, combined_idx,
                output_slices_fused, offset, not overwrite, used_gmm, self._no_lora_cpu,
                group_list_type,
            )
            return None

        if overwrite:
            raise ValueError(
                "overwrite=True (delta-buffer mode) is only supported on the "
                "decode fused/bgmv path (AllGather, non-fully-sharded, no "
                "routed-weight fold)"
            )

        if group_list is not None and not mul_routed_weight:
            n_slices = len(lora_a_stacked)
            max_loras = lora_a_stacked[0].shape[0]
            local_rank = lora_a_stacked[0].shape[-2]
            full_rank = lora_b_stacked[0].shape[-1]
            rank_sharded = local_rank != full_rank
            if phase == "expand":
                if shrink_buffers is None:
                    raise ValueError('phase="expand" requires shrink_buffers from phase="shrink"')
                buffers, used_gmm = shrink_buffers
            else:
                buffers, used_gmm = torch.ops._C_ascend.add_lora_moe_shrink(
                    x2d, list(lora_a_stacked), combined_idx, group_list, 1.0,
                    self._no_lora_cpu, fully_sharded, group_list_type,
                )
                if phase == "shrink":
                    return (buffers, used_gmm)
            output_slices = [b.shape[-2] for b in lora_b_stacked]
            # partial_expand folds the cross-rank completion of a w2-style
            # apply into the MoE runner's final TP all-reduce: each rank
            # expands its own partial shrink (input-sharded A) into its
            # tp_rank output slice, and sum(rank partials) == full delta
            # after the final reduction. Only valid for the non-rank-sharded
            # (w2) case and never under EP, where the FS sharding group and
            # the combine collectives do not match.
            fold_partial = partial_expand and not rank_sharded
            if fully_sharded and not fold_partial and get_tensor_model_parallel_world_size() > 1:
                # The C++ shrink writes the per-slice [T, L*R_local] buffers on
                # the gmm branch and the single stacked [S, T, R_local] fp32
                # buffer (buffers[0]) on the bgmv/fast branch; the collective
                # must target the one that was actually written. Rank shards
                # are contiguous by tp_rank (slice_lora_a), so an all_gather
                # along the last dim concatenates them in the correct order.
                # The gmm per-slice buffer carries L lora blocks per row, so
                # it is viewed [T, L, R_local] first to keep the blocks from
                # interleaving; the stacked buffer needs no such view.
                if used_gmm:
                    if rank_sharded:
                        # w13-style: A is rank-sharded; gather the rank axis.
                        T = x2d.shape[0]
                        for s in range(n_slices):
                            v = buffers[s].view(T, max_loras, local_rank)
                            buffers[s] = tensor_model_parallel_all_gather(v, dim=-1).reshape(T, max_loras * full_rank)
                    else:
                        # w2-style: A is unsharded; shrink outputs are partial
                        # sums over this rank's input shard.
                        for s in range(n_slices):
                            buffers[s] = tensor_model_parallel_all_reduce(buffers[s])
                else:
                    stacked = buffers[0]
                    if rank_sharded:
                        buffers[0] = tensor_model_parallel_all_gather(stacked, dim=-1)
                    else:
                        buffers[0] = tensor_model_parallel_all_reduce(stacked)
            # add_lora_moe_expand folds the delta into y in place and always
            # returns None -- the former gmm-expand-cannot-accumulate full-
            # width special case (add_lora_expand_delta) is now handled
            # entirely inside it.
            torch.ops._C_ascend.add_lora_moe_expand(
                y2d,
                buffers,
                list(lora_b_stacked),
                combined_idx,
                group_list,
                output_slices,
                offset,
                True,
                used_gmm,
                self._no_lora_cpu,
                group_list_type,
            )
            return None

        if phase != "full":
            raise ValueError('phase="shrink"/"expand" is only supported on the two-op (group_list) branch')
        cur_offset = offset
        for slice_idx in range(len(lora_a_stacked)):
            # lora_a_stacked[s]/lora_b_stacked[s]: [max_loras, num_experts, rank, *].
            # Flattening the leading two dims turns "gather by (lora, expert)"
            # into "the plain per-row gather" to reuse bgmv_shrink/bgmv_expand.
            a = lora_a_stacked[slice_idx]
            b = lora_b_stacked[slice_idx]
            local_rank = a.shape[-2]
            full_rank = b.shape[-1]
            out_size = b.shape[-2]
            a_flat = a.view(-1, local_rank, a.shape[-1])

            # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32
            # (its X_T), so the shrink buffer is fp32.
            shrink_out = torch.zeros(
                (x2d.shape[0], local_rank),
                dtype=torch.float32,
                device=x2d.device,
            )

            self.bgmv_shrink(x2d, a_flat, shrink_out, combined_idx, 1.0)

            if fully_sharded:
                if local_rank == full_rank:
                    shrink_out = tensor_model_parallel_all_reduce(shrink_out)
                else:
                    shrink_out = tensor_model_parallel_all_gather(shrink_out)

            if shrink_out.shape[-1] != full_rank:
                raise ValueError(
                    "MoE LoRA rank mismatch after TP communication: "
                    f"A projection has rank {shrink_out.shape[-1]}, "
                    f"but LoRA B expects rank {full_rank}."
                )
            b_flat = b.view(-1, out_size, full_rank)

            delta = shrink_out
            if mul_routed_weight and topk_weights is not None:
                delta = shrink_out * topk_weights.view(-1, 1)

            self.bgmv_expand_slice(delta, b_flat, y2d, combined_idx, cur_offset, out_size, add_inputs=True)
            cur_offset += out_size

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))

        if y.size(1) < r:
            # Ascend bgmv_expand requires hidden_out >= hidden_in (LoRA rank).
            # The sequence-classification LoRA head (#53555) can have
            # num_labels < rank, so use the matmul fallback.
            self._add_lora_logits_matmul(y, x, lora_a_stacked, lora_b_stacked, scale, indices)
            y = y.view_as(y_org)
            return

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        if y.dtype in (torch.half, torch.bfloat16):
            self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)
        else:
            # Ascend bgmv_expand only writes half/bf16 outputs. Compute
            # the LoRA delta into a workspace of the weight dtype (matching
            # the kernel's output dtype) and merge it back, so fp32
            # classification outputs (classifier LoRA, #53555) work on
            # vLLM main without quantizing the base logits.
            op_dtype = lora_b_stacked.dtype
            delta = torch.zeros(y.shape, dtype=op_dtype, device=y.device)
            self.bgmv_expand(buffer, lora_b_stacked, delta, indices, add_inputs=True)
            y.add_(delta.to(y.dtype))

        y = y.view_as(y_org)

    def _add_lora_logits_matmul(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale: float,
        indices: torch.Tensor,
    ) -> None:
        """Per-adapter matmul fallback for add_lora_logits (see call site)."""
        active = indices >= 0
        idx = indices.clamp(min=0)
        lora_a = lora_a_stacked[idx]
        lora_b = lora_b_stacked[idx]
        if lora_a.ndim == 4:
            lora_a = lora_a[:, 0]
        if lora_b.ndim == 4:
            lora_b = lora_b[:, 0]

        delta = torch.bmm(x.to(torch.float32).unsqueeze(1), lora_a.to(torch.float32).transpose(1, 2)).squeeze(1)
        delta = torch.bmm(delta.unsqueeze(1), lora_b.to(torch.float32).transpose(1, 2)).squeeze(1)
        delta.mul_(scale)
        delta.masked_fill_(~active.unsqueeze(1), 0)
        y.add_(delta.to(y.dtype))
