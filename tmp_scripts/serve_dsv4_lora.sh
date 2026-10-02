#!/bin/bash
# Serve DeepSeek-V4-Flash w8a8 + MoE LoRA adapter for the e2e A/B of the fused
# w13 epilogue. Paths point at this host's copies (the qlora scripts assume
# /data/models, which this container does not mount).
#
# Speculative decoding (dspark) is deliberately left OFF: the fused epilogue
# only engages on the gmm/prefill path, so spec-decode would add decode-side
# variance to an experiment about prefill.
set -x

export VLLM_DISABLE_COMPILE_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

source /usr/local/Ascend/ascend-toolkit/set_env.sh

MODEL=/home/russia_mmo/models/dsv4-w8a8-mtp
LORA=/home/russia_mmo/models/lora-dsv4-w8a8-mtp-r8

vllm serve "$MODEL" \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --served-model-name ds \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 4 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 1995 \
  --block-size 32 \
  --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": "true","multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false}' \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 8 \
  --lora-modules lora-adapter="$LORA"
