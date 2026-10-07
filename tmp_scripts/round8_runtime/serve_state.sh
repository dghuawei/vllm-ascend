#!/bin/bash
# round8 identity test: serve one branch state with the REAL nonzero ckpt.
# usage: serve_state.sh <worktree_path> <log_file>
# Mirrors run_qwen_ds6_lora_node1.sh exactly, except:
#   - lora-modules -> /data/models/DeepSeek-V4-Flash-0731-lora-rank16 (real, nonzero)
#   - profiler-config dropped
#   - PYTHONPATH selects the branch worktree (built .so symlinked inside)
set -x
WT="$1"
LOG="$2"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_LORA_MOE_OVERLAP=1
export PYTHONPATH="$WT"

vllm serve /data/models/DeepSeek-V4-Flash-0731-w8a8-6L \
  --safetensors-load-strategy prefetch \
  --max-model-len 90000 \
  --max-num-batched-tokens 16384 \
  --served-model-name ds \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 8 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 1995 \
  --block-size 32 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": "true","multistream_overlap_shared_expert":false}' \
  --enable-lora \
  --max-loras 3 \
  --max-lora-rank 16 \
  --lora-modules lora-adapter1="/data/models/DeepSeek-V4-Flash-0731-lora-rank16" \
  > "$LOG" 2>&1
