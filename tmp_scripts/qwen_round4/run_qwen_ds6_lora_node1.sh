export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_LORA_MOE_OVERLAP=1

# qwen/ copy of run_ds_lora_prof.sh for the round-4 b3-vs-b4 comparison.
# Deltas vs the original:
#   MODEL   -> 6-layer truncated export (same on both nodes)
#   MTP     -> --speculative-config REMOVED (no MTP, per experiment spec)
#   prof dir-> /vllm-workspace/qwen/logs/prof_node1
# Everything else kept identical so the two nodes' traces are comparable.
# TP=8 kept: matches the production-like b3 config; re-check against bz.

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
  --lora-modules lora-adapter1="/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16" lora-adapter2="/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16" lora-adapter3="/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16" \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/vllm-workspace/qwen/logs/prof_node1","torch_profiler_with_stack":false}'
