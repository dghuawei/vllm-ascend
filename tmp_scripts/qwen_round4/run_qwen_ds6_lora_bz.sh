export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_LORA_MOE_OVERLAP=1

# qwen/ round-6 serve for the bz-ascend (910B4) side of the b3-vs-b4 comparison.
# Byte-for-byte the same flags as run_qwen_ds6_lora_node1.sh except paths:
#   MODEL   -> /home/russia_mmo/models/dsv4-w8a8-mtp  (6L truncated, reconciled
#             to print 6 [0, 0, 4, 128, 4, 128] 0 on 2026-10-07)
#   LORA    -> zero-rank16 adapter regenerated on bz with gen_zero_lora_ds.py
#   prof dir-> /home/russia_mmo/qwen/logs/prof_bz (root fs of container has
#             only 19G; /home/russia_mmo is bind-mounted into moegmm_rc, 5.6T)
# TP=8: all 8x910B4 healthy and idle at prep time.
#
# 2026-10-07 OOM fix (capture attempt 1): bz boards are 32768 MiB/card (node1
# 910B3 boards are 65536 MiB). With bare --gpu-memory-utilization 0.93 the auto
# KV pool (17.45 GiB) overshot: components summed 28.07 GiB vs the 27.42 GiB
# budget -> engine NPU-OOM during the capture warmup. --kv-cache-memory below is
# vLLM's OWN suggested value from that log line (16.66 GiB, fits the 0.93
# budget). Chosen over lowering util: sanctioned mechanism, and it pins the KV
# pool to the same absolute size on both nodes. Pool is still ~1.8M tokens >>
# the 8x8192-token capture, so no request behavior changes. gmu 0.93 kept.

vllm serve /home/russia_mmo/models/dsv4-w8a8-mtp \
  --safetensors-load-strategy prefetch \
  --max-model-len 90000 \
  --max-num-batched-tokens 16384 \
  --served-model-name ds \
  --gpu-memory-utilization 0.93 \
  --kv-cache-memory 17080836097 \
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
  --lora-modules lora-adapter1="/home/russia_mmo/qwen/lora-zero-rank16" lora-adapter2="/home/russia_mmo/qwen/lora-zero-rank16" lora-adapter3="/home/russia_mmo/qwen/lora-zero-rank16" \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/russia_mmo/qwen/logs/prof_bz","torch_profiler_with_stack":false}'
