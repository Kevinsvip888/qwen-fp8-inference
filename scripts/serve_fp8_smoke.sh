#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-inference-lab
source .venv-serve/bin/activate

export HF_HOME="/root/autodl-tmp/qwen-inference-lab/.cache/huggingface"

exec vllm serve ./models/Qwen3-4B-Instruct-2507-FP8-smoke \
    --served-model-name qwen-fp8 \
    --quantization modelopt \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --seed 42 \
    --host 127.0.0.1 \
    --port 8000
