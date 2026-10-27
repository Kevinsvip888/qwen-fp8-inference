#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-inference-lab
source .venv-serve/bin/activate

method="${1:?Usage: serve_fp8_experiment.sh METHOD [POOL_SEED]}"
pool_seed="${2:-42}"

case "$method" in
    Random32|Selected32|Random128) ;;
    *) echo "Unknown method: $method"; exit 1 ;;
esac

[[ "$pool_seed" =~ ^[0-9]+$ ]] || {
    echo "POOL_SEED must be an integer"
    exit 1
}

model="./models/Qwen3-4B-Instruct-2507-FP8-${method}-seed${pool_seed}"
[[ -f "$model/calibration_manifest.json" ]] || {
    echo "Missing checkpoint manifest: $model"
    exit 1
}

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1

vllm serve "$model" \
    --served-model-name "qwen-${method,,}" \
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
    --port 8000 \
    2>&1 | tee "logs/serve_${method}_seed${pool_seed}.log"
