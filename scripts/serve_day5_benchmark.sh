#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-inference-lab
source .venv-serve/bin/activate

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1

method="${1:?Usage: bash scripts/serve_day5_benchmark.sh BF16|Random32}"
quantization_args=()

case "$method" in
    BF16)
        model="./models/Qwen3-4B-Instruct-2507"
        alias="qwen-bench-bf16"
        ;;
    Random32)
        model="./models/Qwen3-4B-Instruct-2507-FP8-Random32-seed42"
        alias="qwen-bench-fp8"
        quantization_args=(--quantization modelopt)
        ;;
    *)
        echo "Unknown method: $method" >&2
        exit 1
        ;;
esac

test -f "$model/config.json"

python - <<'PY'
import socket

with socket.socket() as connection:
    connection.settimeout(1)
    if connection.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit(
            "Port 8000 is occupied. Stop the previous server first."
        )
PY

exec vllm serve "$model" \
    --served-model-name "$alias" \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --kv-cache-memory-bytes 4294967296 \
    --attention-backend FLASH_ATTN \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 1 \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --seed 42 \
    --host 127.0.0.1 \
    --port 8000 \
    "${quantization_args[@]}"
