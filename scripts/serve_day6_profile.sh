#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-inference-lab
source .venv-serve/bin/activate

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1

method="${1:?Specify BF16 or Random32}"
quantization_args=()

case "$method" in
    BF16)
        model="./models/Qwen3-4B-Instruct-2507"
        alias="qwen-profile-bf16"
        ;;
    Random32)
        model="./models/Qwen3-4B-Instruct-2507-FP8-Random32-seed42"
        alias="qwen-profile-fp8"
        quantization_args=(--quantization modelopt)
        ;;
    *)
        echo "Unknown method: $method" >&2
        exit 1
        ;;
esac

python - <<'PY'
import socket

with socket.socket() as connection:
    connection.settimeout(1)
    if connection.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit("Port 8000 is occupied. Stop the previous server.")
PY

stamp=$(date -u +%Y%m%dT%H%M%S)
trace_dir="$PWD/outputs/profiling/day6/${method}_${stamp}"
mkdir -p "$trace_dir"

printf '%s\n' "$trace_dir" \
    > "logs/profiling/day6/${method}_trace_dir.txt"

profiler_config=$(python - "$trace_dir" <<'PY'
import json
import sys

print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": sys.argv[1],
    "torch_profiler_activities": ["CPU", "CUDA"],
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,
    "torch_profiler_use_gzip": True,
    "ignore_frontend": True,
    "max_iterations": 20
}))
PY
)

printf 'Trace directory: %s\n' "$trace_dir"

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
    --profiler-config "$profiler_config" \
    "${quantization_args[@]}"
