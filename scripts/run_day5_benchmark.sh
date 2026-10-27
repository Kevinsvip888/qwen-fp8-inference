#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-inference-lab
source .venv-serve/bin/activate

method="${1:?Specify BF16 or Random32}"
repetition="${2:?Specify repetition 1, 2, or 3}"
input_len="${3:-1024}"
concurrency="${4:-4}"

case "$method" in
    BF16) alias="qwen-bench-bf16" ;;
    Random32) alias="qwen-bench-fp8" ;;
    *) echo "Unknown method: $method" >&2; exit 1 ;;
esac

case "$repetition" in
    1|2|3) ;;
    *) echo "Repetition must be 1, 2, or 3" >&2; exit 1 ;;
esac

case "$input_len" in
    256|1024|2048) ;;
    *) echo "Unsupported input length" >&2; exit 1 ;;
esac

case "$concurrency" in
    1|4) ;;
    *) echo "Concurrency must be 1 or 4" >&2; exit 1 ;;
esac

parent="outputs/benchmark/day5/formal_v1"
run="$parent/$method/isl${input_len}_osl128_c${concurrency}/rep${repetition}"

mkdir -p "$(dirname "$run")"
if [[ -e "$run" ]]; then
    echo "Run directory already exists; preserving it: $run" >&2
    exit 1
fi

python - "$alias" <<'PY'
import json
import sys
from urllib.request import build_opener, ProxyHandler

opener = build_opener(ProxyHandler({}))
with opener.open("http://127.0.0.1:8000/health", timeout=10) as response:
    assert response.status == 200

with opener.open("http://127.0.0.1:8000/v1/models", timeout=10) as response:
    models = json.load(response)

available = [item["id"] for item in models["data"]]
assert available == [sys.argv[1]], (
    f"Expected {sys.argv[1]}, server exposes {available}"
)
print("PASS: correct server alias:", sys.argv[1])
PY

mkdir "$run"

uv pip freeze > "$run/environment.txt"
nvidia-smi > "$run/gpu_before.txt"

python - "$parent" "$run" <<'PY'
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

parent, run = map(Path, sys.argv[1:])
parent.mkdir(parents=True, exist_ok=True)

gpu_uuid = subprocess.check_output(
    [
        "nvidia-smi", "--id=0", "--query-gpu=uuid",
        "--format=csv,noheader",
    ],
    text=True,
).strip()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

protocol = {
    "study": "day5_formal_v1",
    "gpu_uuid": gpu_uuid,
    "packages": {
        name: version(name)
        for name in ("torch", "vllm", "transformers")
    },
    "environment_sha256": sha256(run / "environment.txt"),
    "server_script_sha256": sha256("scripts/serve_day5_benchmark.sh"),
    "runner_script_sha256": sha256("scripts/run_day5_benchmark.sh"),
    "source_revision": json.loads(
        Path("logs/model_revision.json").read_text()
    )["revision"],
    "fp8_manifest_sha256": sha256(
        "models/Qwen3-4B-Instruct-2507-FP8-Random32-seed42/"
        "calibration_manifest.json"
    ),
    "requests": 200,
    "warmup_requests": 20,
    "output_tokens": 128,
    "workload_seed": 42,
    "generation_seed": 42,
    "temperature": 0,
    "top_p": 1,
    "ignore_eos": True,
    "kv_cache_bytes": 4294967296,
    "kv_cache_dtype": "bfloat16",
    "attention_backend": "FLASH_ATTN",
    "prefix_caching": False,
}

path = parent / "protocol.json"
if path.exists():
    assert json.loads(path.read_text()) == protocol, (
        "Protocol, environment, scripts, or GPU changed."
    )
else:
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

print("PASS: formal benchmark protocol recorded/verified.")
PY

monitor_pid=""

cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=""
    fi
}
trap cleanup EXIT

nvidia-smi --id=0 \
    --query-gpu=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits \
    --loop-ms=1000 \
    > "$run/gpu_samples.csv" \
    2> "$run/gpu_monitor.log" &
monitor_pid=$!

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HUB_OFFLINE=1 \
    vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model "$alias" \
    --served-model-name "$alias" \
    --tokenizer ./models/Qwen3-4B-Instruct-2507 \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len 128 \
    --random-range-ratio 0 \
    --random-prefix-len 0 \
    --num-prompts 200 \
    --num-warmups 20 \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --ignore-eos \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --extra-body '{"seed":42,"add_special_tokens":false}' \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,95 \
    --save-result \
    --save-detailed \
    --result-dir "$run" \
    --result-filename benchmark.json \
    2>&1 | tee "$run/client.log"

cleanup
nvidia-smi > "$run/gpu_after.txt"

python - "$run" "$input_len" <<'PY'
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
input_len = int(sys.argv[2])
report = json.loads((run / "benchmark.json").read_text())

assert report["completed"] == 200
assert report["input_lens"] == [input_len] * 200
assert report["output_lens"] == [128] * 200

memory_samples = []
with (run / "gpu_samples.csv").open() as handle:
    for row in csv.reader(handle):
        if len(row) == 6:
            try:
                memory_samples.append(float(row[2]))
            except ValueError:
                pass

assert memory_samples, "No valid GPU-memory samples recorded"

memory = {
    "sample_interval_seconds": 1,
    "scope": "whole GPU during benchmark client, including warmup",
    "samples": len(memory_samples),
    "sampled_min_used_mib": min(memory_samples),
    "sampled_max_used_mib": max(memory_samples),
}
(run / "memory_summary.json").write_text(
    json.dumps(memory, indent=2) + "\n"
)

print("\nPASS: 200 successful requests; exact input/output lengths.")
print("Sampled maximum GPU memory:", max(memory_samples), "MiB")
print("Saved:", run)
PY
