import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", required=True,
    choices=["Random32", "Selected32", "Random128"],
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()

def freeze(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing protocol differs: {path}")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

pool_dir = ROOT / f"data/calibration/seed_{args.seed}"
pool_bytes = (pool_dir / "pool.jsonl").read_bytes()
pool_hash = digest(pool_bytes)
pool = {
    item["id"]: item
    for item in map(json.loads, pool_bytes.splitlines())
}
pool_manifest = read_json(pool_dir / "manifest.json")
audit = read_json(pool_dir / "pool_audit.json")

assert len(pool) == 512
assert pool_hash == pool_manifest["pool_sha256"]
assert pool_hash == audit["pool_sha256"]
assert audit["source_tokens_match"]
assert audit["math_reserved_rows_excluded"]

if args.method == "Selected32":
    subset_manifest = read_json(pool_dir / "selected32_manifest.json")
    entry = subset_manifest
else:
    subset_manifest = read_json(pool_dir / "random_subsets_manifest.json")
    entry = subset_manifest["subsets"][args.method]

assert subset_manifest["pool_sha256"] == pool_hash
subset_path = pool_dir / entry["file"]
subset_bytes = subset_path.read_bytes()
subset_hash = digest(subset_bytes)
assert subset_hash == entry["sha256"]

blocks = [json.loads(line) for line in subset_bytes.splitlines()]
ids = [block["id"] for block in blocks]
expected_count = 128 if args.method == "Random128" else 32

assert ids == entry["sequence_ids"]
assert len(ids) == len(set(ids)) == expected_count
assert all(block == pool[block["id"]] for block in blocks)
assert all(len(block["input_ids"]) == 512 for block in blocks)
assert Counter(block["domain"] for block in blocks) == {
    "general": expected_count // 2,
    "math": expected_count // 2,
}

source = read_json(ROOT / "logs/model_revision.json")
assert source == pool_manifest["model_source"]

versions = {
    name: importlib.metadata.version(name)
    for name in ("torch", "transformers", "nvidia-modelopt", "safetensors")
}
assert versions["nvidia-modelopt"] == "0.47.0"

# Preserve the default exclusion rules and modify existing FP8 entries.
recipe = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
assert recipe["algorithm"] == "max"
assert isinstance(recipe["quant_cfg"], list)

modified = set()
for rule in recipe["quant_cfg"]:
    name = rule.get("quantizer_name")
    if name in ("*weight_quantizer", "*input_quantizer"):
        rule["cfg"]["type"] = "static"
        modified.add(name)
assert modified == {"*weight_quantizer", "*input_quantizer"}

protocol = {
    "source": source,
    "packages": versions,
    "recipe": recipe,
    "model_dtype": "bfloat16",
    "attention_backend": "eager",
    "batch_size": 1,
    "sequence_length": 512,
    "use_cache": False,
    "torch_seed": 42,
}
protocol_bytes = json_bytes(protocol)
protocol_hash = digest(protocol_bytes)
freeze(ROOT / "data/calibration/fp8_protocol.json", protocol_bytes)

identity = {
    "method": args.method,
    "pool_seed": args.seed,
    "pool_sha256": pool_hash,
    "subset_sha256": subset_hash,
    "protocol_sha256": protocol_hash,
    "sequence_ids": ids,
    "calibration_sequences": expected_count,
    "calibration_tokens": expected_count * 512,
}

destination = ROOT / (
    "models/Qwen3-4B-Instruct-2507-FP8-"
    f"{args.method}-seed{args.seed}"
)

# Completed exports can be reused after verifying their identity and files.
if destination.exists():
    previous = read_json(destination / "calibration_manifest.json")
    assert previous["identity"] == identity
    for entry in previous["export_files"]:
        path = destination / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert file_hash(path) == entry["sha256"]
    print(f"PASS: existing {args.method} checkpoint verified.")
    print("Checkpoint:", destination)
    raise SystemExit(0)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

free_bytes, _ = torch.cuda.mem_get_info()
print(f"Free GPU memory: {free_bytes / 1024**3:.2f} GiB", flush=True)
if free_bytes < 16 * 1024**3:
    raise RuntimeError("Need 16 GiB free GPU memory; stop serving first.")
if shutil.disk_usage(ROOT / "models").free < 8 * 1024**3:
    raise RuntimeError("Need at least 8 GiB free disk space for this export.")

torch.manual_seed(42)
model_dir = ROOT / "models/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, local_files_only=True
)

print(f"Loading fresh BF16 model for {args.method}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="eager",
    local_files_only=True,
).eval()
model.requires_grad_(False)

seen = []

def forward_loop(quantized_model):
    with torch.no_grad():
        for number, block in enumerate(blocks, start=1):
            input_ids = torch.tensor(
                [block["input_ids"]],
                dtype=torch.long,
                device="cuda",
            )
            output = quantized_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            del output, input_ids
            seen.append(block["id"])

            if number == 1 or number % 8 == 0:
                print(
                    f"Calibration {number}/{len(blocks)}",
                    flush=True,
                )

print("Starting static FP8 calibration...", flush=True)
torch.cuda.synchronize()
start = time.perf_counter()
model = mtq.quantize(model, copy.deepcopy(recipe), forward_loop)
torch.cuda.synchronize()
calibration_seconds = time.perf_counter() - start
assert seen == ids

# Record scalar activation ranges before export.
activation_amax = {}
for name, module in model.named_modules():
    if not name.endswith("input_quantizer"):
        continue
    value = getattr(module, "amax", None)
    if value is None:
        continue

    value = torch.as_tensor(value).detach().float().cpu()
    assert value.numel() == 1, name
    assert torch.isfinite(value).all(), name
    assert (value > 0).all(), name
    activation_amax[name] = value.item()

assert len(activation_amax) == 252, len(activation_amax)
print("Calibrated FP8 input quantizers:", len(activation_amax))

# Export into a temporary directory; publish its final name after checks.
building = Path(tempfile.mkdtemp(
    prefix=f"{destination.name}-building-",
    dir=destination.parent,
))
print("Exporting checkpoint...", flush=True)
print("Temporary export directory:", building, flush=True)

torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=str(building),
    )
tokenizer.save_pretrained(building)
torch.cuda.synchronize()
export_seconds = time.perf_counter() - start

quant_config = read_json(building / "hf_quant_config.json")
assert quant_config["quantization"]["quant_algo"] == "FP8"
assert quant_config["quantization"].get("kv_cache_quant_algo") is None

weight_files = sorted(building.glob("*.safetensors"))
assert weight_files

tensor_counts = Counter()
for path in weight_files:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor_counts[handle.get_slice(name).get_dtype()] += 1

assert tensor_counts["F8_E4M3"] == 252, dict(tensor_counts)

print("Hashing exported files...", flush=True)
export_files = [
    {
        "path": str(path.relative_to(building)),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
    }
    for path in sorted(building.rglob("*"))
    if path.is_file()
]

manifest = {
    "identity": identity,
    "protocol": json.loads(protocol_bytes),
    "activation_amax": activation_amax,
    "tensor_counts_by_dtype": dict(tensor_counts),
    "safetensors_bytes": sum(p.stat().st_size for p in weight_files),
    "calibration_seconds": calibration_seconds,
    "export_seconds": export_seconds,
    "export_files": export_files,
}
(building / "calibration_manifest.json").write_bytes(json_bytes(manifest))
building.rename(destination)

print("\nMethod:", args.method)
print("Calibration sequences:", expected_count)
print("Calibration tokens:", expected_count * 512)
print("Calibrated input quantizers:", len(activation_amax))
print("Exported FP8 tensors:", tensor_counts["F8_E4M3"])
print(f"Calibration seconds: {calibration_seconds:.2f}")
print(f"Export seconds: {export_seconds:.2f}")
print("Protocol SHA256:", protocol_hash)
print("Checkpoint:", destination)
print(f"PASS: {args.method} FP8 export verified.")
