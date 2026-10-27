import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save_json(path, value):
    content = (json.dumps(value, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing file differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

pool_dir = ROOT / f"data/calibration/seed_{args.seed}"
payload = (pool_dir / "pool.jsonl").read_bytes()
pool_hash = digest(payload)
pool_manifest = json.loads((pool_dir / "manifest.json").read_text())
audit = json.loads((pool_dir / "pool_audit.json").read_text())

assert pool_hash == pool_manifest["pool_sha256"]
assert pool_hash == audit["pool_sha256"]
assert audit["verified_sequences"] == 512
assert audit["source_tokens_match"]
assert audit["math_reserved_rows_excluded"]

pool = [json.loads(line) for line in payload.splitlines()]
assert len(pool) == 512
assert len({item["id"] for item in pool}) == 512
assert all(len(item["input_ids"]) == 512 for item in pool)

model_dir = ROOT / "models/Qwen3-4B-Instruct-2507"
model_source = json.loads(
    (ROOT / "logs/model_revision.json").read_text()
)
assert model_source == pool_manifest["model_source"]

config = AutoConfig.from_pretrained(
    model_dir, local_files_only=True
)
assert config.num_hidden_layers == 36

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"

feature_names = [
    f"layer_{layer:02d}_{stat}"
    for layer in LAYERS
    for stat in ("rms", "abs_p99")
]

folder = ROOT / f"data/features/seed_{args.seed}"
records_dir = folder / "records"
records_dir.mkdir(parents=True, exist_ok=True)

metadata = {
    "definition_version": 1,
    "script_sha256": digest(Path(__file__).read_bytes()),
    "pool_sha256": pool_hash,
    "pool_seed": args.seed,
    "model_source": model_source,
    "layers": LAYERS,
    "feature_names": feature_names,
    "observed_tensor": "input to model.layers[i].mlp.down_proj",
    "statistics": {
        "rms": "sqrt(mean(x.float() ** 2)) over all elements",
        "abs_p99": "exact quantile(abs(x.float()), 0.99), linear interpolation",
    },
    "model_dtype": "bfloat16",
    "attention_backend": "eager",
    "batch_size": 1,
    "sequence_length": 512,
    "use_cache": False,
    "gpu": torch.cuda.get_device_name(0),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers")
    },
}
save_json(folder / "metadata.json", metadata)

def check_record(record, item):
    assert record["id"] == item["id"]
    assert record["domain"] == item["domain"]
    values = record["features"]
    assert len(values) == 16
    assert all(math.isfinite(v) and v >= 0 for v in values)
    assert math.isfinite(record["forward_statistics_seconds"])
    assert record["forward_statistics_seconds"] >= 0

pending = []
for item in pool:
    path = records_dir / f"{item['id']}.json"
    if path.exists():
        check_record(json.loads(path.read_text()), item)
    else:
        pending.append(item)

print(f"Already saved: {512 - len(pending)}/512", flush=True)

if pending:
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    print(f"Free GPU memory: {free_gib:.2f} GiB", flush=True)
    if free_gib < 16:
        raise RuntimeError("Stop the serving process; need 16 GiB free.")

    torch.manual_seed(42)
    print("Loading the original BF16 model...", flush=True)
    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    model.requires_grad_(False)

    torch.cuda.synchronize()
    print(
        f"Model load seconds: {time.perf_counter() - load_start:.1f}",
        flush=True,
    )

    backbone = model.model
    assert len(backbone.layers) == 36
    current = {}

    def make_hook(layer):
        def hook(module, inputs):
            activation = inputs[0]
            assert activation.dtype == torch.bfloat16
            assert tuple(activation.shape) == (
                1, 512, module.in_features
            )
            assert layer not in current

            flat = activation.detach().float().reshape(-1)
            if flat.numel() > 2**24:
                raise RuntimeError("Tensor exceeds exact quantile size limit.")

            rms = flat.square().mean().sqrt()
            p99 = torch.quantile(
                flat.abs(), 0.99, interpolation="linear"
            )
            current[layer] = torch.stack([rms, p99]).cpu().tolist()
            # Return None: observe the input without modifying it.
        return hook

    handles = []
    try:
        for layer in LAYERS:
            module = backbone.layers[layer].mlp.down_proj
            assert isinstance(module, torch.nn.Linear)
            handles.append(
                module.register_forward_pre_hook(make_hook(layer))
            )
            print(f"Observing layer {layer}: mlp.down_proj input")

        completed = 512 - len(pending)

        with torch.inference_mode():
            for item in pending:
                current.clear()
                torch.cuda.synchronize()
                start = time.perf_counter()

                ids = torch.tensor(
                    [item["input_ids"]],
                    dtype=torch.long,
                    device="cuda",
                )

                # Vocabulary logits are unnecessary for these features.
                output = backbone(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                del output, ids

                assert set(current) == set(LAYERS)
                values = [
                    value
                    for layer in LAYERS
                    for value in current[layer]
                ]
                record = {
                    "id": item["id"],
                    "domain": item["domain"],
                    "features": values,
                    "forward_statistics_seconds": elapsed,
                }
                check_record(record, item)
                save_json(records_dir / f"{item['id']}.json", record)

                completed += 1
                if completed == 1 or completed % 16 == 0:
                    print(
                        f"Saved {completed}/512; "
                        f"last sequence {elapsed:.2f}s",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    del backbone, model
    torch.cuda.empty_cache()

records = []
for item in pool:
    record = json.loads(
        (records_dir / f"{item['id']}.json").read_text()
    )
    check_record(record, item)
    records.append(record)

result = {
    "metadata_sha256": digest((folder / "metadata.json").read_bytes()),
    "feature_names": feature_names,
    "records": records,
}
save_json(folder / "features.json", result)

seconds = sum(r["forward_statistics_seconds"] for r in records)
print("\nFeature matrix shape: (512, 16)")
print("All features finite and nonnegative: True")
print(f"Sum of forward/statistics seconds: {seconds:.1f}")
print("Saved:", folder / "features.json")
print("PASS: BF16 activation features complete.")
