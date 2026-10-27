import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
METHODS = ("Random32", "Selected32", "Random128")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

def read_json(path):
    return json.loads(path.read_text())

def digest(data):
    return hashlib.sha256(data).hexdigest()

protocol_bytes = (
    ROOT / "data/calibration/fp8_protocol.json"
).read_bytes()
protocol = json.loads(protocol_bytes)
protocol_hash = digest(protocol_bytes)
assert protocol["recipe"]["algorithm"] == "max"

pool_dir = ROOT / f"data/calibration/seed_{args.seed}"
pool_hash = digest((pool_dir / "pool.jsonl").read_bytes())

manifests = {}
for method in METHODS:
    folder = ROOT / (
        f"models/Qwen3-4B-Instruct-2507-FP8-{method}-seed{args.seed}"
    )
    manifest = read_json(folder / "calibration_manifest.json")
    identity = manifest["identity"]

    assert identity["method"] == method
    assert identity["pool_seed"] == args.seed
    assert identity["pool_sha256"] == pool_hash
    assert identity["protocol_sha256"] == protocol_hash
    assert manifest["protocol"] == protocol

    expected = 128 if method == "Random128" else 32
    assert identity["calibration_sequences"] == expected
    assert identity["calibration_tokens"] == expected * 512
    assert len(set(identity["sequence_ids"])) == expected
    assert manifest["tensor_counts_by_dtype"]["F8_E4M3"] == 252

    config_path = folder / "hf_quant_config.json"
    config_record = next(
        entry for entry in manifest["export_files"]
        if entry["path"] == "hf_quant_config.json"
    )
    assert digest(config_path.read_bytes()) == config_record["sha256"]
    config = read_json(config_path)["quantization"]
    assert config["quant_algo"] == "FP8"
    assert config.get("kv_cache_quant_algo") is None

    manifests[method] = manifest

small_ids = manifests["Random32"]["identity"]["sequence_ids"]
large_ids = manifests["Random128"]["identity"]["sequence_ids"]
assert small_ids == large_ids[:32]

names = sorted(manifests["Random32"]["activation_amax"])
assert len(names) == 252

ranges = {}
for method in METHODS:
    values = manifests[method]["activation_amax"]
    assert set(values) == set(names)
    array = np.asarray([values[name] for name in names], dtype=np.float64)
    assert np.isfinite(array).all()
    assert (array > 0).all()
    ranges[method] = array

reference = ranges["Random32"]
tolerance = 1e-6 + 1e-5 * np.abs(reference)
comparisons = {}

print("Reference: Random32 activation amax")
print("Ratios above 1 mean larger calibrated ranges.\n")
print(
    f"{'Method':<12} {'Larger':>7} {'Equal*':>7} {'Smaller':>8} "
    f"{'Min ratio':>10} {'Median':>10} {'Max ratio':>10}"
)

for method in ("Selected32", "Random128"):
    difference = ranges[method] - reference
    ratios = ranges[method] / reference

    larger = int(np.sum(difference > tolerance))
    smaller = int(np.sum(difference < -tolerance))
    equal = len(names) - larger - smaller

    summary = {
        "larger": larger,
        "equal_within_tolerance": equal,
        "smaller": smaller,
        "minimum_ratio": float(ratios.min()),
        "median_ratio": float(np.median(ratios)),
        "maximum_ratio": float(ratios.max()),
    }
    comparisons[method] = summary

    print(
        f"{method:<12} {larger:>7} {equal:>7} {smaller:>8} "
        f"{ratios.min():>10.4f} {np.median(ratios):>10.4f} "
        f"{ratios.max():>10.4f}"
    )

violations = [
    {
        "name": names[i],
        "Random32": float(reference[i]),
        "Random128": float(ranges["Random128"][i]),
    }
    for i in np.flatnonzero(
        ranges["Random128"] < reference - tolerance
    )
]

report = {
    "pool_seed": args.seed,
    "pool_sha256": pool_hash,
    "protocol_sha256": protocol_hash,
    "quantizers_compared": len(names),
    "random_subsets_nested": True,
    "comparison_tolerance": {"absolute": 1e-6, "relative": 1e-5},
    "comparisons_against_Random32": comparisons,
    "nested_max_violations": violations,
    "per_quantizer": [
        {
            "name": name,
            **{
                method: float(ranges[method][i])
                for method in METHODS
            },
        }
        for i, name in enumerate(names)
    ],
}

output = ROOT / f"outputs/calibration_ranges_seed{args.seed}.json"
output.parent.mkdir(parents=True, exist_ok=True)
content = (json.dumps(report, indent=2) + "\n").encode()

if output.exists() and output.read_bytes() != content:
    raise RuntimeError(f"Existing report differs: {output}")

temporary = output.with_suffix(".tmp")
temporary.write_bytes(content)
temporary.replace(output)

print("\n*Equal uses the recorded numerical tolerance.")
print("Shared protocol SHA256:", protocol_hash)
print("Nested-max violations:", len(violations))
print("Saved:", output)

if violations:
    for item in violations[:5]:
        print(item)
    print("CHECK REQUIRED: investigate smaller Random128 maxima.")
    raise SystemExit(1)

print("PASS: shared protocol, valid ranges, and nested-max consistency.")
