import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

pool_dir = ROOT / f"data/calibration/seed_{args.seed}"
feature_dir = ROOT / f"data/features/seed_{args.seed}"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def save(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing file differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

def save_json(path, value):
    save(path, (json.dumps(value, indent=2) + "\n").encode())

# Bind selection to the audited pool and saved feature definition.
pool_bytes = (pool_dir / "pool.jsonl").read_bytes()
pool_hash = digest(pool_bytes)
pool = [json.loads(line) for line in pool_bytes.splitlines()]
lookup = {block["id"]: block for block in pool}

assert len(pool) == len(lookup) == 512
assert all(len(block["input_ids"]) == 512 for block in pool)
assert read_json(pool_dir / "pool_audit.json")["pool_sha256"] == pool_hash

metadata_bytes = (feature_dir / "metadata.json").read_bytes()
metadata = json.loads(metadata_bytes)
feature_bytes = (feature_dir / "features.json").read_bytes()
features = json.loads(feature_bytes)

assert metadata["pool_sha256"] == pool_hash
assert metadata["pool_seed"] == args.seed
assert features["metadata_sha256"] == digest(metadata_bytes)
assert features["feature_names"] == metadata["feature_names"]
assert len(features["feature_names"]) == 16

records = features["records"]
feature_lookup = {record["id"]: record for record in records}
assert len(records) == len(feature_lookup) == 512
assert set(feature_lookup) == set(lookup)

for sid, record in feature_lookup.items():
    assert record["domain"] == lookup[sid]["domain"]
    assert record == read_json(feature_dir / "records" / f"{sid}.json")
    values = np.asarray(record["features"], dtype=np.float64)
    assert values.shape == (16,)
    assert np.isfinite(values).all()
    assert (values >= 0).all()

# Validate the existing random subsets.
random_manifest = read_json(pool_dir / "random_subsets_manifest.json")
assert random_manifest["pool_sha256"] == pool_hash
methods = {}

for name, expected_size in (("Random32", 32), ("Random128", 128)):
    entry = random_manifest["subsets"][name]
    content = (pool_dir / entry["file"]).read_bytes()
    assert digest(content) == entry["sha256"]

    blocks = [json.loads(line) for line in content.splitlines()]
    ids = [block["id"] for block in blocks]
    assert ids == entry["sequence_ids"]
    assert len(ids) == len(set(ids)) == expected_size
    assert Counter(block["domain"] for block in blocks) == {
        "general": expected_size // 2,
        "math": expected_size // 2,
    }
    assert all(block == lookup[block["id"]] for block in blocks)
    methods[name] = ids

selection_start = time.perf_counter()
selected = {}
normalization = {}
traces = {}
geometry = {}

for domain in ("general", "math"):
    # Sorted IDs make exact-distance ties resolve consistently.
    ids = sorted(
        sid for sid, block in lookup.items()
        if block["domain"] == domain
    )
    assert len(ids) == 256

    x = np.asarray(
        [feature_lookup[sid]["features"] for sid in ids],
        dtype=np.float64,
    )

    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)

    # Ignore constant or numerically negligible feature dimensions.
    threshold = 1e-12 * np.maximum(1.0, np.abs(mean))
    active = std > threshold
    assert active.any(), f"No varying features in {domain}"

    scale = np.where(active, std, 1.0)
    z = (x - mean) / scale
    z[:, ~active] = 0.0
    assert np.isfinite(z).all()

    # Pairwise squared Euclidean distances: 256 x 256.
    distance2 = np.sum(
        (z[:, None, :] - z[None, :, :]) ** 2,
        axis=2,
    )

    # Start nearest the centroid, then use farthest-point selection.
    centroid = z.mean(axis=0)
    first = int(np.argmin(np.sum((z - centroid) ** 2, axis=1)))
    chosen = [first]
    nearest2 = distance2[:, first].copy()
    trace = []

    while True:
        trace.append({
            "step": len(chosen),
            "added_id": ids[chosen[-1]],
            "maximum_nearest_distance": float(
                np.sqrt(nearest2.max())
            ),
        })
        if len(chosen) == 16:
            break

        scores = nearest2.copy()
        scores[chosen] = -np.inf
        next_index = int(np.argmax(scores))
        chosen.append(next_index)
        nearest2 = np.minimum(nearest2, distance2[:, next_index])

    assert len(set(chosen)) == 16
    selected[domain] = [ids[i] for i in chosen]
    normalization[domain] = {
        "mean": mean.tolist(),
        "population_std": std.tolist(),
        "active_dimensions": active.tolist(),
        "constant_threshold": threshold.tolist(),
    }
    traces[domain] = trace
    geometry[domain] = (ids, distance2)

    print(f"{domain}: {int(active.sum())}/16 active features")

# Alternate domains, matching the ordering convention of random subsets.
selected_ids = [
    sid
    for pair in zip(selected["general"], selected["math"])
    for sid in pair
]
assert len(selected_ids) == len(set(selected_ids)) == 32

blocks = [lookup[sid] for sid in selected_ids]
assert Counter(block["domain"] for block in blocks) == {
    "general": 16,
    "math": 16,
}
assert sum(len(block["input_ids"]) for block in blocks) == 16384
methods["Selected32"] = selected_ids

selection_seconds = time.perf_counter() - selection_start

# Measure coverage in the same standardized geometry for every method.
coverage = []
for domain in ("general", "math"):
    ids, distance2 = geometry[domain]
    position = {sid: i for i, sid in enumerate(ids)}

    for method in ("Random32", "Selected32", "Random128"):
        indices = [
            position[sid] for sid in methods[method]
            if lookup[sid]["domain"] == domain
        ]
        nearest = np.sqrt(distance2[:, indices].min(axis=1))
        coverage.append({
            "domain": domain,
            "method": method,
            "selected_sequences": len(indices),
            "mean_nearest_distance": float(nearest.mean()),
            "p95_nearest_distance": float(np.quantile(nearest, 0.95)),
            "maximum_nearest_distance": float(nearest.max()),
        })

content = "".join(
    json.dumps(block, separators=(",", ":")) + "\n"
    for block in blocks
).encode()

selection_manifest = {
    "method": "Selected32",
    "pool_sha256": pool_hash,
    "features_sha256": digest(feature_bytes),
    "feature_metadata_sha256": digest(metadata_bytes),
    "numpy_version": np.__version__,
    "algorithm": {
        "normalization": "per-domain feature standardization, ddof=0",
        "distance": "Euclidean distance in standardized feature space",
        "initialization": "candidate nearest the domain centroid",
        "iteration": "maximize distance to nearest selected candidate",
        "tie_break": "first candidate in lexicographically sorted IDs",
        "sequences_per_domain": 16,
    },
    "normalization": normalization,
    "selection_trace": traces,
    "sequence_ids": selected_ids,
    "sequences": 32,
    "tokens": 16384,
    "file": "selected32.jsonl",
    "sha256": digest(content),
}

save(pool_dir / "selected32.jsonl", content)
save_json(pool_dir / "selected32_manifest.json", selection_manifest)
save_json(
    pool_dir / "feature_coverage.json",
    {
        "pool_sha256": pool_hash,
        "features_sha256": digest(feature_bytes),
        "results": coverage,
    },
)

print("\nFeature coverage: lower distances mean better coverage.")
print(f"{'Domain':<9} {'Method':<12} {'N':>3} "
      f"{'Mean':>9} {'P95':>9} {'Max':>9}")

for row in coverage:
    print(
        f"{row['domain']:<9} {row['method']:<12} "
        f"{row['selected_sequences']:>3} "
        f"{row['mean_nearest_distance']:>9.4f} "
        f"{row['p95_nearest_distance']:>9.4f} "
        f"{row['maximum_nearest_distance']:>9.4f}"
    )

print(f"\nCPU selection seconds: {selection_seconds:.3f}")
print("Selected32: sequences=32, tokens=16384, general=16, math=16")
print("Selected32 SHA256:", digest(content))
print("PASS: Selected32 and feature-coverage report saved.")
