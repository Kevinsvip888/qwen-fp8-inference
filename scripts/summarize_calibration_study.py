import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import gsm8k_eval_protocol as protocol

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
DATA = ROOT / "data/evaluation/gsm8k_dev_v1"
BASE = ROOT / "outputs/evaluation/gsm8k_dev_v1"
SEEDS = (42, 43, 44)
METHODS = ("Random32", "Selected32", "Random128")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text())

evaluation_bytes = (DATA / "manifest.json").read_bytes()
evaluation = json.loads(evaluation_bytes)
request_bytes = (DATA / "requests.jsonl").read_bytes()
gold_bytes = (DATA / "gold.json").read_bytes()

assert digest(request_bytes) == evaluation["requests_sha256"]
assert digest(gold_bytes) == evaluation["gold_sha256"]
assert digest(Path(protocol.__file__).read_bytes()) == (
    evaluation["parser_sha256"]
)

requests = [json.loads(line) for line in request_bytes.splitlines()]
ids = [item["id"] for item in requests]
request_lookup = {item["id"]: item for item in requests}
gold = json.loads(gold_bytes)
assert len(ids) == len(set(ids)) == len(gold) == 256

shared_protocol_hash = digest(
    (ROOT / "data/calibration/fp8_protocol.json").read_bytes()
)
loaded = {}
input_hashes = {}
versions = set()

def load_run(seed, method):
    folder = BASE / f"seed{seed}" / method
    run_bytes = (folder / "run_manifest.json").read_bytes()
    summary_bytes = (folder / "summary_full.json").read_bytes()
    run = json.loads(run_bytes)
    summary = json.loads(summary_bytes)

    assert run["method"] == method
    assert run["pool_seed"] == seed
    assert run["evaluation_manifest_sha256"] == digest(evaluation_bytes)
    assert summary["run_identity_sha256"] == digest(run_bytes)
    assert summary["method"] == method
    assert summary["complete_256"]
    assert summary["questions_evaluated"] == 256
    versions.add(run["vllm_version"])

    rows = {row["id"]: row for row in summary["results"]}
    assert len(summary["results"]) == len(rows) == 256
    assert set(rows) == set(ids)

    for qid in ids:
        row = rows[qid]
        assert row["reference"] == gold[qid]
        assert row["correct"] == (row["prediction"] == gold[qid])
        assert row["parse_failed"] == (row["prediction"] is None)
        assert row["prompt_tokens"] == (
            request_lookup[qid]["expected_prompt_tokens"]
        )
        assert row["finish_reason"] in ("stop", "length")

    correct = sum(rows[qid]["correct"] for qid in ids)
    assert correct == summary["correct"]
    assert sum(rows[qid]["parse_failed"] for qid in ids) == (
        summary["parse_failures"]
    )
    assert sum(
        rows[qid]["finish_reason"] == "length" for qid in ids
    ) == summary["length_finishes"]

    export = None
    if method == "BF16":
        assert run["checkpoint_identity"]["source"] == (
            evaluation["model_source"]
        )
    else:
        export = read_json(
            ROOT / (
                f"models/Qwen3-4B-Instruct-2507-FP8-"
                f"{method}-seed{seed}/calibration_manifest.json"
            )
        )
        assert export["identity"] == run["checkpoint_identity"]
        assert export["identity"]["protocol_sha256"] == shared_protocol_hash
        assert export["protocol"]["source"] == evaluation["model_source"]

    key = f"seed{seed}/{method}"
    input_hashes[key] = {
        "run_manifest_sha256": digest(run_bytes),
        "summary_sha256": digest(summary_bytes),
    }
    loaded[(seed, method)] = {
        "folder": folder,
        "run_hash": digest(run_bytes),
        "summary": summary,
        "rows": rows,
        "export": export,
    }
    return np.asarray(
        [rows[qid]["correct"] for qid in ids], dtype=np.float64
    )

bf16 = load_run(42, "BF16")
matrices = {
    method: np.stack([load_run(seed, method) for seed in SEEDS])
    for method in METHODS
}
assert len(versions) == 1

def exact_mcnemar(gains, losses):
    n = gains + losses
    if not n:
        return 1.0
    return min(
        1.0,
        2 * sum(comb(n, k) for k in range(min(gains, losses) + 1))
        / (1 << n),
    )

paired = []
for index, seed in enumerate(SEEDS):
    a = matrices["Selected32"][index]
    b = matrices["Random32"][index]
    gains = int(np.sum((a == 1) & (b == 0)))
    losses = int(np.sum((a == 0) & (b == 1)))
    paired.append({
        "pool_seed": seed,
        "selected_only_correct": gains,
        "random_only_correct": losses,
        "difference_pp": float(100 * (a - b).mean()),
        "descriptive_exact_mcnemar_p": exact_mcnemar(gains, losses),
    })

# Average across the three observed pools within each question FIRST.
per_question_difference = (
    matrices["Selected32"] - matrices["Random32"]
).mean(axis=0)
mean_difference = float(100 * per_question_difference.mean())

rng = np.random.default_rng(2026)
draws = np.empty(20000)
for start in range(0, 20000, 1000):
    indices = rng.integers(0, 256, size=(1000, 256))
    draws[start:start + 1000] = (
        100 * per_question_difference[indices].mean(axis=1)
    )
interval = np.quantile(draws, [0.025, 0.975]).tolist()

method_metrics = {}
for method in METHODS:
    accuracies = 100 * matrices[method].mean(axis=1)
    exports = [loaded[(seed, method)]["export"] for seed in SEEDS]
    method_metrics[method] = {
        "correct_by_seed": [
            int(matrices[method][i].sum()) for i in range(3)
        ],
        "accuracy_percent_by_seed": accuracies.tolist(),
        "mean_accuracy_percent": float(accuracies.mean()),
        "descriptive_seed_sd_pp": float(accuracies.std(ddof=1)),
        "parse_failures_by_seed": [
            loaded[(seed, method)]["summary"]["parse_failures"]
            for seed in SEEDS
        ],
        "length_finishes_by_seed": [
            loaded[(seed, method)]["summary"]["length_finishes"]
            for seed in SEEDS
        ],
        "mean_modelopt_quantize_seconds": float(np.mean([
            export["calibration_seconds"] for export in exports
        ])),
        "mean_safetensors_GiB": float(np.mean([
            export["safetensors_bytes"] / 1024**3
            for export in exports
        ])),
    }

feature_seconds = []
selection_algorithms = []
feature_definitions = []

for seed in SEEDS:
    folder = ROOT / f"data/features/seed_{seed}"
    feature_bytes = (folder / "features.json").read_bytes()
    features = json.loads(feature_bytes)
    metadata_bytes = (folder / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    selected = read_json(
        ROOT / f"data/calibration/seed_{seed}/selected32_manifest.json"
    )

    assert digest(feature_bytes) == selected["features_sha256"]
    assert features["metadata_sha256"] == digest(metadata_bytes)
    assert loaded[(seed, "Selected32")]["export"]["identity"][
        "subset_sha256"
    ] == selected["sha256"]

    selection_algorithms.append(selected["algorithm"])
    feature_definitions.append({
        key: metadata[key]
        for key in (
            "script_sha256", "layers", "feature_names", "statistics",
            "model_dtype", "attention_backend", "batch_size",
            "sequence_length",
        )
    })
    feature_seconds.append(sum(
        record["forward_statistics_seconds"]
        for record in features["records"]
    ))

assert all(x == selection_algorithms[0] for x in selection_algorithms)
assert all(x == feature_definitions[0] for x in feature_definitions)

# Post-hoc diagnostic only: do stopped, unparsed responses end with
# a bare number that matches the reference? Primary scores stay fixed.
bare_number_diagnostics = []
for (seed, method), data in loaded.items():
    for qid in ids:
        score = data["rows"][qid]
        if not score["parse_failed"] or score["finish_reason"] != "stop":
            continue

        raw = read_json(data["folder"] / "records" / f"{qid}.json")
        assert raw["run_identity_sha256"] == data["run_hash"]
        assert raw["id"] == qid
        text = raw["response"]["choices"][0]["message"]["content"]
        assert protocol.parse_final_answer(text) is None

        lines = text.strip().splitlines()
        last_line = lines[-1].strip() if lines else ""
        try:
            candidate = protocol.canonical_number(
                last_line.replace("**", "").replace("`", "")
            )
        except ValueError:
            candidate = None

        bare_number_diagnostics.append({
            "pool_seed": seed,
            "method": method,
            "id": qid,
            "last_line": last_line,
            "diagnostic_number": candidate,
            "reference": gold[qid],
            "matches_reference": candidate == gold[qid],
        })

report = {
    "evaluation_manifest_sha256": digest(evaluation_bytes),
    "input_hashes": input_hashes,
    "pool_seeds": list(SEEDS),
    "distinct_questions": 256,
    "numpy_version": np.__version__,
    "bf16_reference": {
        "pool_seed_folder": 42,
        "correct": int(bf16.sum()),
        "accuracy_percent": float(100 * bf16.mean()),
    },
    "method_metrics": method_metrics,
    "per_seed_primary_comparisons": paired,
    "primary_mean_difference_pp": mean_difference,
    "approximate_question_bootstrap_95_interval_pp": interval,
    "bootstrap_definition": {
        "resamples": 20000,
        "seed": 2026,
        "unit": "question, retaining all three pools together",
        "scope": "conditional on the three observed calibration pools",
    },
    "feature_forward_statistics_seconds_by_seed": feature_seconds,
    "mean_feature_forward_statistics_seconds": float(np.mean(feature_seconds)),
    "posthoc_bare_number_diagnostics": bare_number_diagnostics,
    "primary_scores_modified": False,
}

output = BASE / "three_seed_summary.json"
content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
if output.exists() and output.read_bytes() != content:
    raise RuntimeError(f"Existing report differs: {output}")
temporary = output.with_suffix(".tmp")
temporary.write_bytes(content)
temporary.replace(output)

print("\nPer-seed paired comparison: Selected32 - Random32")
print(f"{'Seed':<6} {'Gains':>6} {'Losses':>7} {'Delta pp':>10} {'Exact p':>10}")
for row in paired:
    print(
        f"{row['pool_seed']:<6} {row['selected_only_correct']:>6} "
        f"{row['random_only_correct']:>7} "
        f"{row['difference_pp']:>10.4f} "
        f"{row['descriptive_exact_mcnemar_p']:>10.6f}"
    )

print("\nAcross-pool descriptive summary")
print(f"{'Method':<12} {'Mean EM %':>10} {'Seed SD pp':>11}")
for method in METHODS:
    row = method_metrics[method]
    print(
        f"{method:<12} {row['mean_accuracy_percent']:>10.4f} "
        f"{row['descriptive_seed_sd_pp']:>11.4f}"
    )

print(f"\nMean Selected32 - Random32: {mean_difference:.4f} pp")
print(f"Approximate question-bootstrap 95% interval: "
      f"[{interval[0]:.4f}, {interval[1]:.4f}] pp")
print("Interval is conditional on the three observed calibration pools.")
print(f"Mean feature forward/statistics seconds: "
      f"{np.mean(feature_seconds):.2f}")

print("\nPost-hoc bare-number diagnostics; primary scores unchanged:")
for row in bare_number_diagnostics:
    print(
        f"seed={row['pool_seed']} {row['method']} {row['id']} "
        f"number={row['diagnostic_number']} gold={row['reference']} "
        f"match={row['matches_reference']}"
    )

print("\nSaved:", output)
print("PASS: three-pool paired summary complete.")
