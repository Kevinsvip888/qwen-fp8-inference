import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import gsm8k_eval_protocol as protocol

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
DATA = ROOT / "data/evaluation/gsm8k_dev_v1"
METHODS = ("BF16", "Random32", "Selected32", "Random128")
BOOTSTRAP_SEED = 2026
BOOTSTRAP_SAMPLES = 20000

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

BASE = ROOT / f"outputs/evaluation/gsm8k_dev_v1/seed{args.seed}"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def save_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"Existing analysis differs: {path}")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)

evaluation_bytes = (DATA / "manifest.json").read_bytes()
evaluation = json.loads(evaluation_bytes)
request_bytes = (DATA / "requests.jsonl").read_bytes()
gold_bytes = (DATA / "gold.json").read_bytes()

assert digest(request_bytes) == evaluation["requests_sha256"]
assert digest(gold_bytes) == evaluation["gold_sha256"]
assert digest(Path(protocol.__file__).read_bytes()) == (
    evaluation["parser_sha256"]
)
assert evaluation["primary_comparison"] == "Selected32 versus Random32"

requests = [json.loads(line) for line in request_bytes.splitlines()]
ids = [item["id"] for item in requests]
request_lookup = {item["id"]: item for item in requests}
gold = json.loads(gold_bytes)
assert len(ids) == len(set(ids)) == len(gold) == 256

scores = {}
correctness = {}
run_hashes = {}
runs = {}
metrics = {}
input_hashes = {}

for method in METHODS:
    folder = BASE / method
    run_bytes = (folder / "run_manifest.json").read_bytes()
    run = json.loads(run_bytes)
    summary_bytes = (folder / "summary_full.json").read_bytes()
    summary = json.loads(summary_bytes)

    run_hashes[method] = digest(run_bytes)
    runs[method] = run
    assert run["method"] == method
    assert run["evaluation_manifest_sha256"] == digest(evaluation_bytes)
    assert summary["run_identity_sha256"] == run_hashes[method]
    assert summary["method"] == method
    assert summary["complete_256"]
    assert summary["questions_evaluated"] == 256

    rows = {item["id"]: item for item in summary["results"]}
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
    parse_failed = sum(rows[qid]["parse_failed"] for qid in ids)
    length = sum(
        rows[qid]["finish_reason"] == "length" for qid in ids
    )
    assert correct == summary["correct"]
    assert parse_failed == summary["parse_failures"]
    assert length == summary["length_finishes"]
    assert 256 - correct - parse_failed == (
        summary["incorrect_parsed_answers"]
    )

    scores[method] = rows
    correctness[method] = np.asarray(
        [rows[qid]["correct"] for qid in ids], dtype=np.int64
    )
    metrics[method] = {
        "correct": correct,
        "accuracy_percent": 100 * correct / 256,
        "parse_failures": parse_failed,
        "incorrect_parsed_answers": 256 - correct - parse_failed,
        "length_finishes": length,
    }
    input_hashes[method] = {
        "run_manifest_sha256": digest(run_bytes),
        "summary_sha256": digest(summary_bytes),
    }

assert len({runs[m]["vllm_version"] for m in METHODS}) == 1
assert len({
    runs[m]["checkpoint_identity"]["protocol_sha256"]
    for m in METHODS if m != "BF16"
}) == 1

def paired_comparison(first, second):
    a = correctness[first]
    b = correctness[second]
    gains = (a == 1) & (b == 0)
    losses = (a == 0) & (b == 1)

    return {
        "first": first,
        "second": second,
        "both_correct": int(np.sum((a == 1) & (b == 1))),
        "both_incorrect": int(np.sum((a == 0) & (b == 0))),
        "first_only_correct": int(gains.sum()),
        "second_only_correct": int(losses.sum()),
        "difference_percentage_points": float(100 * (a - b).mean()),
        "first_only_correct_ids": [
            qid for qid, flag in zip(ids, gains) if flag
        ],
        "second_only_correct_ids": [
            qid for qid, flag in zip(ids, losses) if flag
        ],
    }

pairs = [
    paired_comparison("Selected32", "Random32"),
    paired_comparison("Random32", "BF16"),
    paired_comparison("Selected32", "BF16"),
    paired_comparison("Random128", "BF16"),
    paired_comparison("Random128", "Random32"),
]
primary = pairs[0]

# Resampling paired differences is equivalent to resampling the same
# question indices for both models.
difference = correctness["Selected32"] - correctness["Random32"]
rng = np.random.default_rng(BOOTSTRAP_SEED)
draws = np.empty(BOOTSTRAP_SAMPLES)

for start in range(0, BOOTSTRAP_SAMPLES, 1000):
    count = min(1000, BOOTSTRAP_SAMPLES - start)
    indices = rng.integers(0, len(ids), size=(count, len(ids)))
    draws[start:start + count] = 100 * difference[indices].mean(axis=1)

interval = np.quantile(draws, [0.025, 0.975]).tolist()

def exact_mcnemar(gains, losses):
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        comb(discordant, k)
        for k in range(min(gains, losses) + 1)
    )
    return min(1.0, 2 * tail / (1 << discordant))

pvalue = exact_mcnemar(
    primary["first_only_correct"],
    primary["second_only_correct"],
)

def load_response(method, qid):
    path = BASE / method / "records" / f"{qid}.json"
    record = read_json(path)
    assert record["run_identity_sha256"] == run_hashes[method]
    assert record["id"] == qid
    assert record["request"] == {
        **request_lookup[qid]["request"],
        "model": runs[method]["served_model_name"],
    }
    choice = record["response"]["choices"][0]
    text = choice["message"]["content"]
    assert protocol.parse_final_answer(text) == (
        scores[method][qid]["prediction"]
    )
    assert choice["finish_reason"] == (
        scores[method][qid]["finish_reason"]
    )
    return text, str(path)

# Save complete responses for parser failures and truncated outputs.
flagged = []
for method in METHODS:
    for qid in ids:
        score = scores[method][qid]
        if score["parse_failed"] or score["finish_reason"] == "length":
            text, path = load_response(method, qid)
            flagged.append({
                "method": method,
                **score,
                "messages": request_lookup[qid]["request"]["messages"],
                "response_text": text,
                "raw_record": path,
            })

# Save both responses for every primary-comparison disagreement.
disagreements = []
changed_ids = set(
    primary["first_only_correct_ids"]
    + primary["second_only_correct_ids"]
)
for qid in ids:
    if qid not in changed_ids:
        continue
    models = {}
    for method in ("Selected32", "Random32"):
        text, path = load_response(method, qid)
        models[method] = {
            "score": scores[method][qid],
            "response_text": text,
            "raw_record": path,
        }
    disagreements.append({
        "id": qid,
        "messages": request_lookup[qid]["request"]["messages"],
        "reference": gold[qid],
        "models": models,
    })

lost_ids = primary["second_only_correct_ids"]
loss_diagnostics = {
    "Selected32_parse_failures": sum(
        scores["Selected32"][qid]["parse_failed"] for qid in lost_ids
    ),
    "Selected32_parsed_wrong": sum(
        not scores["Selected32"][qid]["parse_failed"]
        for qid in lost_ids
    ),
    "Selected32_length_finishes_overlapping_these_losses": sum(
        scores["Selected32"][qid]["finish_reason"] == "length"
        for qid in lost_ids
    ),
}

report = {
    "pool_seed": args.seed,
    "evaluation_manifest_sha256": digest(evaluation_bytes),
    "input_hashes": input_hashes,
    "numpy_version": np.__version__,
    "metrics": metrics,
    "paired_comparisons": pairs,
    "primary_inference": {
        "comparison": "Selected32 minus Random32",
        "bootstrap": {
            "method": "paired percentile bootstrap over questions",
            "resamples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "approximate_95_percent_interval_pp": interval,
        },
        "exact_two_sided_mcnemar_p": pvalue,
        "scope": "fixed checkpoints; excludes calibration-pool variability",
    },
    "primary_loss_diagnostics": loss_diagnostics,
}

save_json(BASE / "paired_analysis.json", report)
save_json(
    BASE / "failure_review.json",
    {
        "flagged_responses": flagged,
        "primary_disagreements": disagreements,
    },
)

print("\nPaired comparisons: first minus second")
print(f"{'Comparison':<25} {'Gains':>6} {'Losses':>7} {'Delta pp':>10}")
for pair in pairs:
    label = f"{pair['first']} - {pair['second']}"
    print(
        f"{label:<25} {pair['first_only_correct']:>6} "
        f"{pair['second_only_correct']:>7} "
        f"{pair['difference_percentage_points']:>10.4f}"
    )

print("\nPrimary comparison: Selected32 - Random32")
print("Both correct:", primary["both_correct"])
print("Both incorrect:", primary["both_incorrect"])
print(f"Approximate paired bootstrap 95% interval: "
      f"[{interval[0]:.4f}, {interval[1]:.4f}] pp")
print(f"Exact two-sided McNemar p-value: {pvalue:.6f}")
print("Loss diagnostics:", loss_diagnostics)

print("\nFlagged response endings:")
for item in flagged:
    text = item["response_text"].strip()
    last_line = text.splitlines()[-1] if text else "<empty>"
    print(
        f"{item['method']} {item['id']} "
        f"parse_failed={item['parse_failed']} "
        f"finish={item['finish_reason']} "
        f"last_line={last_line[-180:]!r}"
    )

print("\nSaved:", BASE / "paired_analysis.json")
print("Saved:", BASE / "failure_review.json")
print("PASS: paired comparison and failure review complete.")
