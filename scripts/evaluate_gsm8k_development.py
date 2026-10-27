import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

import gsm8k_eval_protocol as protocol

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
DATA = ROOT / "data/evaluation/gsm8k_dev_v1"
BASE_URL = "http://127.0.0.1:8000"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", required=True,
    choices=["BF16", "Random32", "Selected32", "Random128"],
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--limit", type=int, default=256)
args = parser.parse_args()
assert 1 <= args.limit <= 256

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text())

def encode_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()

def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encode_json(value))
    temporary.replace(path)

manifest_bytes = (DATA / "manifest.json").read_bytes()
evaluation = json.loads(manifest_bytes)
request_bytes = (DATA / "requests.jsonl").read_bytes()
gold_bytes = (DATA / "gold.json").read_bytes()

assert digest(request_bytes) == evaluation["requests_sha256"]
assert digest(gold_bytes) == evaluation["gold_sha256"]
assert digest(Path(protocol.__file__).read_bytes()) == (
    evaluation["parser_sha256"]
)
assert evaluation["generation"] == protocol.GENERATION
assert evaluation["request_concurrency"] == 1
protocol.self_check()

items = [json.loads(line) for line in request_bytes.splitlines()]
gold = json.loads(gold_bytes)
assert len(items) == len(gold) == 256
assert {item["id"] for item in items} == set(gold)
targets = items[:args.limit]

source = read_json(ROOT / "logs/model_revision.json")
assert source == evaluation["model_source"]
alias = f"qwen-{args.method.lower()}"

if args.method == "BF16":
    checkpoint = ROOT / "models/Qwen3-4B-Instruct-2507"
    checkpoint_identity = {"precision": "BF16", "source": source}
else:
    checkpoint = ROOT / (
        f"models/Qwen3-4B-Instruct-2507-FP8-{args.method}-seed{args.seed}"
    )
    exported = read_json(checkpoint / "calibration_manifest.json")
    checkpoint_identity = exported["identity"]
    assert checkpoint_identity["method"] == args.method
    assert checkpoint_identity["pool_seed"] == args.seed
    assert exported["protocol"]["source"] == source

    calibration_pool = read_json(
        ROOT / f"data/calibration/seed_{args.seed}/manifest.json"
    )
    assert calibration_pool["math_split_sha256"] == (
        evaluation["split_sha256"]
    )
    assert checkpoint_identity["pool_sha256"] == (
        calibration_pool["pool_sha256"]
    )

assert checkpoint.is_dir()

folder = ROOT / (
    f"outputs/evaluation/gsm8k_dev_v1/seed{args.seed}/{args.method}"
)
records_dir = folder / "records"
records_dir.mkdir(parents=True, exist_ok=True)

# The requested limit is intentionally excluded: pilot and full run
# share the same identity and can reuse responses.
identity = {
    "collector_version": 1,
    "method": args.method,
    "pool_seed": args.seed,
    "checkpoint": str(checkpoint),
    "checkpoint_identity": checkpoint_identity,
    "served_model_name": alias,
    "evaluation_manifest_sha256": digest(manifest_bytes),
    "vllm_version": importlib.metadata.version("vllm"),
    "request_concurrency": 1,
    "expected_questions": 256,
}
identity_bytes = encode_json(identity)
identity_hash = digest(identity_bytes)
identity_path = folder / "run_manifest.json"

if identity_path.exists():
    assert identity_path.read_bytes() == identity_bytes, (
        "Existing run uses different inputs or environment."
    )
else:
    save_json(identity_path, identity)

opener = build_opener(ProxyHandler({}))

def api_json(path, payload=None):
    request = Request(
        BASE_URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"HTTP {error.code}: {body[:2000]}"
        ) from error

def expected_request(item):
    return {**item["request"], "model": alias}

def validate_record(record, item):
    assert record["run_identity_sha256"] == identity_hash
    assert record["id"] == item["id"]
    assert record["request"] == expected_request(item)

    response = record["response"]
    assert response["model"] == alias
    assert len(response["choices"]) == 1
    choice = response["choices"][0]
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    assert response["usage"]["prompt_tokens"] == (
        item["expected_prompt_tokens"]
    ), (
        item["id"],
        response["usage"]["prompt_tokens"],
        item["expected_prompt_tokens"],
    )

cached = sum(
    (records_dir / f"{item['id']}.json").exists()
    for item in targets
)
print(f"Method: {args.method}", flush=True)
print(f"Requested: {len(targets)}/256; cached: {cached}", flush=True)

server_checked = False
scored = []

for number, item in enumerate(targets, start=1):
    path = records_dir / f"{item['id']}.json"

    if path.exists():
        record = read_json(path)
        origin = "cached"
    else:
        if not server_checked:
            with opener.open(BASE_URL + "/health", timeout=10) as response:
                assert response.status == 200
            available = {
                card["id"] for card in api_json("/v1/models")["data"]
            }
            assert alias in available, (
                f"Expected {alias}; server exposes {sorted(available)}"
            )
            server_checked = True

        request = expected_request(item)
        start = time.perf_counter()
        response = api_json("/v1/chat/completions", request)
        record = {
            "run_identity_sha256": identity_hash,
            "id": item["id"],
            "request": request,
            "response": response,
            "request_wall_seconds": time.perf_counter() - start,
        }
        # Preserve the response even if subsequent validation fails.
        save_json(path, record)
        origin = "new"

    validate_record(record, item)

    response = record["response"]
    choice = response["choices"][0]
    content = choice["message"]["content"]
    prediction = protocol.parse_final_answer(content)

    result = {
        "id": item["id"],
        "prediction": prediction,
        "reference": gold[item["id"]],
        "correct": prediction == gold[item["id"]],
        "parse_failed": prediction is None,
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
    }
    scored.append(result)

    if args.limit <= 8 or number % 16 == 0 or number == len(targets):
        print(
            f"{number}/{len(targets)} {item['id']} [{origin}] "
            f"pred={prediction} gold={result['reference']} "
            f"correct={result['correct']} "
            f"finish={result['finish_reason']}",
            flush=True,
        )
        if args.limit <= 8 and prediction is None:
            print("Unparsed response:\n" + content, flush=True)

correct = sum(result["correct"] for result in scored)
parse_failures = sum(result["parse_failed"] for result in scored)
truncated = sum(
    result["finish_reason"] == "length" for result in scored
)

summary = {
    "run_identity_sha256": identity_hash,
    "method": args.method,
    "questions_evaluated": len(scored),
    "complete_256": len(scored) == 256,
    "correct": correct,
    "numeric_exact_match_percent": 100 * correct / len(scored),
    "parse_failures": parse_failures,
    "incorrect_parsed_answers": len(scored) - correct - parse_failures,
    "length_finishes": truncated,
    "results": scored,
}

summary_name = (
    "summary_full.json"
    if args.limit == 256
    else f"summary_first{args.limit}.json"
)
summary_path = folder / summary_name
save_json(summary_path, summary)

print("\nQuestions evaluated:", len(scored))
print("Correct numerical answers:", correct)
print("Parse failures:", parse_failures)
print("Incorrect parsed answers:", summary["incorrect_parsed_answers"])
print("Length finishes:", truncated)
print(f"Numeric exact match: {summary['numeric_exact_match_percent']:.2f}%")
print("Full 256-question evaluation complete:", summary["complete_256"])
print("Saved:", summary_path)
print("PASS: saved responses and prompt token counts verified.")
