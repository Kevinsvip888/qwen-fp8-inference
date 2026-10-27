import hashlib
import importlib.metadata
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

import gsm8k_eval_protocol as protocol

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
OUTPUT = ROOT / "data/evaluation/gsm8k_dev_v1"
CONTEXT_LIMIT = 4096

def digest(data):
    return hashlib.sha256(data).hexdigest()

def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()

def freeze(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing evaluation file differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

protocol.self_check()

split_path = ROOT / "data/splits/gsm8k_train_split.json"
split_bytes = split_path.read_bytes()
split = json.loads(split_bytes)

pool_manifest = json.loads(
    (ROOT / "data/calibration/seed_42/manifest.json").read_text()
)
assert digest(split_bytes) == pool_manifest["math_split_sha256"]

source = split["source"]
assert source["repo_id"] == "openai/gsm8k"
assert source["split"] == "train"

parts = split["partitions"]
row_sets = {
    name: {item["row"] for item in entries}
    for name, entries in parts.items()
}
assert len(parts["development"]) == len(row_sets["development"]) == 256
assert len(parts["fewshot_reserve"]) == 16
assert row_sets["development"].isdisjoint(row_sets["calibration"])
assert row_sets["development"].isdisjoint(row_sets["fewshot_reserve"])
assert row_sets["calibration"].isdisjoint(row_sets["fewshot_reserve"])

files = []
for entry in source["files"]:
    path = ROOT / entry["path"]
    assert digest(path.read_bytes()) == entry["sha256"]
    files.append(str(path))

dataset = load_dataset(
    "parquet",
    data_files={"train": files},
    split="train",
)
assert len(dataset) == source["rows"]

tokenizer_dir = ROOT / "models/Qwen3-4B-Instruct-2507"
for name, expected in pool_manifest["tokenizer_files_sha256"].items():
    assert digest((tokenizer_dir / name).read_bytes()) == expected

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir, local_files_only=True
)

requests = []
gold = {}

for item in parts["development"]:
    row = item["row"]
    example = dataset[row]
    question = example["question"].strip()

    question_hash = digest(" ".join(question.split()).encode())
    assert question_hash == item["question_sha256"]

    reference = example["answer"]
    assert reference.count("####") == 1, row
    raw_answer = reference.rsplit("####", 1)[1].strip()
    try:
        answer = protocol.canonical_number(raw_answer)
    except ValueError as error:
        raise ValueError(
            f"Cannot parse reference at row {row}: {raw_answer!r}"
        ) from error

    request = protocol.make_request(question)
    rendered = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    assert (
        len(input_ids) + protocol.GENERATION["max_tokens"]
        <= CONTEXT_LIMIT
    ), (row, len(input_ids))

    example_id = f"gsm8k-train-{row:05d}"
    requests.append({
        "id": example_id,
        "source_row": row,
        "question_sha256": question_hash,
        "expected_prompt_tokens": len(input_ids),
        "request": request,
    })
    gold[example_id] = answer

assert len(requests) == len(gold) == 256
assert {item["id"] for item in requests} == set(gold)

request_bytes = "".join(
    json.dumps(item, sort_keys=True) + "\n"
    for item in requests
).encode()
gold_bytes = json_bytes(gold)
parser_bytes = Path(protocol.__file__).read_bytes()

manifest = {
    "evaluation_id": "gsm8k_dev_v1",
    "source": source,
    "split_sha256": digest(split_bytes),
    "model_source": pool_manifest["model_source"],
    "tokenizer_files_sha256": pool_manifest["tokenizer_files_sha256"],
    "questions": len(requests),
    "fewshot_examples": 0,
    "generation": protocol.GENERATION,
    "context_limit": CONTEXT_LIMIT,
    "request_concurrency": 1,
    "primary_metric": "final-line numeric exact match over all 256 questions",
    "parse_failures": "count as incorrect; also report separately",
    "truncation": "report finish_reason=length separately",
    "primary_comparison": "Selected32 versus Random32",
    "parser_sha256": digest(parser_bytes),
    "requests_sha256": digest(request_bytes),
    "gold_sha256": digest(gold_bytes),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("datasets", "transformers", "tokenizers")
    },
}

OUTPUT.mkdir(parents=True, exist_ok=True)
freeze(OUTPUT / "requests.jsonl", request_bytes)
freeze(OUTPUT / "gold.json", gold_bytes)
freeze(OUTPUT / "manifest.json", json_bytes(manifest))

lengths = [item["expected_prompt_tokens"] for item in requests]
print("\nDevelopment questions:", len(requests))
print("Few-shot examples: 0")
print("Prompt tokens min/max:", min(lengths), max(lengths))
print("Maximum output tokens:", protocol.GENERATION["max_tokens"])
print("Maximum prompt + output budget:", max(lengths) + 1024)
print("Context limit:", CONTEXT_LIMIT)
print("Requests SHA256:", digest(request_bytes))
print("Parser SHA256:", digest(parser_bytes))
print("Saved:", OUTPUT)
print("PASS: development requests, references, and scoring protocol frozen.")
