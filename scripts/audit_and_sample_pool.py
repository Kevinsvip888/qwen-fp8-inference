import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

folder = ROOT / f"data/calibration/seed_{args.seed}"

def digest(data):
    return hashlib.sha256(data).hexdigest()

manifest = json.loads((folder / "manifest.json").read_text())
payload = (folder / "pool.jsonl").read_bytes()
assert digest(payload) == manifest["pool_sha256"]
assert manifest["seed"] == args.seed
assert manifest["sequence_length"] == 512
assert manifest["total_sequences"] == 512
assert manifest["total_tokens"] == 262144

pool = [json.loads(line) for line in payload.splitlines()]
assert len(pool) == 512

sources = json.loads(
    (ROOT / "data/source_revisions.json").read_text()
)
assert sources == manifest["source_datasets"]
assert json.loads(
    (ROOT / "logs/model_revision.json").read_text()
) == manifest["model_source"]

split_path = ROOT / "data/splits/gsm8k_train_split.json"
assert digest(split_path.read_bytes()) == manifest["math_split_sha256"]
split = json.loads(split_path.read_text())
assert split["source"] == sources["math"]

allowed_math = {
    item["row"]: item["question_sha256"]
    for item in split["partitions"]["calibration"]
}
reserved_math = {
    item["row"]
    for name in ("development", "fewshot_reserve")
    for item in split["partitions"][name]
}
assert set(allowed_math).isdisjoint(reserved_math)

tokenizer_dir = ROOT / "models/Qwen3-4B-Instruct-2507"
for name, expected in manifest["tokenizer_files_sha256"].items():
    assert digest((tokenizer_dir / name).read_bytes()) == expected

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir, local_files_only=True
)
assert tokenizer.eos_token_id == manifest["eos_token_id"]

datasets = {}
for domain in ("general", "math"):
    files = []
    for entry in sources[domain]["files"]:
        path = ROOT / entry["path"]
        assert digest(path.read_bytes()) == entry["sha256"]
        files.append(str(path))

    datasets[domain] = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
    )
    assert len(datasets[domain]) == sources[domain]["rows"]

cache = {}

def reconstruct_source(domain, row):
    key = (domain, row)
    if key not in cache:
        assert 0 <= row < len(datasets[domain])
        example = datasets[domain][row]

        if domain == "general":
            text = example["text"].strip()
            assert text and not text.startswith("=")
        else:
            assert row in allowed_math
            assert row not in reserved_math
            question = example["question"].strip()
            normalized = " ".join(question.split())
            assert digest(normalized.encode()) == allowed_math[row]
            text = (
                f"Question: {question}\n"
                f"Answer: {example['answer'].strip()}"
            )

        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        ids = list(ids) + [tokenizer.eos_token_id]
        cache[key] = (ids, digest(text.encode()))

    return cache[key]

counts = Counter()
used_rows = {"general": set(), "math": set()}
previous = {"general": None, "math": None}
sequence_hashes = set()

print("Reconstructing saved sequences...", flush=True)

for number, block in enumerate(pool, start=1):
    domain = block["domain"]
    assert domain in used_rows
    assert block["id"] == f"{domain}-{counts[domain]:03d}"
    assert len(block["input_ids"]) == 512

    cursor = 0
    for span in block["sources"]:
        row = span["row"]
        ids, text_hash = reconstruct_source(domain, row)
        assert span["text_sha256"] == text_hash
        assert span["source_token_count"] == len(ids)

        a = span["source_token_start"]
        b = span["source_token_end"]
        start = span["sequence_token_start"]
        end = span["sequence_token_end"]

        assert 0 <= a < b <= len(ids)
        assert start == cursor
        assert end == start + b - a
        assert end <= 512
        assert block["input_ids"][start:end] == ids[a:b]

        # Check stream continuity, including across sequence boundaries.
        last = previous[domain]
        if last is not None and last[0] == row:
            assert a == last[1]
        else:
            if last is not None:
                assert last[1] == last[2]
            assert a == 0
            assert row not in used_rows[domain]
            used_rows[domain].add(row)

        previous[domain] = (row, b, len(ids))
        cursor = end

    assert cursor == 512
    sequence_hashes.add(digest(json.dumps(block["input_ids"]).encode()))
    counts[domain] += 1

    if number % 128 == 0:
        print(f"Verified {number}/512 sequences", flush=True)

assert counts == {"general": 256, "math": 256}
assert len(sequence_hashes) == 512

for domain in counts:
    assert len(used_rows[domain]) == (
        manifest["domains"][domain]["source_rows_used"]
    )

audit = {
    "pool_sha256": manifest["pool_sha256"],
    "verified_sequences": len(pool),
    "verified_tokens": sum(len(b["input_ids"]) for b in pool),
    "source_rows_reconstructed": {
        domain: len(rows) for domain, rows in used_rows.items()
    },
    "source_tokens_match": True,
    "packing_continuity_verified": True,
    "math_reserved_rows_excluded": True,
}

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

save_json(folder / "pool_audit.json", audit)
print("\nPASS: all saved tokens reconstructed from source records.")

# Use one fixed random ordering per domain for both subset sizes.
lookup = {block["id"]: block for block in pool}
orders = {}
seeds = {}

for domain in ("general", "math"):
    ids = sorted(
        sid for sid, block in lookup.items()
        if block["domain"] == domain
    )
    seed = f"random-subsets:{args.seed}:{domain}:v1"
    random.Random(seed).shuffle(ids)
    orders[domain] = ids
    seeds[domain] = seed

subset_manifest = {
    "pool_sha256": manifest["pool_sha256"],
    "random_seed_strings": seeds,
    "nested": True,
    "subsets": {},
}

for name, per_domain in (("Random32", 16), ("Random128", 64)):
    # Alternate general and math sequences.
    selected = [
        sid
        for pair in zip(
            orders["general"][:per_domain],
            orders["math"][:per_domain],
        )
        for sid in pair
    ]
    assert len(selected) == len(set(selected)) == 2 * per_domain

    blocks = [lookup[sid] for sid in selected]
    tokens = sum(len(block["input_ids"]) for block in blocks)
    assert tokens == len(selected) * 512
    assert Counter(b["domain"] for b in blocks) == {
        "general": per_domain,
        "math": per_domain,
    }

    content = "".join(
        json.dumps(block, separators=(",", ":")) + "\n"
        for block in blocks
    ).encode()
    filename = f"{name.lower()}.jsonl"
    save(folder / filename, content)

    subset_manifest["subsets"][name] = {
        "file": filename,
        "sha256": digest(content),
        "sequence_ids": selected,
        "sequences": len(selected),
        "tokens": tokens,
        "sequences_per_domain": per_domain,
    }
    print(f"{name}: sequences={len(selected)}, tokens={tokens}")

small = subset_manifest["subsets"]["Random32"]["sequence_ids"]
large = subset_manifest["subsets"]["Random128"]["sequence_ids"]
assert small == large[:32]

save_json(folder / "random_subsets_manifest.json", subset_manifest)
print("PASS: balanced, reproducible, nested random subsets saved.")
