import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
LENGTH = 512
PER_DOMAIN = 256

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

def digest(data):
    return hashlib.sha256(data).hexdigest()

sources = json.loads(
    (ROOT / "data/source_revisions.json").read_text()
)
split_path = ROOT / "data/splits/gsm8k_train_split.json"
split = json.loads(split_path.read_text())
model_source = json.loads(
    (ROOT / "logs/model_revision.json").read_text()
)

assert split["source"] == sources["math"]
assert all(sources[d]["split"] == "train" for d in ("general", "math"))

calibration_rows = {
    item["row"]: item["question_sha256"]
    for item in split["partitions"]["calibration"]
}
heldout_rows = {
    item["row"]
    for name in ("development", "fewshot_reserve")
    for item in split["partitions"][name]
}
assert set(calibration_rows).isdisjoint(heldout_rows)

print("Loading the local tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    ROOT / "models/Qwen3-4B-Instruct-2507",
    local_files_only=True,
)
assert tokenizer.eos_token_id is not None

def load_local(domain):
    files = []
    for entry in sources[domain]["files"]:
        path = ROOT / entry["path"]
        assert digest(path.read_bytes()) == entry["sha256"], path
        files.append(str(path))

    dataset = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
    )
    assert len(dataset) == sources[domain]["rows"]
    return dataset

def tokenized_records(dataset, domain):
    indices = (
        list(range(len(dataset)))
        if domain == "general"
        else sorted(calibration_rows)
    )
    random.Random(f"pool:{args.seed}:{domain}").shuffle(indices)
    seen = set()

    for row in indices:
        example = dataset[row]

        if domain == "general":
            text = example["text"].strip()
            if not text or text.startswith("="):
                continue
        else:
            question = example["question"].strip()
            normalized = " ".join(question.split())
            assert digest(normalized.encode()) == calibration_rows[row]
            text = (
                f"Question: {question}\n"
                f"Answer: {example['answer'].strip()}"
            )

        normalized_hash = digest(" ".join(text.split()).encode())
        if normalized_hash in seen:
            continue
        seen.add(normalized_hash)

        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if not ids:
            continue
        ids = list(ids) + [tokenizer.eos_token_id]
        yield row, text, ids

def pack(records, domain, count=256, length=512):
    blocks, buffer, spans = [], [], []

    for row, text, ids in records:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        offset = 0

        while offset < len(ids) and len(blocks) < count:
            take = min(length - len(buffer), len(ids) - offset)
            start = len(buffer)
            buffer.extend(ids[offset:offset + take])
            spans.append({
                "row": row,
                "text_sha256": text_hash,
                "source_token_start": offset,
                "source_token_end": offset + take,
                "source_token_count": len(ids),
                "sequence_token_start": start,
                "sequence_token_end": start + take,
            })
            offset += take

            if len(buffer) == length:
                blocks.append({
                    "id": f"{domain}-{len(blocks):03d}",
                    "domain": domain,
                    "input_ids": buffer,
                    "sources": spans,
                })
                buffer, spans = [], []

        if len(blocks) == count:
            return blocks

    raise RuntimeError(f"Insufficient tokens for {domain}")

pool = []
counts = {}

for domain in ("general", "math"):
    print(f"Building {domain} sequences...", flush=True)
    dataset = load_local(domain)
    blocks = pack(
        tokenized_records(dataset, domain),
        domain,
        count=PER_DOMAIN,
        length=LENGTH,
    )
    used_rows = {
        span["row"] for block in blocks for span in block["sources"]
    }
    if domain == "math":
        assert used_rows <= set(calibration_rows)
        assert used_rows.isdisjoint(heldout_rows)

    counts[domain] = {
        "sequences": len(blocks),
        "tokens": sum(len(b["input_ids"]) for b in blocks),
        "source_rows_used": len(used_rows),
    }
    print(domain, counts[domain], flush=True)
    pool.extend(blocks)

assert len(pool) == 512
assert all(len(block["input_ids"]) == LENGTH for block in pool)
assert len({block["id"] for block in pool}) == len(pool)

sequence_hashes = {
    digest(json.dumps(block["input_ids"]).encode())
    for block in pool
}
assert len(sequence_hashes) == len(pool), "Duplicate token sequences"

payload = "".join(
    json.dumps(block, separators=(",", ":")) + "\n"
    for block in pool
).encode()

tokenizer_dir = ROOT / "models/Qwen3-4B-Instruct-2507"
tokenizer_files = sorted({
    *tokenizer_dir.glob("tokenizer*"),
    *tokenizer_dir.glob("special_tokens_map.json"),
    *tokenizer_dir.glob("added_tokens.json"),
})

manifest = {
    "seed": args.seed,
    "sequence_length": LENGTH,
    "total_sequences": len(pool),
    "total_tokens": sum(len(b["input_ids"]) for b in pool),
    "domains": counts,
    "source_datasets": sources,
    "model_source": model_source,
    "math_split_sha256": digest(split_path.read_bytes()),
    "tokenizer_files_sha256": {
        p.name: digest(p.read_bytes())
        for p in tokenizer_files if p.is_file()
    },
    "eos_token_id": tokenizer.eos_token_id,
    "packing": {
        "general": "stripped nonempty rows, excluding rows starting with =",
        "math": "Question: {question}\\nAnswer: {answer}",
        "separator": "one tokenizer EOS token after each record",
        "windows": "consecutive nonoverlapping windows within each domain",
        "padding": False,
        "unused_tail": "discarded after each domain reaches its quota",
        "deduplication": "skip repeated whitespace-normalized records encountered",
        "attention": "standard causal attention across packed record boundaries",
    },
    "python": sys.version,
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("datasets", "transformers", "tokenizers")
    },
    "pool_sha256": digest(payload),
}

output_dir = ROOT / f"data/calibration/seed_{args.seed}"
output_dir.mkdir(parents=True, exist_ok=True)

def save_unchanged_or_new(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite different content: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

save_unchanged_or_new(output_dir / "pool.jsonl", payload)
save_unchanged_or_new(
    output_dir / "manifest.json",
    (json.dumps(manifest, indent=2) + "\n").encode(),
)

print("\nSequences:", manifest["total_sequences"])
print("Tokens per sequence:", LENGTH)
print("Total tokens:", manifest["total_tokens"])
print("Pool SHA256:", manifest["pool_sha256"])
print("Saved:", output_dir)
print("\nPASS: exact lengths, unique sequences, and math split separation.")
