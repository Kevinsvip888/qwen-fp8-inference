import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
SEED = 42
N_DEV = 256
N_FEWSHOT = 16

sources = json.loads(
    (ROOT / "data/source_revisions.json").read_text()
)
source = sources["math"]
assert source["split"] == "train"

# Verify that we are using the exact downloaded files.
files = []
for entry in source["files"]:
    path = ROOT / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"], f"File changed: {path}"
    files.append(str(path))

dataset = load_dataset(
    "parquet",
    data_files={"train": files},
    split="train",
)
assert len(dataset) == source["rows"]

# Normalize whitespace so formatting-only duplicates stay together.
seen = set()
records = []

for row, example in enumerate(dataset):
    question = " ".join(example["question"].split())
    answer = example["answer"].strip()
    assert question and answer, f"Empty example at row {row}"

    question_hash = hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()

    if question_hash in seen:
        continue

    seen.add(question_hash)
    records.append({
        "row": row,
        "question_sha256": question_hash,
    })

assert len(records) >= N_DEV + N_FEWSHOT + 128

# Split whole examples before tokenization or sequence packing.
random.Random(SEED).shuffle(records)

partitions = {
    "development": records[:N_DEV],
    "fewshot_reserve": records[N_DEV:N_DEV + N_FEWSHOT],
    "calibration": records[N_DEV + N_FEWSHOT:],
}

# Check both source-row and normalized-question separation.
names = list(partitions)
for i, left in enumerate(names):
    for right in names[i + 1:]:
        left_rows = {x["row"] for x in partitions[left]}
        right_rows = {x["row"] for x in partitions[right]}
        assert left_rows.isdisjoint(right_rows)

        left_hashes = {
            x["question_sha256"] for x in partitions[left]
        }
        right_hashes = {
            x["question_sha256"] for x in partitions[right]
        }
        assert left_hashes.isdisjoint(right_hashes)

result = {
    "seed": SEED,
    "source": source,
    "deduplication": "question text with whitespace normalized",
    "original_rows": len(dataset),
    "unique_questions": len(records),
    "duplicates_removed": len(dataset) - len(records),
    "partitions": partitions,
}

destination = ROOT / "data/splits/gsm8k_train_split.json"
destination.parent.mkdir(parents=True, exist_ok=True)

if destination.exists():
    previous = json.loads(destination.read_text())
    if previous != result:
        raise RuntimeError(
            "Existing partition differs; refusing to overwrite it."
        )
else:
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)

print("\nOriginal training rows:", len(dataset))
print("Unique questions:", len(records))
print("Duplicates removed:", result["duplicates_removed"])

for name, entries in partitions.items():
    print(f"{name}: {len(entries)}")

print("Source revision:", source["revision"])
print("Saved:", destination)
print("\nPASS: partitions are disjoint and reproducible.")
