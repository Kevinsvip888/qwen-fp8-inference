import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
manifest_path = ROOT / "data/source_revisions.json"

specs = {
    "general": {
        "repo_id": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "columns": ["text"],
    },
    "math": {
        "repo_id": "openai/gsm8k",
        "config": "main",
        "columns": ["question", "answer"],
    },
}

manifest = (
    json.loads(manifest_path.read_text())
    if manifest_path.exists()
    else {}
)
api = HfApi()

def save_manifest():
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for domain, spec in specs.items():
    if domain not in manifest:
        info = api.dataset_info(spec["repo_id"])
        manifest[domain] = {
            "repo_id": spec["repo_id"],
            "config": spec["config"],
            "split": "train",
            "revision": info.sha,
        }
        save_manifest()

    record = manifest[domain]
    assert record["repo_id"] == spec["repo_id"]
    assert record["config"] == spec["config"]
    assert record["split"] == "train"

    local_dir = ROOT / "data/raw" / domain

    snapshot_download(
        repo_id=record["repo_id"],
        repo_type="dataset",
        revision=record["revision"],
        allow_patterns=[f'{spec["config"]}/train-*.parquet'],
        local_dir=str(local_dir),
        max_workers=2,
    )

    files = sorted(
        (local_dir / spec["config"]).glob("train-*.parquet")
    )
    if not files:
        raise RuntimeError(f"No training files found for {domain}")

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in files]},
        split="train",
    )

    for column in spec["columns"]:
        assert column in dataset.column_names, (domain, column)

    record["rows"] = len(dataset)
    record["columns"] = dataset.column_names
    record["files"] = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    save_manifest()

    print(f"\n{domain.upper()}")
    print("Repository:", record["repo_id"])
    print("Revision:", record["revision"])
    print("Split:", record["split"])
    print("Rows:", record["rows"])
    print("Columns:", record["columns"])
    print("Training files:", len(files))

print("\nPASS: training sources downloaded and revisions recorded.")
