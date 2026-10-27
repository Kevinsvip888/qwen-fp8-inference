#!/usr/bin/env python3
"""Pin and verify final-evaluation data; do not run or score any test item.

Run in .venv-eval. Existing Day 3 source metadata is required. Network/proxy
settings should be supplied before starting Python (see the accompanying command).
"""

import argparse
import hashlib
import json
import re
from importlib.metadata import distribution, version
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SPECS = {
    "gsm8k": {
        "repo_id": "openai/gsm8k", "config": "main",
        "splits": ["train", "test"],
    },
    "arc_challenge": {
        "repo_id": "allenai/ai2_arc", "config": "ARC-Challenge",
        "splits": ["train", "validation", "test"],
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/root/autodl-tmp/qwen-inference-lab"))
    args = parser.parse_args()
    root = args.root.resolve()
    original = json.loads((root / "data/source_revisions.json").read_text())
    require(original["math"]["repo_id"] == "openai/gsm8k",
            "Unexpected original math repository")
    require(original["math"]["revision"] == GSM_REVISION,
            "Original GSM8K revision differs; do not silently change sources")

    destination = root / "data/evaluation/day7_sources"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "source_revisions.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else {"schema_version": 1, "datasets": {}})
    require(manifest.get("schema_version") == 1, "Unknown manifest format")
    api = HfApi()

    for name, spec in SPECS.items():
        if name not in manifest["datasets"]:
            revision = (GSM_REVISION if name == "gsm8k" else
                        api.dataset_info(spec["repo_id"], timeout=120).sha)
            require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
                    "Expected a full dataset commit SHA")
            record = {**spec, "revision": revision, "complete": False,
                      "files": []}
            manifest["datasets"][name] = record
            save_json(manifest_path, manifest)  # Pin before downloading.

        record = manifest["datasets"][name]
        for key, expected in spec.items():
            require(record[key] == expected, f"Existing {name}/{key} differs")
        require(bool(re.fullmatch(r"[0-9a-f]{40}", record["revision"])),
                "Invalid saved revision")
        if name == "gsm8k":
            require(record["revision"] == GSM_REVISION, "GSM8K pin changed")

        # Save the exact list of shards once; reruns reuse it and the commit pin.
        if "repository_files" not in record:
            repo_files = api.list_repo_files(
                spec["repo_id"], repo_type="dataset", revision=record["revision"])
            pattern = re.compile(
                rf"{re.escape(spec['config'])}/(train|validation|test)-.*\.parquet")
            record["repository_files"] = sorted(
                path for path in repo_files
                if (match := pattern.fullmatch(path))
                and match.group(1) in spec["splits"])
            require(record["repository_files"], f"No Parquet files for {name}")
            save_json(manifest_path, manifest)

        previous = {item["repository_path"]: item for item in record["files"]}
        verified = []
        for repo_path in record["repository_files"]:
            split = Path(repo_path).name.split("-", 1)[0]
            local = destination / "raw" / name / repo_path
            if repo_path in previous:
                expected = previous[repo_path]
                require(local.is_file(), f"Previously verified file missing: {local}")
                require(sha256(local) == expected["sha256"],
                        f"Previously verified file changed: {local}")
            else:
                print(f"Downloading {name}/{repo_path}...", flush=True)
                hf_hub_download(
                    repo_id=spec["repo_id"], repo_type="dataset",
                    revision=record["revision"], filename=repo_path,
                    local_dir=destination / "raw" / name)
            # Read only Parquet metadata; no question/answer content is printed.
            rows = pq.ParquetFile(local).metadata.num_rows
            item = {"repository_path": repo_path, "split": split,
                    "path": str(local.relative_to(root)),
                    "bytes": local.stat().st_size, "sha256": sha256(local),
                    "rows": rows}
            if repo_path in previous:
                require(item == previous[repo_path], "Saved file metadata differs")
            verified.append(item)
            # Preserve progress if a later download fails.
            previous[repo_path] = item
            record["files"] = [previous[key] for key in sorted(previous)]
            save_json(manifest_path, manifest)

        for split in spec["splits"]:
            require(any(item["split"] == split for item in verified),
                    f"Missing {name}/{split} shard")
        record["files"] = verified
        record["complete"] = True
        save_json(manifest_path, manifest)
        print(f"\n{name}: {spec['repo_id']} @ {record['revision']}")
        for split in spec["splits"]:
            count = sum(item["rows"] for item in verified if item["split"] == split)
            print(f"  {split}: {count} rows")

    # Archive the installed task templates for the next protocol-creation step.
    tasks = Path(distribution("lm_eval").locate_file("lm_eval/tasks"))
    templates = {}
    for relative in ("gsm8k/gsm8k.yaml", "arc/arc_challenge.yaml", "arc/arc_easy.yaml"):
        source = tasks / relative
        require(source.is_file(), f"Installed task template missing: {source}")
        target = destination / "task_defaults" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(target.read_bytes() == source.read_bytes(),
                    f"Installed task template changed: {relative}")
        else:
            target.write_bytes(source.read_bytes())
        templates[relative] = {"sha256": sha256(target),
                               "path": str(target.relative_to(root))}
        if relative == "gsm8k/gsm8k.yaml":
            print("\nInstalled GSM8K task defaults:")
            print(source.read_text())
    defaults = {"lm_eval_version": version("lm_eval"), "files": templates}
    defaults_path = destination / "task_defaults_manifest.json"
    if defaults_path.exists():
        require(json.loads(defaults_path.read_text()) == defaults,
                "Task defaults manifest changed")
    else:
        save_json(defaults_path, defaults)
    print(f"\nSaved: {manifest_path}")
    print("PASS: source revisions, file hashes, split counts and task defaults saved.")
    print("No test questions were scored; evaluation protocol is not yet frozen.")


if __name__ == "__main__":
    main()
