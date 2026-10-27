import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parents[1]
repo_id = "Qwen/Qwen3-4B-Instruct-2507"
model_dir = root / "models" / "Qwen3-4B-Instruct-2507"
record = root / "logs" / "model_revision.json"

if record.exists():
    metadata = json.loads(record.read_text())
    assert metadata["repo_id"] == repo_id
    print("Reusing the recorded model revision.", flush=True)
else:
    print("Contacting Hugging Face for the model revision...", flush=True)
    info = HfApi().model_info(repo_id, timeout=30)

    metadata = {
        "repo_id": repo_id,
        "revision": info.sha,
        "local_dir": str(model_dir),
    }
    record.write_text(json.dumps(metadata, indent=2) + "\n")

print("Model:", repo_id, flush=True)
print("Revision:", metadata["revision"], flush=True)
print("Destination:", model_dir, flush=True)

snapshot_download(
    repo_id=repo_id,
    revision=metadata["revision"],
    local_dir=str(model_dir),
    max_workers=2,
)

config = json.loads((model_dir / "config.json").read_text())
print("Model type:", config["model_type"])
print("Transformer layers:", config["num_hidden_layers"])
print("DOWNLOAD COMPLETE")
