# Archived experiment metadata

`original/` preserves supplied metadata bytes at their original relative paths. These are evidence records, not a replacement for weights, data files, or complete run directories.

- `original/logs/model_revision.json`: original model repository/revision and local path.
- `original/data/source_revisions.json`: WikiText and GSM8K training shard revisions/hashes.
- `original/data/splits/`: training row IDs and question hashes; no full training corpus.
- `original/data/calibration/`: shared quantization recipe, pool definitions, and selected sequence IDs.
- `original/data/features/`: feature definition and environment metadata, not all extracted feature records.
- `original/data/evaluation/`: source manifests, task definitions, original split/scoring settings, final model plan, and preparation receipts. GSM8K task definitions contain the five fixed training demonstrations.
- `original/models/`: calibration/export manifests only; no tensors or tokenizer files.
- `original/outputs/`: original performance, profiling, and final-quality runtime protocols.

`script_sha256.json` records the original 36 experiment scripts. `public_file_sha256.json` inventories the publication files other than itself. `report_sources.json` maps copied reports in `results/` to their original local paths. Neither inventory is a signature or a substitute for raw experiment evidence.

The presence of `original/data/evaluation/.../final_plan.json` does not make a fresh study compatible with that plan: it identifies nine specific exported checkpoints and their exact manifests. Original audit scripts read live paths outside this archive and still require their complete dependencies.
