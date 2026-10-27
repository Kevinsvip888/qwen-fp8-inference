# Captured environments

These installed-package lists were exported from the completed experiment machine during publication preparation. Keep the three environments separate.

| File | Role |
|---|---|
| `serve-requirements.txt` | GPU inference, native serving benchmarks, and profiling |
| `quant-requirements.txt` | BF16 feature extraction and ModelOpt static FP8 calibration/export |
| `eval-requirements.txt` | CPU harness client, tokenizer, data preparation, and score analysis |

The serving environment uses Transformers 5.19.0; quantization uses 5.14.1. The evaluation environment intentionally has no PyTorch. Exact versions are recorded in each text file and critical versions are also linked from archived protocols.

These are package snapshots rather than a platform-independent lock or an assurance that every listed wheel remains downloadable. In particular, CUDA-enabled PyTorch wheels require a compatible package source and driver. See `docs/REPRODUCE.md` for reconstruction commands and boundaries. Do not upgrade a working archived-study environment to resolve an unrelated publication task.
