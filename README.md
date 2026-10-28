# Qwen FP8 Inference

Measured FP8 quantization, serving performance, GPU kernel behavior, and answer quality for **Qwen3-4B-Instruct-2507 on an RTX 4090 24 GB**.

The fixed Random32 seed-42 checkpoint delivered **1.3429× output throughput** and used **3.25 GiB less sampled GPU memory** than BF16 on the primary serving workload. On the full quality tests, its observed differences were **−0.3033 percentage points on GSM8K** and **+0.6826 points on ARC-Challenge**. The paired confidence intervals include zero; this is not proof of equivalent accuracy.

## Results

| Measurement | BF16 | FP8 Random32, seed 42 | Comparison |
|---|---:|---:|---|
| Primary output throughput | 323.61 tokens/s | 434.57 tokens/s | 1.3429× |
| Primary sampled GPU maximum | 12,913 MiB | 9,585 MiB | 3.25 GiB less |
| GSM8K test, strict exact match | 89.61% | 89.31% | −0.3033 pp |
| ARC-Challenge test, normalized accuracy | 58.70% | 59.39% | +0.6826 pp |

Primary performance workload: 1,024 input tokens, 128 output tokens, concurrency 4; three runs of 200 measured requests per model after warm-up. Both models used a **BF16 KV cache with the same 4 GiB budget**. The quality tests used a separate, fixed evaluation protocol.

The initial FP8 smoke export also reduced safetensors storage from 7.4925 GiB to 4.1087 GiB (**45.16%**). That measurement belongs to the eight-example smoke checkpoint, not to a measurement of whole-GPU memory or every experimental checkpoint.

| Report | Contents |
|---|---|
| [Serving performance](results/day5_report.md) | 36 runs, 7,200 measured requests; throughput, latency, and sampled memory |
| [GPU profiling](results/day6_report.md) | 18 repeated windows; prefill/decode kernel groups and timing limits |
| [Full quality evaluation](results/day7_report.md) | BF16 plus nine FP8 checkpoints, paired comparisons, and stopping diagnostics |
| [Development summary](results/day4_summary.json) | Earlier, separate 256-question development protocol |
| [Reproduction guide](docs/REPRODUCE.md) | Environment setup and commands grouped by day |
| [Evidence and limitations](docs/EVIDENCE.md) | What is published, what requires the local archive, and interpretation limits |
| [Script map](docs/SCRIPT_MAP.md) | The role of each experiment script |

## What was compared

All experimental FP8 checkpoints use the same ModelOpt static FP8 recipe. The model weights and linear input activations are quantized; embeddings and the output head remain unquantized. The KV cache stays BF16. vLLM logs confirmed `CutlassFP8ScaledMMLinearKernel` execution.

For each calibration-pool seed (42, 43, 44), a balanced candidate pool contains **512 sequences of exactly 512 tokens**: 256 WikiText sequences and 256 GSM8K training sequences. Each sequence can contain parts of several source records.

| Method | Calibration sequences | Calibration tokens | Selection |
|---|---:|---:|---|
| Random32 | 32 | 16,384 | 16 sequences per domain |
| Selected32 | 32 | 16,384 | 16 per domain using BF16 activation features and farthest-point selection |
| Random128 | 128 | 65,536 | 64 per domain; contains Random32 from the same pool |

Selected32 uses 16 features: RMS and absolute-value P99 at eight MLP input locations. Features are standardized within each domain. Selection starts near the feature centroid, then repeatedly chooses the point farthest from the current selected set. This is an activation-coverage heuristic, not an accuracy objective.

**Finding:** activation-based selection did not consistently improve accuracy. Across the three observed calibration pools, Selected32 minus Random32 averaged −0.6571 pp on math and +0.2844 pp on science. The corresponding question-bootstrap intervals include zero. Random128 also showed no clear overall accuracy advantage over Random32. The negative result is retained rather than choosing the best seed after seeing test scores.

## Full quality table

All models used the same 1,319 math and 1,172 science test questions. Percentages below are rounded for display; the JSON retains counts and full precision.

| Model | Math strict % | Science acc_norm % | Math length finishes |
|---|---:|---:|---:|
| BF16 | 89.61 | 58.70 | 73 |
| Random32-seed42 | 89.31 | 59.39 | 78 |
| Selected32-seed42 | 89.16 | 58.79 | 113 |
| Random128-seed42 | 89.16 | 59.30 | 86 |
| Random32-seed43 | 89.46 | 58.62 | 72 |
| Selected32-seed43 | 89.08 | 59.39 | 71 |
| Random128-seed43 | 89.69 | 59.56 | 108 |
| Random32-seed44 | 90.67 | 59.30 | 116 |
| Selected32-seed44 | 89.23 | 59.98 | 98 |
| Random128-seed44 | 90.22 | 59.64 | 105 |

## Data separation

Three dataset repositories serve different purposes:

| Source | Calibration / development use | Final quality test |
|---|---|---|
| WikiText-103 raw | Training text for general-domain calibration | None |
| GSM8K | Training split divided into 7,201 calibration-eligible, 256 development, and 16 reserved few-shot questions | All 1,319 official test questions |
| ARC-Challenge | 299 validation questions for development checks | All 1,172 official test questions |

The 7,201 GSM8K questions form the eligible calibration source; only the source rows needed to build each candidate pool are consumed. **32 means packed calibration sequences, not test questions.** The official tests are separate from calibration and are evaluated in full for all ten checkpoints.

Final GSM8K evaluation uses five fixed examples from the reserved training partition, greedy generation, a 2,048-token output limit, and the harness strict `#### number` extractor. ARC-Challenge uses zero-shot answer-choice loglikelihood scoring, with `acc_norm` as the primary metric. Neither task applies a chat template. Calibration-pool seeds vary; generation seed remains 42.

## Environment

Linux / Ubuntu 22.04.5, x86-64; Python 3.12.15; one RTX 4090 24 GB; 16 vCPU Intel Xeon Gold 6430; recorded NVIDIA driver 580.76.05.

| Environment | Main recorded packages |
|---|---|
| `.venv-serve` | PyTorch 2.13.0+cu132, vLLM 0.32.0, Transformers 5.19.0 |
| `.venv-quant` | PyTorch 2.13.0+cu132, NVIDIA ModelOpt 0.47.0, Transformers 5.14.1 |
| `.venv-eval` | lm-eval 0.4.13, datasets 5.1.0, Transformers 5.19.0; no PyTorch |

Captured package lists are in [environment/](environment/). They are environment snapshots, not portable dependency lockfiles. GPU wheel availability and driver compatibility must be checked on another machine. The PyTorch-free evaluation environment only prepares/tokenizes requests and calls the separate GPU server.

Performance and profiling used one physical RTX 4090; the final quality evaluation used another RTX 4090 of the same model. GPU identities are preserved in the archived protocols. All models within each comparison used that experiment's common GPU and environment.

## Inspect the public snapshot

From a checkout, with Python 3.10 or later:

```bash
python3 tools/verify_public_snapshot.py
```

This CPU-only command checks published file hashes, original script hashes, frozen configuration links, checkpoint sidecar identities, training-split separation, and report totals. It does **not** replay raw-response or trace audits: those require the original experiment archive.

## Reproduce or extend

Start with [docs/REPRODUCE.md](docs/REPRODUCE.md). The original scripts assume `/root/autodl-tmp/qwen-inference-lab`; some later scripts additionally accept `--root`. The scripts are preserved as experiment source, including their original paths and checksum guards.

This repository publishes the source, result summaries, and a configuration snapshot. **A fresh clone is not a one-command replay of the full archived study.** Raw downloaded datasets, model weights, packed calibration tokens, per-question generated answers, and profiler traces are excluded. The final runner intentionally requires the original frozen artifact identities, including checkpoint hashes and runtime metadata. Freshly generated exports may have different bytes even with the same scientific settings. See the guide for the distinction between a new experiment and an archived replay.

| Directory | Purpose |
|---|---|
| `scripts/` | Original experiment scripts, with recorded hashes |
| `tools/` | Public-snapshot verification and source-pin restoration |
| `environment/` | Captured package versions for the three environments |
| `results/` | Copied original aggregate reports and JSON summaries |
| `provenance/original/` | Original revisions, split IDs, protocols, task definitions, and checkpoint manifests |
| `data/`, `models/`, `logs/`, `outputs/`, `backups/` | Local experiment state, excluded from Git |

## Limits

- Results cover one model family, one GPU model, and the stated workloads; they are not general production guarantees.
- The final tests measure math-answer extraction and science answer-choice likelihoods. They do not measure every aspect of instruction following or conversation quality.
- GSM8K responses sometimes repeated content or hit the token limit. BF16 had 73/1,319 length finishes; Random32 seed 42 had 78/1,319. A correct extracted number can coexist with a poor continuation.
- Profiling indicates that matrix calculations account for the largest recorded GPU-time savings. Kernel-duration sums are not wall-clock latency, and no bandwidth-saturation claim is made.
- Three-pool comparisons condition on those observed pools. Secondary per-seed p-values are unadjusted exploratory diagnostics, not a search for a winning seed.
- The final tests were held out from this project's calibration and development. This does not establish absence from the pretrained model's original training data.

## Upstream projects and data

[Qwen model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), [vLLM](https://github.com/vllm-project/vllm), [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), and [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc).

Model weights and bulk datasets are not bundled. Archived GSM8K task definitions contain the five fixed training demonstrations used in the experiment. Dataset revisions and file hashes are retained for attribution and reproducibility.
