# Evidence and interpretation

## What this snapshot contains

The original experiment scripts and aggregate reports are accompanied by the exact source revisions, GSM8K training-split IDs and hashes, calibration-pool manifests, selected sequence IDs, feature definitions, FP8 recipe, checkpoint sidecars, Day 7 task definitions, and frozen final plan. Metadata is archived under `provenance/original/` using its original relative path. These copies are records; they are not automatically read by scripts expecting live `data/`, `logs/`, `models/`, or `outputs/` files.

The original report copies are under `results/`. `provenance/report_sources.json` maps them to their former locations. Paths inside these files still describe the original experiment. Reports were copied without rewriting their evidence hashes or claims.

`environment/*-requirements.txt` records the three installed environments when the publication metadata was exported. The serve and evaluation snapshots match the respective package-list files supplied for review. The quantization recipe separately records its critical package versions. These snapshots contain installed package pins; they are not independently tested installers for arbitrary GPUs.

## What is excluded

No model weights, downloaded parquet datasets, packed calibration-token files, per-question evaluation samples, raw HTTP responses, server/client logs, or GPU trace files are included in this public snapshot. The original local backup remains necessary to rerun the full raw-evidence audit. A published hash identifies an expected file; it does not make the missing file available or independently prove its contents.

`tools/verify_public_snapshot.py` validates the public files and internal links only. The original Day 5, Day 6, and Day 7 summarizers performed deeper checks against raw evidence on the experiment machine before writing the published reports. The publication check must not be described as a repetition of those audits.

## Frozen protocol identifiers

| Record | SHA256 |
|---|---|
| FP8 calibration protocol | `aaca5144109cec26410a142939d5d65e69551bbb8990dd5d1267ec8b25528e9c` |
| Day 7 development protocol | `44dc329b87c700a065e15c7395f040d8beb4ea13d6063ab28efa9f806ca7449f` |
| Day 7 final model plan | `1bd1f014f7409f32e93347cbd25e39eca79a782a614acd0234a5ed7c003e4558` |
| Day 7 final preparation | `1ddf3a16aad32e1273d90b2e0ba7dd0d84f9e0edd281051c396fc9ca405faa95` |

The development manifest retains its historical pre-evaluation status fields. Final scope and completion are established by the final plan and result summaries, not by changing that older manifest after the fact.

## How to read the results

The primary throughput is the ratio of the two models' means across three run-level rates. Repetitions characterize observed run variability; they are not thousands of independent hardware trials. Memory is the whole-GPU sampled maximum, sampled once per second over the client startup/warm-up/measurement window. It excludes server startup, can miss short spikes, and includes allocator state. Checkpoint size, model-load allocation, and total GPU allocation are different measurements.

GPU profile values sum recorded kernel durations. Overlap, CPU scheduling, idle gaps, and profiling overhead prevent interpreting those sums as complete request latency. Group labels are coarse name-based classifications. A fused arithmetic/conversion kernel's entire duration is not the cost of quantization alone.

For quality, one question is one paired unit. GSM8K's strict and flexible filters produce two sample records for the same generated answer; they must not be counted as two independent questions. ARC produces one likelihood request per answer choice; 4,687 API requests correspond to 1,172 questions.

Each checkpoint was evaluated on 2,491 questions using 6,006 API requests. Ten checkpoints give 24,910 model-question evaluations and 60,060 recorded inference requests; the unique official test set still contains only 2,491 questions.

The primary comparison is Random32 seed 42 minus BF16, separately for each task. The 20,000-replicate question bootstrap uses seed 20261027. Exact two-sided McNemar tests condition on discordant question pairs. Across-pool intervals resample questions jointly across the three fixed pools rather than treating the three predictions for one question as independent samples. They do not estimate uncertainty over all possible calibration pools. No multiple-comparison adjustment is applied to the exploratory per-seed tests.

Math strict-match can score an early `#### number` correctly even if later text repeats or contradicts it. Missing markers and token-limit finishes are separate diagnostics. No parser or output budget was retuned from the official test results. Day 4 used a different zero-shot prompt and parser and is not directly comparable to the Day 7 five-shot scores.

## Known implementation fixes during development

The development client initially supplied `batch_size` twice; it now passes it only through the harness CLI. The sample auditor was also corrected to identify each `(question ID, filter)` pair, and to read loglikelihood values serialized as numeric strings. Existing saved responses were re-audited after these format-handling fixes. These were pipeline fixes, not changes to model answers or the frozen scoring rules.

## Review scope

Publication preparation checked Python parsing, shell syntax, obvious credential patterns, original script hashes, recorded package versions, final-plan/checkpoint links, and aggregate score arithmetic. GPU execution was performed on the experiment machine, not repeated during publication preparation. Static checks do not establish that every cold-start command will work on a different host.

The public file inventories detect accidental changes relative to this snapshot. They are not cryptographic signatures of authorship. If maintained code or documentation changes later, keep this archived snapshot identifiable and record a new inventory instead of silently relabeling changed files as the original experiment.
