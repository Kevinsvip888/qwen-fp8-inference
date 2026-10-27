# Script map

The `scripts/` directory preserves the original experiment source. Most early scripts assume the recorded AutoDL project path. New publication helpers are isolated under `tools/` so that historical source hashes remain valid.

## Setup and smoke checks

| Script | Purpose |
|---|---|
| `download_model.py` | Download the source model at the revision in `logs/model_revision.json`; resolves a revision if no record exists |
| `serve_bf16.sh` | Start the original BF16 smoke/development server |
| `check_fixed_prompts.py` | Send the six fixed chat smoke prompts to BF16 or the smoke FP8 server |
| `quantize_fp8_smoke.py` | Calibrate/export the separate eight-example FP8 smoke checkpoint |
| `serve_fp8_smoke.sh` | Serve that smoke checkpoint |

## Calibration study

| Script | Purpose |
|---|---|
| `prepare_data_sources.py` | Download pinned training shards and record dataset hashes |
| `split_math_training.py` | Deduplicate and divide GSM8K training questions into development, few-shot reserve, and calibration-eligible partitions |
| `build_calibration_pool.py` | Pack the seed-specific balanced 512-sequence candidate pool |
| `audit_and_sample_pool.py` | Reconstruct packed tokens from sources; save balanced nested Random32/Random128 subsets |
| `extract_activation_features.py` | Measure BF16 RMS/P99 features at eight MLP locations |
| `select_calibration_sequences.py` | Standardize within domain and select the farthest-point Selected32 subset |
| `quantize_fp8_experiment.py` | Load fresh BF16 weights, calibrate the chosen subset, export/check the FP8 checkpoint and manifest |
| `compare_calibration_ranges.py` | Compare activation amax values and verify nested-max consistency |
| `prepare_remaining_seeds.sh` | Run pool construction, auditing, feature extraction, selection, and three exports for seeds 43 and 44 |
| `serve_fp8_experiment.sh` | Serve a chosen experimental FP8 checkpoint for smoke/Day 4 checks |
| `smoke_fp8_experiment.py` | Check the experimental checkpoint's six responses and kernel evidence |

## Development scoring

| Script | Purpose |
|---|---|
| `gsm8k_eval_protocol.py` | Define and self-check Day 4 numerical normalization and final-answer parsing |
| `prepare_gsm8k_development.py` | Freeze the separate 256-question zero-shot development requests and references |
| `evaluate_gsm8k_development.py` | Call the active matching server, save/cache answers, and score the chosen method/seed |
| `analyze_gsm8k_development.py` | Paired seed-specific development comparisons and failure review |
| `evaluate_remaining_seeds.py` | Sequentially evaluate the six seed-43/44 FP8 checkpoints |
| `summarize_calibration_study.py` | Aggregate the three calibration pools without treating repeated questions as independent |

## Performance and GPU profiles

| Script | Purpose |
|---|---|
| `serve_day5_benchmark.sh` | Start the fixed-cache, fixed-attention BF16 or Random32 performance server |
| `run_day5_benchmark.sh` | Run native vLLM serving measurements, sample GPU memory, and check exact token counts |
| `summarize_day5_benchmarks.py` | Audit and aggregate the 36 native benchmark runs |
| `serve_day6_profile.sh` | Start a profiling server and record its unique trace directory |
| `profile_day6_phases.py` | Capture single-request prefill and decode windows with raw-response/trace records |
| `profile_day6_decode_c4.py` | Capture four-request steady decode windows |
| `summarize_day6_profiles.py` | Verify selected sessions/phase labels, classify kernels, and aggregate 18 windows |

## Final quality evaluation

| Script | Purpose |
|---|---|
| `prepare_day7_sources.py` | Download/check pinned GSM8K and ARC splits and archive installed harness defaults |
| `prepare_day7_quality.py` | Construct development tasks, fixed five-shot examples, real harness prompts, and budgets |
| `run_day7_development.py` | Run/audit eight- or 100-question-per-task pipeline checks through the recording proxy |
| `prepare_day7_final.py` | Check the exact frozen plan and build full official-test requests without inference |
| `run_day7_final.py` | Evaluate/audit one frozen-plan checkpoint; validate raw responses and ARC likelihoods |
| `run_day7_remaining.py` | Sequentially finish the ten-model plan while preserving completed checkpoints |
| `summarize_day7_quality.py` | Verify all ten runs, pair question outcomes, and compute final comparisons |

## Publication helpers

| Tool | Purpose |
|---|---|
| `tools/verify_public_snapshot.py` | Standard-library check of the published source/configuration/report snapshot; no GPU or raw-response audit |
| `tools/restore_source_pins.py` | Restore four source/split records into live locations only if absent or already identical |

Check the reproduction guide before running a script. A server launcher occupies the terminal until stopped; the request client runs in a separate terminal. Only one model server should occupy port 8000 at a time.
