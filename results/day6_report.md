# Day 6 — GPU profiling findings

Validated 18 repeated capture windows: BF16 and FP8, three repetitions each for single-request prefill, single-request decode, and four-request steady decode. One-repetition pilots are excluded.

## GPU work comparison

Values are summed GPU kernel durations, averaged across three captures. They are not end-to-end request latency or throughput.

| Window | Engine steps | BF16 ms ± sample SD | FP8 ms ± sample SD | Less GPU work |
| --- | ---: | ---: | ---: | ---: |
| prefill_c1 | 1 | 53.097 ± 0.106 | 42.514 ± 0.029 | 19.93% |
| decode_c1 | 8 | 76.844 ± 0.020 | 57.487 ± 0.028 | 25.19% |
| decode_c4 | 8 | 85.338 ± 0.021 | 63.407 ± 0.009 | 25.70% |

## Kernel groups

Coarse categories use kernel names. Full names, assignments, per-capture durations, and trace hashes remain in the companion JSON. Fused kernels stay combined: their full time cannot be attributed to quantization alone. Fusion can move work between categories, so category changes need this context.

### prefill_c1

| Category | BF16 ms | FP8 ms | BF16 minus FP8 ms |
| --- | ---: | ---: | ---: |
| Activation / pointwise | 1.372 | 0.000 | 1.372 |
| Attention | 3.414 | 3.414 | -0.000 |
| Fused arithmetic / conversion | 0.009 | 2.436 | -2.427 |
| Matrix calculations | 46.901 | 35.880 | 11.021 |
| Normalization / residual | 0.665 | 0.007 | 0.658 |
| Other | 0.736 | 0.777 | -0.041 |

Largest positive category difference: **Matrix calculations**, 11.021 ms. Net GPU kernel-work saving: 10.583 ms.

### decode_c1

| Category | BF16 ms | FP8 ms | BF16 minus FP8 ms |
| --- | ---: | ---: | ---: |
| Activation / pointwise | 0.297 | 0.000 | 0.297 |
| Attention | 4.310 | 4.782 | -0.473 |
| Fused arithmetic / conversion | 0.019 | 1.778 | -1.759 |
| Matrix calculations | 69.951 | 49.466 | 20.485 |
| Normalization / residual | 0.865 | 0.013 | 0.852 |
| Other | 1.403 | 1.447 | -0.044 |

Largest positive category difference: **Matrix calculations**, 20.485 ms. Net GPU kernel-work saving: 19.358 ms.

### decode_c4

| Category | BF16 ms | FP8 ms | BF16 minus FP8 ms |
| --- | ---: | ---: | ---: |
| Activation / pointwise | 0.336 | 0.000 | 0.336 |
| Attention | 8.435 | 9.954 | -1.519 |
| Fused arithmetic / conversion | 0.021 | 1.765 | -1.744 |
| Matrix calculations | 73.492 | 49.495 | 23.997 |
| Normalization / residual | 0.913 | 0.013 | 0.900 |
| Other | 2.142 | 2.180 | -0.037 |

Largest positive category difference: **Matrix calculations**, 23.997 ms. Net GPU kernel-work saving: 21.932 ms.

## What the evidence supports

The comparisons identify where recorded GPU work changed. Use the tables above to determine whether matrix kernels account for the largest savings at both concurrency settings. Attention, conversion, activation, normalization, and other kernels can offset part of a matrix-kernel improvement.

Day 5 remains the source for unprofiled serving speed and memory measurements. Day 6 uses a fixed synthetic token-ID prompt, not the exact Day 5 prompt set. Both models receive the same Day 6 prompt and output-token budget.

GPU kernel-time sums do not measure the whole critical path: kernels can overlap, and CPU scheduling, frontend/network work, and idle gaps are outside these sums. No hardware-counter evidence of memory bandwidth saturation or Tensor Core utilization was collected. Profiling overhead and GPU clock state can influence durations. Three repetitions summarize observed variation; no statistical significance claim is made.

## Protocol

- Original BF16 Qwen3-4B-Instruct-2507 versus fixed ModelOpt FP8 Random32 seed 42.
- Input/output: 1,024 / 128 tokens per request, temperature 0, seed 42, EOS ignored.
- Prefill C1: delay 0, maximum 1 engine step.
- Decode C1: delay 8, maximum 8 engine steps.
- Decode C4: delay 16, maximum 8 engine steps; annotations must show four generating requests and zero context requests. Eight steps serve four requests together; the GPU-work total is not a per-request latency.
- PyTorch Profiler: worker CPU/CUDA activity, stacks/shapes/memory tracking disabled, frontend capture disabled; Day 5 compilation/CUDA-graph behavior retained.
- BF16 KV cache, 4 GiB cache budget, FLASH_ATTN, prefix caching disabled, context limit 4,096, maximum four sequences and 4,096 batched tokens.
- GPU UUID: `GPU-da0e682c-68a8-c840-d8a7-81e529a186b6`.
- Recorded serving packages: `torch==2.13.0+cu132`, `transformers==5.19.0`, `vllm==0.32.0`.

## Selected sessions

- BF16_c1: `outputs/profiling/day6/BF16_20261025T214526/captures/20261025T214620632648Z`
- BF16_c4: `outputs/profiling/day6/BF16_20261025T214526/captures_c4/20261025T220144935131Z`
- Random32_c1: `outputs/profiling/day6/Random32_20261025T213935/captures/20261025T214323961921Z`
- Random32_c4: `outputs/profiling/day6/Random32_20261025T220436/captures_c4/20261025T220526534128Z`

## Reproduce the analysis

```bash
cd /root/autodl-tmp/qwen-inference-lab
python scripts/summarize_day6_profiles.py
```

The script checks frozen protocol/request/source identities, warm-up and capture response counts, raw trace hashes, matching phase labels, and kernel totals before writing this report. It performs CPU-only analysis and preserves original evidence.
