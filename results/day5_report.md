# Day 5 — Validated serving benchmark report

All 36 saved runs passed validation: 7,200 measured requests and 921,600 output tokens. Initial sanity requests, warm-up requests, and pilots are excluded from these totals.

## Primary result

Preselected workload: 1,024 input tokens, 128 output tokens, concurrency 4; three repetitions of 200 measured requests per model.

FP8 throughput speedup: **1.3429×**. Sampled memory maximum decreased by **3.25 GiB**.

| Metric | BF16 mean ± sample SD | FP8 mean ± sample SD |
| --- | ---: | ---: |
| Output throughput (tok/s) | 323.61 ± 0.29 | 434.57 ± 0.53 |
| Mean TTFT (ms) | 183.21 ± 2.01 | 141.15 ± 0.78 |
| Mean TPOT (ms) | 11.01 ± 0.01 | 8.16 ± 0.02 |
| Mean request latency (ms) | 1581.63 ± 1.45 | 1177.77 ± 1.45 |
| Sampled GPU maximum (MiB) | 12913.00 ± 0.00 | 9585.00 ± 0.00 |

## Throughput matrix

Values are arithmetic means of three run-level rates; SD is the sample SD across those runs.

| Input tokens | Concurrency | BF16 tok/s ± SD | FP8 tok/s ± SD | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 103.49 ± 0.05 | 138.72 ± 0.05 | 1.3404× |
| 256 | 4 | 379.94 ± 0.04 | 525.05 ± 0.22 | 1.3819× |
| 1024 | 1 | 98.97 ± 0.09 | 131.53 ± 0.08 | 1.3290× |
| 1024 | 4 | 323.61 ± 0.29 | 434.57 ± 0.53 | 1.3429× |
| 2048 | 1 | 92.76 ± 0.01 | 121.92 ± 0.07 | 1.3144× |
| 2048 | 4 | 271.95 ± 0.04 | 360.63 ± 0.01 | 1.3261× |

## Mean latency matrix

Each entry is BF16 / FP8, in milliseconds, averaged across the three run means.

| Input tokens | Concurrency | TTFT | TPOT | Request latency |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 30.47 / 26.92 | 9.49 / 7.05 | 1235.99 / 921.88 |
| 256 | 4 | 63.95 / 56.60 | 10.10 / 7.23 | 1347.12 / 974.75 |
| 1024 | 1 | 76.17 / 62.34 | 9.58 / 7.17 | 1292.49 / 972.41 |
| 1024 | 4 | 183.21 / 141.15 | 11.01 / 8.16 | 1581.63 / 1177.77 |
| 2048 | 1 | 138.54 / 107.02 | 9.77 / 7.42 | 1379.10 / 1049.02 |
| 2048 | 4 | 350.20 / 249.61 | 12.06 / 9.21 | 1882.01 / 1419.15 |

## Sampled GPU memory

Mean of each run's whole-GPU sampled maximum, in MiB. Sampling interval: one second. The window includes client startup and warm-up, and excludes server startup. Brief spikes between samples can be missed. Values include persistent allocator/workspace state from earlier workloads. The primary workload supplies the headline memory comparison.

| Input tokens | Concurrency | BF16 MiB | FP8 MiB | Difference GiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 12513.00 | 9301.00 | 3.137 |
| 256 | 4 | 12579.00 | 9301.00 | 3.201 |
| 1024 | 1 | 12913.00 | 9301.00 | 3.527 |
| 1024 | 4 | 12913.00 | 9585.00 | 3.250 |
| 2048 | 1 | 12693.00 | 9469.00 | 3.148 |
| 2048 | 4 | 13029.00 | 9773.00 | 3.180 |

## Tail latency

Each entry is BF16 / FP8. These are means of the three native run-level P95 values, not P95 values calculated by pooling all requests.

| Input tokens | Concurrency | Mean run P95 TTFT (ms) | Mean run P95 TPOT (ms) | Mean run P95 request latency (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 34.13 / 30.99 | 9.50 / 7.06 | 1239.65 / 925.60 |
| 256 | 4 | 74.67 / 65.30 | 10.25 / 7.35 | 1351.91 / 979.48 |
| 1024 | 1 | 80.58 / 66.05 | 9.59 / 7.18 | 1296.49 / 975.63 |
| 1024 | 4 | 236.74 / 179.68 | 11.85 / 8.78 | 1589.08 / 1184.88 |
| 2048 | 1 | 143.27 / 111.67 | 9.78 / 7.43 | 1383.32 / 1053.25 |
| 2048 | 4 | 470.17 / 338.36 | 13.70 / 10.32 | 1893.62 / 1426.75 |

## Protocol and interpretation

- Physical GPU UUID: `GPU-da0e682c-68a8-c840-d8a7-81e529a186b6`.
- Recorded serving packages: `torch==2.13.0+cu132`, `transformers==5.19.0`, `vllm==0.32.0`.
- Original Qwen3-4B-Instruct-2507 BF16 versus ModelOpt static FP8 Random32, calibration pool seed 42.
- Both servers: BF16 KV cache, 4 GiB budget, 29,120-token capacity, FLASH_ATTN, context limit 4,096, maximum 4 sequences, batching budget 4,096 tokens, prefix caching disabled.
- Synthetic completions, output length 128, temperature 0, top-p 1, workload and generation seed 42, EOS ignored, 20 warm-up requests per run, client and server on the same machine.
- Primary execution order: FP8-1, BF16-1, BF16-2, FP8-2, FP8-3, BF16-3. Secondary cases were collected in server blocks; run variation is descriptive.
- Native peak concurrent-request statistics count requests touching one-second buckets. Use the configured concurrency limit when interpreting these experiments.
- This is a warm serving comparison on one GPU. Startup/compilation time, production-network latency, other GPUs, and answer quality are separate measurements.
- Three repetitions describe observed variability; no population confidence intervals or significance tests are claimed.
- Keep the Day 4 quality results separate: Selected32 showed no consistent advantage over Random32. This benchmark uses the fixed Random32 seed-42 checkpoint.

## Evidence

The companion JSON retains all 36 run summaries, the frozen protocol, and SHA256 hashes of the result, environment, client log, server log, and GPU samples for every run. The raw evidence remains in the original run directories.

Protocol SHA256: `1c776c059b1f35c1220511ed6fc6bfcee9ebd69bc004bb311369f843bb318994`.
