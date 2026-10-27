# Day 7 full-test quality results

Frozen tests: 1,319 GSM8K questions and 1,172 ARC-Challenge questions per checkpoint.

| Model | Math strict % | Science acc_norm % | Math length finishes | Missing strict marker |
|---|---:|---:|---:|---:|
| BF16 | 89.61 | 58.70 | 73 | 5 |
| Random32-seed42 | 89.31 | 59.39 | 78 | 2 |
| Selected32-seed42 | 89.16 | 58.79 | 113 | 1 |
| Random128-seed42 | 89.16 | 59.30 | 86 | 5 |
| Random32-seed43 | 89.46 | 58.62 | 72 | 2 |
| Selected32-seed43 | 89.08 | 59.39 | 71 | 1 |
| Random128-seed43 | 89.69 | 59.56 | 108 | 6 |
| Random32-seed44 | 90.67 | 59.30 | 116 | 6 |
| Selected32-seed44 | 89.23 | 59.98 | 98 | 4 |
| Random128-seed44 | 90.22 | 59.64 | 105 | 6 |

Paired differences are first model minus second. Positive means more correct answers.

| Comparison | Task | Gains | Losses | Difference pp | Bootstrap 95% interval pp | Exact p |
|---|---|---:|---:|---:|---|---:|
| Random32-seed42 - BF16 | gsm8k | 34 | 38 | -0.3033 | [-1.5163, 0.9856] | 0.723948 |
| Random32-seed42 - BF16 | arc_challenge | 22 | 14 | 0.6826 | [-0.3413, 1.7065] | 0.242985 |
| Selected32-seed42 - Random32-seed42 | gsm8k | 31 | 33 | -0.1516 | [-1.3647, 1.0614] | 0.900653 |
| Selected32-seed42 - Random32-seed42 | arc_challenge | 11 | 18 | -0.5973 | [-1.5358, 0.2560] | 0.264931 |
| Random128-seed42 - Random32-seed42 | gsm8k | 31 | 33 | -0.1516 | [-1.3647, 1.0614] | 0.900653 |
| Random128-seed42 - Random32-seed42 | arc_challenge | 9 | 10 | -0.0853 | [-0.8532, 0.6826] | 1.000000 |
| Selected32-seed43 - Random32-seed43 | gsm8k | 35 | 40 | -0.3791 | [-1.6679, 0.9098] | 0.644464 |
| Selected32-seed43 - Random32-seed43 | arc_challenge | 20 | 11 | 0.7679 | [-0.1706, 1.7065] | 0.149613 |
| Random128-seed43 - Random32-seed43 | gsm8k | 33 | 30 | 0.2274 | [-0.9856, 1.4405] | 0.801306 |
| Random128-seed43 - Random32-seed43 | arc_challenge | 21 | 10 | 0.9386 | [0.0000, 1.8771] | 0.070756 |
| Selected32-seed44 - Random32-seed44 | gsm8k | 27 | 46 | -1.4405 | [-2.7293, -0.1516] | 0.034416 |
| Selected32-seed44 - Random32-seed44 | arc_challenge | 17 | 9 | 0.6826 | [-0.1706, 1.5358] | 0.168638 |
| Random128-seed44 - Random32-seed44 | gsm8k | 29 | 35 | -0.4549 | [-1.6679, 0.7582] | 0.532309 |
| Random128-seed44 - Random32-seed44 | arc_challenge | 16 | 12 | 0.3413 | [-0.5119, 1.1945] | 0.571588 |

## Three-pool descriptive comparisons

- Selected32 - Random32; gsm8k: mean -0.6571 pp; interval [-1.4152, 0.1011] pp.
- Selected32 - Random32; arc_challenge: mean +0.2844 pp; interval [-0.1991, 0.7964] pp.
- Random128 - Random32; gsm8k: mean -0.1264 pp; interval [-0.8845, 0.6065] pp.
- Random128 - Random32; arc_challenge: mean +0.3982 pp; interval [-0.0853, 0.9101] pp.

These intervals condition on the three observed calibration pools. They do not measure uncertainty over all possible calibration seeds.

Strict math scoring can accept an answer before a repetitive or contradictory tail. Length finishes and missing markers are separate diagnostics, not extra deductions from the frozen accuracy score.

Use the saved JSON for full-precision values and secondary scores. No parser, prompt, or generation budget was changed. A nonsignificant result does not establish equivalence.
