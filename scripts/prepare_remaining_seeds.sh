#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/qwen-inference-lab
source .venv-quant/bin/activate
export HF_HUB_OFFLINE=1

run_stage() {
    local log_name="$1"
    shift
    "$@" 2>&1 | tee "logs/$log_name"
}

for pool_seed in 43 44; do
    echo "Preparing calibration pool seed $pool_seed"

    run_stage "build_calibration_pool_seed${pool_seed}.log" \
        python -u scripts/build_calibration_pool.py \
        --seed "$pool_seed"

    run_stage "audit_and_sample_pool_seed${pool_seed}.log" \
        python -u scripts/audit_and_sample_pool.py \
        --seed "$pool_seed"

    run_stage "activation_features_seed${pool_seed}.log" \
        python -u scripts/extract_activation_features.py \
        --seed "$pool_seed"

    run_stage "select_calibration_sequences_seed${pool_seed}.log" \
        python -u scripts/select_calibration_sequences.py \
        --seed "$pool_seed"

    for method in Random32 Selected32 Random128; do
        run_stage "quantize_${method}_seed${pool_seed}.log" \
            python -u scripts/quantize_fp8_experiment.py \
            --method "$method" --seed "$pool_seed"
    done

    run_stage "compare_calibration_ranges_seed${pool_seed}.log" \
        python -u scripts/compare_calibration_ranges.py \
        --seed "$pool_seed"

    echo "PASS: seed $pool_seed preparation and three FP8 exports complete."
done
