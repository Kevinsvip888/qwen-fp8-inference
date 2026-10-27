#!/usr/bin/env python3
"""Check published source, metadata, and report integrity without CUDA or network.

This does not replay the raw-response or GPU-trace audits behind the reports.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    for inventory in ('provenance/public_file_sha256.json', 'provenance/script_sha256.json'):
        mapping = read(root / inventory)
        for relative, digest in mapping.items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root), 'Inventory path escapes repository')
            require(path.is_file() and sha(path) == digest, f'Missing or changed published file: {relative}')
        print(f'Verified {len(mapping)} files against {inventory}')
    original = root / 'provenance/original'
    folder = original / 'data/evaluation/day7_quality_v1'
    plan_path, prep_path = folder / 'final_plan.json', folder / 'final/preparation_manifest.json'
    plan, prep = read(plan_path), read(prep_path)
    runtime_path = original / 'outputs/evaluation/day7_quality_v1/final/runtime_protocol.json'
    runtime = read(runtime_path)
    require(sha(plan_path) == '1bd1f014f7409f32e93347cbd25e39eca79a782a614acd0234a5ed7c003e4558', 'Frozen final plan differs')
    require(sha(prep_path) == '1ddf3a16aad32e1273d90b2e0ba7dd0d84f9e0edd281051c396fc9ca405faa95', 'Frozen preparation differs')
    require(runtime['final_plan_sha256'] == prep['final_plan_sha256'] == sha(plan_path), 'Plan links differ')
    require(runtime['preparation_sha256'] == sha(prep_path), 'Runtime/preparation link differs')
    require(plan['development_protocol_sha256'] == sha(folder / 'development_protocol.json'), 'Development protocol differs')
    require(plan['source_manifest_sha256'] == sha(original / 'data/evaluation/day7_sources/source_revisions.json'), 'Evaluation sources differ')
    names = ['BF16'] + [f'{m}-seed{s}' for s in (42, 43, 44)
                        for m in ('Random32', 'Selected32', 'Random128')]
    require([m['name'] for m in plan['models']] == names, 'Model scope differs')
    protocol_sha = sha(original / 'data/calibration/fp8_protocol.json')
    for model in plan['models'][1:]:
        path = original / model['path'] / 'calibration_manifest.json'
        manifest = read(path)
        require(sha(path) == model['calibration_manifest_sha256'], f'Checkpoint identity differs: {model["name"]}')
        require(manifest['identity'] == model['identity'], 'Checkpoint identity fields differ')
        require(manifest['identity']['protocol_sha256'] == protocol_sha, 'FP8 recipe differs')
    split = read(original / 'data/splits/gsm8k_train_split.json')
    groups = split['partitions']
    require({k: len(v) for k, v in groups.items()} ==
            {'development': 256, 'fewshot_reserve': 16, 'calibration': 7201}, 'Split sizes differ')
    indices = [x['row'] for group in groups.values() for x in group]
    require(len(indices) == 7473 and set(indices) == set(range(7473)), 'Training partitions overlap or have omissions')
    reserve_ids = {f'gsm8k-train-{x["row"]:05d}' for x in groups['fewshot_reserve']}
    require(len(plan['fewshot_ids']) == 5 and len(set(plan['fewshot_ids'])) == 5
            and set(plan['fewshot_ids']) <= reserve_ids, 'Few-shot IDs escape the reserved partition')
    for seed in (42, 43, 44):
        pool_dir = original / f'data/calibration/seed_{seed}'
        pool = read(pool_dir / 'manifest.json')
        subsets = read(pool_dir / 'random_subsets_manifest.json')['subsets']
        small = subsets['Random32']['sequence_ids']
        large = subsets['Random128']['sequence_ids']
        require(len(small) == len(set(small)) == 32 and len(large) == len(set(large)) == 128
                and set(small) <= set(large), 'Random subsets are not nested or have wrong sizes')
        require(pool['seed'] == seed and pool['total_sequences'] == 512
                and pool['sequence_length'] == 512, 'Pool dimensions differ')
    final = read(root / 'results/day7_summary.json')
    require(final['final_plan_sha256'] == sha(plan_path) and final['preparation_sha256'] == sha(prep_path), 'Final report pins differ')
    require(final['runtime_protocol_sha256'] == sha(runtime_path), 'Final report runtime differs')
    rows = final['model_results']
    require([x['model'] for x in rows] == names, 'Final report model coverage differs')
    for row in rows:
        for task, count in (('gsm8k', 1319), ('arc_challenge', 1172)):
            require(math.isclose(row[task + '_primary_percent'],
                                100 * row[task + '_correct'] / count, abs_tol=1e-10), 'Accuracy/count mismatch')
        require(0 <= row['math_length_finishes'] <= 1319 and
                0 <= row['strict_extraction_failures'] <= 1319, 'Invalid diagnostics')
    benchmark = read(root / 'results/day5_summary.json')
    profile = read(root / 'results/day6_summary.json')
    require(benchmark['validated_runs'] == 36 and benchmark['measured_requests'] == 7200, 'Benchmark run totals differ')
    require(profile['validated_capture_windows'] == 18, 'Profiling window totals differ')
    print('PASS: public file integrity, nine FP8 checkpoint identities, training splits, and report totals verified.')
    print('Scope: published summaries/configurations only; raw responses, weights, and traces are not included.')


if __name__ == '__main__':
    main()
