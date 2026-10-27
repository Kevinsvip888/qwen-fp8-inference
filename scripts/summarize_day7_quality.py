#!/usr/bin/env python3
"""Compare the ten frozen full-test runs using saved, audited results only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PLAN = '1bd1f014f7409f32e93347cbd25e39eca79a782a614acd0234a5ed7c003e4558'
PREP = '1ddf3a16aad32e1273d90b2e0ba7dd0d84f9e0edd281051c396fc9ca405faa95'
TASKS = {'gsm8k': ('day7_gsm8k_test', 1319, 'strict-match', 'exact_match'),
         'arc_challenge': ('day7_arc_challenge_test', 1172, 'none', 'acc_norm')}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bootstrap(values, repetitions, seed):
    """Resample question units; each value can average several fixed pools."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws.extend(values[indices].mean(axis=1))
    return [float(x * 100) for x in np.percentile(draws, [2.5, 97.5])]


def paired(first, second, repetitions, seed):
    diff = first - second
    gains, losses = int((diff == 1).sum()), int((diff == -1).sum())
    discordant = gains + losses
    # Exact two-sided binomial McNemar test, conditional on discordant pairs.
    p = min(1., 2 * sum(math.comb(discordant, k)
                       for k in range(min(gains, losses) + 1)) / 2**discordant)
    return {'gains': gains, 'losses': losses,
            'both_correct': int(((first == 1) & (second == 1)).sum()),
            'both_incorrect': int(((first == 0) & (second == 0)).sum()),
            'delta_pp': float(diff.mean() * 100),
            'question_bootstrap_95_interval_pp': bootstrap(diff, repetitions, seed),
            'exact_two_sided_mcnemar_p': p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/qwen-inference-lab'))
    parser.add_argument('--bootstrap-repetitions', type=int, default=20000)
    args = parser.parse_args()
    require(args.bootstrap_repetitions >= 1000, 'Use at least 1000 bootstrap repetitions')
    root = args.root.resolve()
    data = root / 'data/evaluation/day7_quality_v1'
    output = root / 'outputs/evaluation/day7_quality_v1/final'
    require(sha(data / 'final_plan.json') == PLAN, 'Frozen final plan changed')
    require(sha(data / 'final/preparation_manifest.json') == PREP, 'Frozen preparation changed')
    plan = read(data / 'final_plan.json')
    runtime_sha = sha(output / 'runtime_protocol.json')
    names = [m['name'] for m in plan['models']]
    expected_names = ['BF16'] + [f'{m}-seed{s}' for s in (42, 43, 44)
                               for m in ('Random32', 'Selected32', 'Random128')]
    require(names == expected_names, 'Expected the frozen ten-model scope')
    scores, rows, identities, receipts = {}, [], {}, {}
    expected_counts = {v[0]: v[1] for v in TASKS.values()}
    for name in names:
        matches = list((output / name).glob('*/final_summary.json'))
        require(len(matches) == 1, f'{name}: expected exactly one completed run')
        path = matches[0]
        run, summary = read(path.parent / 'run_manifest.json'), read(path)
        require(run['model'] == name and run['final_plan_sha256'] == PLAN
                and run['preparation_sha256'] == PREP
                and run['runtime_protocol_sha256'] == runtime_sha, f'{name}: run identity differs')
        require(summary['scope'] == 'full official tests'
                and summary['questions_per_task'] == expected_counts
                and summary['http_requests_verified'] == 6006, f'{name}: incomplete audit')
        hashes = summary['raw_response_and_result_sha256']
        require(bool(hashes), f'{name}: missing audit file hashes')
        for relative, digest in hashes.items():
            require(sha(path.parent / relative) == digest, f'{name}: saved file changed: {relative}')
        row = {'model': name, 'math_length_finishes': summary['math_length_finishes'],
               'strict_extraction_failures': summary['strict_extraction_failures']}
        scores[name] = {}
        for short, (task, count, primary_filter, metric) in TASKS.items():
            files = list((path.parent / 'harness').rglob(f'samples_{task}_*.jsonl'))
            require(len(files) == 1, f'{name}/{task}: expected one samples file')
            samples = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
            groups = {}
            for sample in samples:
                filt = sample['filter']
                group = groups.setdefault(filt, {})
                doc = sample['doc_id']
                require(type(doc) is int and doc not in group, f'{name}/{task}: duplicate or invalid ID')
                group[doc] = sample
            filters = ('strict-match', 'flexible-extract') if short == 'gsm8k' else ('none',)
            require(set(groups) == set(filters), f'{name}/{task}: filters differ')
            for filt in filters:
                group = groups[filt]
                require(set(group) == set(range(count)), f'{name}/{task}: missing question IDs')
                metric_names = ('exact_match',) if short == 'gsm8k' else ('acc', 'acc_norm')
                for met in metric_names:
                    values = [group[i][met] for i in range(count)]
                    require(all(v in (0, 1) for v in values), 'Expected binary per-question scores')
                    require(math.isclose(sum(values)/count, summary['metrics'][task][f'{met},{filt}'],
                                         abs_tol=1e-10), f'{name}: score totals differ')
                identity = [(group[i]['doc_hash'], group[i]['prompt_hash'], group[i]['target_hash'])
                            for i in range(count)]
                key = (short, filt)
                if key not in identities:
                    identities[key] = identity
                require(identity == identities[key], f'{name}/{task}: paired questions/prompts differ')
            group = groups[primary_filter]
            values = np.array([group[i][metric] for i in range(count)], dtype=float)
            scores[name][short] = values
            row[short + '_correct'] = int(values.sum())
            row[short + '_primary_percent'] = float(values.mean()*100)
            secondary_key = 'exact_match,flexible-extract' if short == 'gsm8k' else 'acc,none'
            row[short + '_secondary_percent'] = 100 * summary['metrics'][task][secondary_key]
        require(row['gsm8k_correct'] == summary['correct_math']
                and row['arc_challenge_correct'] == summary['correct_science_acc_norm'],
                f'{name}: summary count mismatch')
        rows.append(row)
        receipts[name] = {'summary_path': str(path.relative_to(root)), 'summary_sha256': sha(path)}
        print(f'Verified {name}: 1319 math + 1172 science questions', flush=True)
    comparisons, conditional = {}, {}
    pairs = [('Random32-seed42', 'BF16')]
    for seed in (42, 43, 44):
        pairs.extend([(f'Selected32-seed{seed}', f'Random32-seed{seed}'),
                      (f'Random128-seed{seed}', f'Random32-seed{seed}')])
    for first, second in pairs:
        label = f'{first} - {second}'
        comparisons[label] = {task: paired(scores[first][task], scores[second][task],
                                             args.bootstrap_repetitions, 20261027)
                              for task in TASKS}
    for method in ('Selected32', 'Random128'):
        label = method + ' - Random32'
        conditional[label] = {}
        for task in TASKS:
            diffs = np.array([scores[f'{method}-seed{s}'][task] - scores[f'Random32-seed{s}'][task]
                              for s in (42, 43, 44)])
            conditional[label][task] = {
                'mean_delta_pp': float(diffs.mean()*100),
                'question_bootstrap_95_interval_pp': bootstrap(diffs.mean(axis=0),
                                                              args.bootstrap_repetitions, 20261027),
                'scope': 'Conditional on the three observed calibration pools; questions resampled jointly across seeds'}
    result = {'final_plan_sha256': PLAN, 'preparation_sha256': PREP,
              'runtime_protocol_sha256': runtime_sha, 'script_sha256': sha(Path(__file__).resolve()),
              'bootstrap_repetitions': args.bootstrap_repetitions, 'bootstrap_seed': 20261027,
              'model_results': rows, 'paired_comparisons': comparisons,
              'across_pool_comparisons': conditional, 'input_audit_receipts': receipts,
              'notes': ['No inference performed and no scores changed.',
                        'Primary scores: GSM8K strict-match; ARC-Challenge acc_norm.',
                        'Intervals and unadjusted p-values are reported separately per task.',
                        'A small or nonsignificant difference does not establish equivalence.',
                        'Length finishes may overlap correct answers; strict extraction uses the frozen marker rule.']}
    lines = ['# Day 7 full-test quality results', '',
             'Frozen tests: 1,319 GSM8K questions and 1,172 ARC-Challenge questions per checkpoint.', '',
             '| Model | Math strict % | Science acc_norm % | Math length finishes | Missing strict marker |',
             '|---|---:|---:|---:|---:|']
    print('\nModel                    Math %  Science %  Length finishes')
    for row in rows:
        name, math_pct, arc_pct = row['model'], row['gsm8k_primary_percent'], row['arc_challenge_primary_percent']
        print(f'{name:24} {math_pct:7.2f} {arc_pct:10.2f} {row["math_length_finishes"]:16}')
        lines.append(f'| {name} | {math_pct:.2f} | {arc_pct:.2f} | {row["math_length_finishes"]} | {row["strict_extraction_failures"]} |')
    lines.extend(['', 'Paired differences are first model minus second. Positive means more correct answers.', '',
                  '| Comparison | Task | Gains | Losses | Difference pp | Bootstrap 95% interval pp | Exact p |',
                  '|---|---|---:|---:|---:|---|---:|'])
    for label, tasks in comparisons.items():
        for task, item in tasks.items():
            lo, hi = item['question_bootstrap_95_interval_pp']
            lines.append(f'| {label} | {task} | {item["gains"]} | {item["losses"]} | {item["delta_pp"]:.4f} | [{lo:.4f}, {hi:.4f}] | {item["exact_two_sided_mcnemar_p"]:.6f} |')
            print(f'{label}; {task}: {item["delta_pp"]:+.4f} pp; 95% interval [{lo:.4f}, {hi:.4f}]')
    lines.extend(['', '## Three-pool descriptive comparisons', ''])
    for label, tasks in conditional.items():
        for task, item in tasks.items():
            lo, hi = item['question_bootstrap_95_interval_pp']
            text = f'{label}; {task}: mean {item["mean_delta_pp"]:+.4f} pp; interval [{lo:.4f}, {hi:.4f}] pp.'
            lines.append('- ' + text)
            print(text)
    lines.extend(['', 'These intervals condition on the three observed calibration pools. They do not measure uncertainty over all possible calibration seeds.', '',
                  'Strict math scoring can accept an answer before a repetitive or contradictory tail. Length finishes and missing markers are separate diagnostics, not extra deductions from the frozen accuracy score.', '',
                  'Use the saved JSON for full-precision values and secondary scores. No parser, prompt, or generation budget was changed. A nonsignificant result does not establish equivalence.'])
    for filename, content in [('day7_summary.json', json.dumps(result, indent=2)+'\n'),
                              ('day7_report.md', '\n'.join(lines)+'\n')]:
        target = output / filename
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(target)
        print('Saved:', target)
    print('PASS: ten full-test runs, saved file hashes, paired question identities, and score totals verified.')


if __name__ == '__main__':
    main()
