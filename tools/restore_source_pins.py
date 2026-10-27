#!/usr/bin/env python3
"""Restore only source revisions and the training split; never overwrite different files."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    paths = ('logs/model_revision.json', 'data/source_revisions.json',
             'data/splits/gsm8k_train_split.json',
             'data/evaluation/day7_sources/source_revisions.json')
    pairs = [(root / 'provenance/original' / p, root / p) for p in paths]
    for source, target in pairs:
        if not source.is_file():
            raise RuntimeError(f'Missing archived pin: {source}')
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f'Existing file differs; left untouched: {target}')
    for source, target in pairs:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        print('Verified source pin:', target)
    print('No weights, checkpoint manifests, or evaluation runtime protocols restored.')


if __name__ == '__main__':
    main()
