#!/usr/bin/env python3
"""Validate split isolation, file existence and sample trainId ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from road.data import parse_id, read_ids
from scripts.prepare_road_data import TRANSFERS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', default='validation/data_check.json')
    args = p.parse_args()
    results = []
    for transfer, (source, target, ratio, count, pool, classes) in TRANSFERS.items():
        cfg = yaml.safe_load(Path(f'configs/road_hrda_source/{transfer}.yaml').read_text())
        root = Path(cfg['data_root'])
        source_root = Path('data') / source
        source_ids = read_ids(source_root / 'train.txt')
        val_ids = read_ids(Path('splits') / target / 'val.txt')
        val_images = {parse_id(x)[0] for x in val_ids}
        for selection in ('random', 'rdkc'):
            base = Path('splits') / target / selection / ratio
            labeled, unlabeled = [read_ids(base / f'{x}.txt') for x in ('labeled', 'unlabeled')]
            a, b = [{parse_id(x)[0] for x in ids} for ids in (labeled, unlabeled)]
            assert len(a) == count and len(a | b) == pool and not a & b
            assert not (a | b) & val_images
            # Only images are required for the unlabeled stream.
            for ids, data_root, labels_required in [(labeled, root, True), (unlabeled, root, False), (val_ids, root, True), (source_ids, source_root, True)]:
                for line in ids:
                    image, mask = parse_id(line)
                    if not (data_root / image).is_file():
                        raise FileNotFoundError(data_root / image)
                    if labels_required and (mask is None or not (data_root / mask).is_file()):
                        raise FileNotFoundError(f'{data_root}/{mask}')
                if labels_required:
                    for line in ids[:3]:
                        _, mask = parse_id(line)
                        with Image.open(data_root / mask) as f:
                            values = np.array(f)
                        assert values.ndim == 2 and np.all((values < classes) | (values == 255)), (transfer, mask)
            results.append({'transfer': transfer, 'selection': selection, 'labeled': len(a), 'unlabeled': len(b),
                            'classes': classes, 'source_images': len(source_ids), 'val_images': len(val_ids),
                            'files_verified': True, 'train_val_disjoint': True,
                            'labeled_sha256': hashlib.sha256((base/'labeled.txt').read_bytes()).hexdigest()})
            print(f'{transfer}/{selection}: files and split isolation verified')
    output = Path(args.result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
