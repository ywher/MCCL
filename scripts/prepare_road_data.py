#!/usr/bin/env python3
"""Reuse the exact controlled UniMatchV2 splits and SSDA trainId datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TRANSFERS = {
    'gta2cityscapes': ('gta', 'cityscapes', '1_64', 46, 2975, 19),
    'synthia2cityscapes': ('synthia', 'cityscapes16', '1_64', 46, 2975, 16),
    'cityscapes2acdc': ('cityscapes', 'acdc', '1_64', 25, 1600, 19),
    'cityscapes2muses': ('cityscapes', 'muses', '1_64', 23, 1500, 19),
    'cityscapes2mapillary': ('cityscapes', 'mapillary', '1_128', 140, 18000, 19),
}


def image_id(line):
    return line.split(',', 1)[0].split()[0]


def link(target, source):
    source = source.resolve(strict=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise FileExistsError(f'Refusing to replace {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssda-root', required=True, type=Path)
    p.add_argument('--unimatch-root', required=True, type=Path)
    args = p.parse_args()
    manifest = {'reference_root': str(args.unimatch_root.resolve()), 'files': {}, 'transfers': TRANSFERS}
    for dataset in ('gta', 'synthia', 'cityscapes', 'acdc', 'muses', 'mapillary'):
        link(ROOT / 'data' / dataset, args.ssda_root / 'data' / dataset)
    link(ROOT / 'pretrained/dinov3_base.pth', args.unimatch_root / 'pretrained/dinov3_base.pth')
    for transfer, (source, target, ratio, count, pool, nclass) in TRANSFERS.items():
        for selection in ('random', 'rdkc'):
            base = Path('splits') / target / selection / ratio
            for name in ('labeled.txt', 'unlabeled.txt'):
                src = args.unimatch_root / base / name
                dst = ROOT / base / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and dst.read_bytes() != src.read_bytes():
                    raise FileExistsError(f'Refusing to change existing split: {dst}')
                shutil.copyfile(src, dst)
                manifest['files'][str(dst.relative_to(ROOT))] = hashlib.sha256(src.read_bytes()).hexdigest()
            labeled = (ROOT / base / 'labeled.txt').read_text().splitlines()
            unlabeled = (ROOT / base / 'unlabeled.txt').read_text().splitlines()
            a, b = {image_id(x) for x in labeled}, {image_id(x) for x in unlabeled}
            if len(labeled) != count or len(a) != count or len(b) != len(unlabeled) or a & b or len(a | b) != pool:
                raise ValueError(f'Invalid {transfer} {selection} split')
        val = args.unimatch_root / 'splits' / target / 'val.txt'
        dest = ROOT / 'splits' / target / 'val.txt'
        if dest.exists() and dest.read_bytes() != val.read_bytes():
            raise FileExistsError(dest)
        shutil.copyfile(val, dest)
        manifest['files'][str(dest.relative_to(ROOT))] = hashlib.sha256(val.read_bytes()).hexdigest()
        print(f'{transfer}: {count}/{pool}, {nclass} classes, source={source}; Random/R-DKC ready')
    (ROOT / 'splits/road_preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
