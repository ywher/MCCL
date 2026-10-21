#!/usr/bin/env python3
"""Evaluate a controlled MCCL adapter checkpoint on its target validation set."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mccl_source import evaluate, write_eval
from road.data import RoadDataset
from road.model import MCCLHRDA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--val-id-path')
    p.add_argument('--data-root')
    p.add_argument('--pretrained-path')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    model = MCCLHRDA(cfg['nclass'], args.pretrained_path or cfg['vfm_pretrained']).to(args.device)
    model.load_peft_state_dict(state['model'])
    dataset = RoadDataset(args.data_root or cfg['data_root'],
                          args.val_id_path or f"splits/{cfg['dataset']}/val.txt",
                          'val', cfg['nclass'], pre_resize=cfg['pre_resize'])
    miou, ious = evaluate(model, DataLoader(dataset, batch_size=1), cfg['nclass'], args.device)
    write_eval(Path(args.output), miou, ious)
    print(f'iteration={state["iteration"]}, mIoU={miou:.4f}, output={args.output}')


if __name__ == '__main__':
    main()
