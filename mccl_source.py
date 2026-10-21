#!/usr/bin/env python3
"""Controlled five-transfer MCCL Semi(+Source), DINOv3-B/ReIN/HRDA."""
import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from road.data import BatchStream, RoadDataset
from road.engine import train_step
from road.losses import PrototypeBank
from road.model import MCCLHRDA


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--labeled-id-path', required=True)
    p.add_argument('--unlabeled-id-path', required=True)
    p.add_argument('--val-id-path')
    p.add_argument('--source-data-root')
    p.add_argument('--source-id-path')
    p.add_argument('--source-batch-size', type=int, default=2)
    p.add_argument('--source-loss-weight', type=float, default=1.0)
    p.add_argument('--save-path', required=True)
    p.add_argument('--pretrained-path')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-iters', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--device', default='cuda')
    return p


def learning_rate(step, cfg):
    scale = (1 - step / cfg['total_iters']) ** cfg.get('lr_power', 1.0)
    warmup = cfg.get('warmup_iters', 0)
    if step < warmup:
        scale *= cfg.get('warmup_ratio', 1e-6) + (1 - cfg.get('warmup_ratio', 1e-6)) * step / warmup
    return cfg['lr'] * scale


def optimizer_for(model, cfg):
    return torch.optim.AdamW(model.optimizer_param_groups(cfg['lr'], cfg['lr_multi'], cfg['weight_decay']),
                             lr=cfg['lr'], eps=cfg.get('optimizer_eps', 1e-8))


@torch.no_grad()
def evaluate(model, loader, nclass, device):
    model.eval()
    histogram = torch.zeros(nclass, nclass, dtype=torch.float64, device=device)
    for images, labels, _ in loader:
        labels = labels.to(device)
        pred = model(images.to(device)).argmax(1)
        valid = labels != 255
        encoded = labels[valid] * nclass + pred[valid]
        histogram += torch.bincount(encoded, minlength=nclass ** 2).reshape(nclass, nclass)
    union = histogram.sum(0) + histogram.sum(1) - histogram.diag()
    iou = histogram.diag() / union.clamp_min(1) * 100
    # Match SSDA/UniMatch: average every protocol class, including absent ones.
    return float(iou.mean()), iou.cpu().tolist()


def write_eval(path, miou, ious):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mIoU'] + [f'class_{i}' for i in range(len(ious))])
        writer.writerow([f'{x:.4f}' for x in [miou] + ious])


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    args = parser().parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This controlled protocol uses exactly one GPU; multi-rank training is not supported')
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_iters is not None:
        cfg['total_iters'] = args.max_iters
    if args.workers is not None:
        cfg['workers'] = args.workers
    if args.pretrained_path:
        cfg['vfm_pretrained'] = args.pretrained_path
    if cfg['total_iters'] <= 0 or cfg['batch_size'] < 2:
        raise ValueError('Positive total_iters and target batch >= 2 required')
    if bool(args.source_data_root) != bool(args.source_id_path):
        raise ValueError('Source root and source split must be supplied together')
    if args.source_loss_weight < 0 or args.source_batch_size <= 0:
        raise ValueError('Invalid source weight/batch size')
    if cfg['workers'] < 0 or cfg['eval_interval_iters'] <= 0:
        raise ValueError('Invalid worker count or evaluation interval')
    cfg['source_loss_weight'] = args.source_loss_weight
    split_paths = {'labeled': args.labeled_id_path, 'unlabeled': args.unlabeled_id_path,
                   'source': args.source_id_path,
                   'val': args.val_id_path or f"splits/{cfg['dataset']}/val.txt"}
    split_sha256 = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in split_paths.items() if v}
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if (save_path / 'latest.pth').exists() and not args.resume:
        raise FileExistsError('Output already has a checkpoint; pass --resume or choose a new save-path')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(save_path / 'train.log')])
    for seed in (random.seed, np.random.seed, torch.manual_seed):
        seed(cfg.get('seed', 0))
    device = torch.device(args.device)
    unlabeled = RoadDataset(cfg['data_root'], args.unlabeled_id_path, 'train_u', cfg['nclass'], cfg['crop_size'], cfg['pre_resize'])
    labeled = RoadDataset(cfg['data_root'], args.labeled_id_path, 'train_l', cfg['nclass'], cfg['crop_size'], cfg['pre_resize'], len(unlabeled))
    val = RoadDataset(cfg['data_root'], args.val_id_path or f"splits/{cfg['dataset']}/val.txt", 'val', cfg['nclass'], pre_resize=cfg['pre_resize'])
    def stream(dataset, batch):
        return BatchStream(DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=cfg['workers'],
                                      drop_last=True, pin_memory=device.type == 'cuda'))
    streams = [stream(labeled, cfg['batch_size']), stream(unlabeled, cfg['batch_size'])]
    source = None
    if args.source_data_root:
        source = stream(RoadDataset(args.source_data_root, args.source_id_path, 'train_l', cfg['nclass'],
                                   cfg['crop_size'], cfg['source_pre_resize']), args.source_batch_size)
    valloader = DataLoader(val, batch_size=1, num_workers=min(cfg['workers'], 1))
    model = MCCLHRDA(cfg['nclass'], cfg['vfm_pretrained'], save_path).to(device)
    model.freeze_backbone_check()
    optimizer = optimizer_for(model, cfg)
    bank = PrototypeBank(cfg['nclass'], momentum=cfg['prototype_momentum'], threshold=cfg['prototype_conf_thresh']).to(device)
    start, best, best_iteration = 0, -1.0, 0
    if args.resume:
        state = torch.load(save_path / 'latest.pth', map_location='cpu', weights_only=False)
        if state['config'] != cfg or state['input_splits'] != {k: getattr(args, k) for k in ('labeled_id_path', 'unlabeled_id_path', 'source_id_path', 'source_data_root', 'val_id_path')}:
            raise ValueError('Resume requires the same config and input splits')
        if state['split_sha256'] != split_sha256:
            raise ValueError('Split file contents changed since the checkpoint')
        model.load_peft_state_dict(state['model'])
        bank.load_state_dict(state['prototype'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['iteration'], state['best_miou']
        best_iteration = state['best_iteration']
        logging.info('resumed: iteration=%d best=%.4f', start, best)
    config_record = {'config': cfg, 'args': vars(args), 'split_sha256': split_sha256,
                     'protocol': 'MCCL Semi+Source' if source else 'MCCL Semi'}
    (save_path / 'config.json').write_text(json.dumps(config_record, indent=2) + '\n')
    train_start = time.monotonic()
    for step in range(start, cfg['total_iters']):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_scale']
        metrics = train_step(model, bank, optimizer, streams[0].next(), streams[1].next(),
                             source.next() if source else None, cfg, step)
        iteration = step + 1
        if iteration % 50 == 0 or iteration == 1:
            eta = (time.monotonic() - train_start) / (iteration - start) * (cfg['total_iters'] - iteration)
            logging.info('iter: %d/%d, lr: %.8f, eta: %s, losses: %s', iteration, cfg['total_iters'], lr,
                         str(datetime.timedelta(seconds=int(eta))), json.dumps(metrics))
            metrics_path = save_path / 'training_metrics.csv'
            exists = metrics_path.exists()
            with metrics_path.open('a', newline='') as f:
                record = {'iteration': iteration, 'lr': lr, **metrics}
                writer = csv.DictWriter(f, fieldnames=list(record))
                if not exists:
                    writer.writeheader()
                writer.writerow(record)
        if iteration % cfg['eval_interval_iters'] == 0 or iteration == cfg['total_iters']:
            miou, ious = evaluate(model, valloader, cfg['nclass'], device)
            improved = miou > best
            best = max(best, miou)
            if improved:
                best_iteration = iteration
            write_eval(save_path / 'eval/model_final_ious.csv', miou, ious)
            history = save_path / 'eval/validation_history.csv'
            exists = history.exists()
            with history.open('a', newline='') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(['iteration', 'mIoU', 'best_miou', 'best_iteration'])
                writer.writerow([iteration, miou, best, best_iteration])
            state = {'model': model.peft_state_dict(), 'prototype': bank.state_dict(),
                     'optimizer': optimizer.state_dict(), 'iteration': iteration, 'best_miou': best,
                     'best_iteration': best_iteration, 'config': cfg, 'split_sha256': split_sha256,
                     'input_splits': {k: getattr(args, k) for k in ('labeled_id_path', 'unlabeled_id_path', 'source_id_path', 'source_data_root', 'val_id_path')}}
            atomic_save(state, save_path / 'latest.pth')
            if improved:
                atomic_save(state, save_path / 'model_best.pth')
                write_eval(save_path / 'eval/model_best_ious.csv', miou, ious)
            if iteration == cfg['total_iters']:
                atomic_save(state, save_path / 'model_final.pth')
            logging.info('validation: iteration=%d mIoU=%.4f best=%.4f', iteration, miou, best)
    logging.info('completed: %d/%d, best=%.4f', cfg['total_iters'], cfg['total_iters'], best)


if __name__ == '__main__':
    main()
