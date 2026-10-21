#!/usr/bin/env python3
"""One synthetic optimizer step through the real MCCL HRDA adapter."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mccl_source import evaluate, optimizer_for
from road.engine import train_step
from road.losses import PrototypeBank
from road.model import MCCLHRDA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/road_hrda_source/gta2cityscapes.yaml')
    parser.add_argument('--device', required=True)
    parser.add_argument('--result')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(0)
    device = torch.device(args.device)
    # Test-only threshold overrides exercise all branches on random inputs.
    cfg['conf_thresh'] = cfg['prototype_conf_thresh'] = 0.0
    model = MCCLHRDA(cfg['nclass'], cfg['vfm_pretrained']).to(device)
    model.freeze_backbone_check()
    optimizer = optimizer_for(model, cfg)
    bank = PrototypeBank(cfg['nclass'], threshold=0).to(device)
    size, batch = cfg['crop_size'], cfg['batch_size']
    images = torch.randn(batch, 3, size, size, device=device)
    labels = torch.randint(cfg['nclass'], (batch, size, size), device=device)
    ignore = torch.zeros_like(labels)
    box = torch.zeros_like(labels)
    box[:, size // 4:3 * size // 4, size // 4:3 * size // 4] = 1
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    before = next(model.backbone.adapter.parameters()).detach().clone()
    metrics = train_step(model, bank, optimizer, (images, labels),
                         (images, images + .1, images - .1, ignore, box, 1 - box),
                         (images, labels), cfg, cfg['total_iters'] // 2)
    assert not torch.equal(before, next(model.backbone.adapter.parameters()).detach())
    assert model.head.hr_crop_box is None
    model.freeze_backbone_check()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'checkpoint.pth'
        torch.save({'model': model.peft_state_dict(), 'prototype': bank.state_dict()}, path)
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_peft_state_dict(state['model'])
        bank.load_state_dict(state['prototype'])
    with torch.no_grad():
        model.eval()
        prediction = model(images[:1])
        assert prediction.shape == (1, cfg['nclass'], size, size)
        assert torch.isfinite(prediction).all()
    result = {'status': 'passed', 'config': args.config, 'shape': list(prediction.shape),
              'metrics': metrics, 'prototype_initialized_classes': int(bank.initialized.sum()),
              'test_only_confidence_threshold': 0.0,
              'checkpoint_roundtrip': True, 'frozen_encoder_verified': True,
              'peak_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None}
    if args.result:
        Path(args.result).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
