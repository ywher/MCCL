import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from road.data import RoadDataset
from road.engine import mix, train_step
from road.losses import PrototypeBank, alignment, masked_ce, masked_mse, supervised_loss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 1)
        self.classifier = nn.Conv2d(8, 3, 1)

    def forward(self, x, return_features=False):
        feature = self.encoder(x)
        logits = self.classifier(feature)
        return (logits, feature) if return_features else logits

    def classify_context(self, feature, size):
        return self.classifier(feature)


class RoadTests(unittest.TestCase):
    def test_unlabeled_never_reads_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (16, 16)).save(root / 'image.png')
            (root / 'ids.txt').write_text('image.png,DOES_NOT_EXIST.png\n')
            ds = RoadDataset(root, root / 'ids.txt', 'train_u', 19, size=16)
            batch = ds[0]
            self.assertEqual(len(batch), 6)
            self.assertEqual(batch[0].shape, (3, 16, 16))

    def test_16_class_protocol_rejects_19_class_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (16, 16)).save(root / 'image.png')
            Image.fromarray(np.full((16, 16), 18, np.uint8)).save(root / 'mask.png')
            (root / 'ids.txt').write_text('image.png mask.png\n')
            with self.assertRaises(ValueError):
                RoadDataset(root, root / 'ids.txt', 'train_l', 16, size=16)[0]

    def test_all_ignore_losses_are_finite_and_have_backward(self):
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        ignore = torch.full((2, 4, 4), 255)
        labels = torch.zeros_like(ignore)
        probs = logits.detach().softmax(1)
        loss = masked_ce(logits, labels, probs.max(1).values, ignore, 0.95)
        loss += masked_mse(logits, probs, ignore)
        loss += supervised_loss(logits, ignore, {'criterion': {'name': 'CELoss'}})
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_prototype_updates_all_present_classes_and_roundtrips(self):
        bank = PrototypeBank(3, channels=2, threshold=0.9)
        features = torch.tensor([[[[1., 2., 3.]], [[2., 4., 6.]]]])
        probs = F.one_hot(torch.tensor([[[0, 1, 2]]]), 3).permute(0, 3, 1, 2).float()
        ignore = torch.zeros(1, 1, 3, dtype=torch.long)
        bank.update(features, probs, ignore)
        self.assertTrue(bank.initialized.all())
        strong = features.clone().requires_grad_(True)
        loss = bank(features, strong, probs, ignore)
        loss.backward()
        self.assertTrue(torch.isfinite(strong.grad).all())
        other = PrototypeBank(3, channels=2)
        other.load_state_dict(bank.state_dict())
        self.assertTrue(torch.equal(bank.memory, other.memory))

    def test_cutmix_keeps_probability_and_ignore_coordinates(self):
        box = torch.tensor([[[0, 1], [1, 0]]])
        field = torch.zeros(1, 2, 2, 2)
        result = mix(field, torch.ones_like(field), box)
        self.assertTrue(torch.equal(result[:, 0].long(), box))

    def test_full_step_source_fka_sai_and_disabled_perturbations(self):
        cfg = dict(criterion={'name': 'CELoss'}, conf_thresh=0, feat_nearest=0.01,
                   feat_ali_weight=0.1, total_iters=10, unsupervised_w=0.01,
                   source_loss_weight=1.0, **{'F-Noise': True, 'F-Drop': True})
        for enabled, use_source in ((True, True), (False, True), (False, False)):
            torch.manual_seed(1)
            model = TinyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            bank = PrototypeBank(3, channels=8, threshold=0)
            images = torch.randn(2, 3, 8, 8)
            labels = torch.randint(3, (2, 8, 8))
            expected_target = float(F.cross_entropy(model(images), labels).detach()) / (4 if use_source else 2)
            before = model.classifier.weight.detach().clone()
            cfg['F-Noise'] = cfg['F-Drop'] = enabled
            unlabeled = (images, images + 0.2, images - 0.2, torch.zeros_like(labels),
                         torch.ones_like(labels), torch.zeros_like(labels))
            metrics = train_step(model, bank, optimizer, (images, labels), unlabeled,
                                 (images, labels) if use_source else None, cfg, 5)
            self.assertAlmostEqual(metrics['target_gt'], expected_target, places=6)
            if use_source:
                self.assertAlmostEqual(metrics['source_gt'], expected_target, places=6)
            else:
                self.assertNotIn('source_gt', metrics)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            self.assertFalse(torch.equal(before, model.classifier.weight))


if __name__ == '__main__':
    unittest.main()
