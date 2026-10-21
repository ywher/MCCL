"""MCCL FKA/SAI with class-keyed, checkpointable prototype memory."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def nearest(mask, size):
    return F.interpolate(mask[:, None].float(), size, mode='nearest')[:, 0].long()


def masked_ce(logits, labels, confidence, ignore, threshold):
    valid = ignore != 255
    weights = valid & (confidence >= threshold)
    loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=255)
    return (loss * weights).sum() / valid.sum().clamp_min(1)


def supervised_loss(logits, labels, cfg):
    valid = labels != 255
    losses = F.cross_entropy(logits, labels, ignore_index=255, reduction='none')
    if not valid.any():
        return logits.sum() * 0
    values = losses[valid]
    if cfg['criterion']['name'] == 'OHEM':
        kwargs = cfg['criterion']['kwargs']
        hard = values > -math.log(float(kwargs.get('thresh', 0.7)))
        kept = min(int(kwargs.get('min_kept', 200000)), values.numel())
        return values[hard].mean() if int(hard.sum()) >= kept else values.topk(kept).values.mean()
    if cfg['criterion']['name'] != 'CELoss':
        raise ValueError('Unsupported criterion')
    return values.mean()


def masked_mse(logits, probabilities, ignore):
    valid = (ignore != 255)[:, None]
    errors = (logits.softmax(1) - probabilities.detach()).square()
    return (errors * valid).sum() / (valid.sum() * logits.shape[1]).clamp_min(1)


def alignment(weak, strong, ignore):
    valid = nearest(ignore, strong.shape[-2:]) != 255
    distance = 1 - F.cosine_similarity(weak.detach(), strong, dim=1)
    return (distance * valid).sum() / valid.sum().clamp_min(1)


def intervene(weak, strong, ignore):
    similarity = F.cosine_similarity(weak.detach(), strong.detach(), dim=1)
    valid = (nearest(ignore, strong.shape[-2:]) != 255).float()
    sim = (similarity * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    amplitude = ((sim + 1) * 3 / 20).clamp(0, 0.3)[:, None, None, None]
    noise = strong * (1 + torch.empty_like(strong).uniform_(-1, 1) * amplitude)
    attention = strong.detach().mean(1, keepdim=True)
    maximum = attention.flatten(1).amax(1)[:, None, None, None]
    factor = 1 - amplitude + torch.empty_like(amplitude).uniform_(-0.1, 0.1)
    dropped = strong * (attention < maximum * factor)
    return noise, dropped


def perturb_weight(step, total_iters, cfg):
    progress = min(max(step / max(total_iters * cfg.get('ramp_up', 1.0), 1), 0), 1)
    return cfg['unsupervised_w'] * math.exp(-10 * (1 - progress) ** 2)


class PrototypeBank(nn.Module):
    def __init__(self, nclass, channels=256, momentum=0.99, threshold=0.95):
        super().__init__()
        self.register_buffer('memory', torch.zeros(nclass, channels))
        self.register_buffer('initialized', torch.zeros(nclass, dtype=torch.bool))
        self.momentum, self.threshold = momentum, threshold

    def forward(self, weak, strong, probabilities, ignore):
        probabilities = F.interpolate(probabilities.detach(), weak.shape[-2:], mode='bilinear', align_corners=False)
        confidence, label = probabilities.max(1)
        valid = nearest(ignore, weak.shape[-2:]) != 255
        weak_flat = weak.detach().permute(0, 2, 3, 1).reshape(-1, weak.shape[1])
        strong_flat = strong.permute(0, 2, 3, 1).reshape(-1, strong.shape[1])
        label, confidence, valid = label.flatten(), confidence.flatten(), valid.flatten()
        loss, count = strong.sum() * 0, 0
        # Match by explicit class, never zip filtered per-class lists.
        for cls in range(len(self.memory)):
            selected = valid & (label == cls)
            reliable = selected & (confidence > self.threshold)
            w, s = weak_flat[reliable], strong_flat[selected]
            if not len(w) or not len(s):
                continue
            reference = self.memory[cls].detach().clone()
            wi = F.cosine_similarity(w, reference[None], dim=1).topk(min(16, len(w))).indices
            si = F.cosine_similarity(s.detach(), reference[None], dim=1).topk(min(256, len(s)), largest=False).indices
            similarities = F.normalize(w[wi], dim=1) @ F.normalize(s[si], dim=1).T
            loss = loss + (1 - similarities.max(dim=1).values).sum()
            count += len(wi)
        return loss / max(count, 1)

    @torch.no_grad()
    def update(self, features, probabilities, ignore):
        probs = F.interpolate(probabilities.detach(), features.shape[-2:], mode='bilinear', align_corners=False)
        confidence, labels = probs.max(1)
        valid = (nearest(ignore, features.shape[-2:]) != 255) & (confidence > self.threshold)
        feats = features.detach().permute(0, 2, 3, 1)
        for cls in range(len(self.memory)):
            values = feats[valid & (labels == cls)]
            if not len(values):
                continue
            mean = values.mean(0)
            if self.initialized[cls]:
                self.memory[cls].lerp_(mean, 1 - self.momentum)
            else:
                self.memory[cls].copy_(mean)
                self.initialized[cls] = True
