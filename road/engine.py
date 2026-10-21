"""One MCCL+Source optimizer step; sequential graphs bound GPU memory."""
import torch
import torch.nn.functional as F

from .losses import (alignment, intervene, masked_ce, masked_mse,
                     perturb_weight, supervised_loss)


def mix(value, donor, box):
    mask = box.bool()
    if value.ndim == 4:
        mask = mask[:, None]
    return torch.where(mask, donor, value)


def train_step(model, bank, optimizer, labeled, unlabeled, source, cfg, step):
    model.train()
    device = next(model.parameters()).device
    labeled = [v.to(device) for v in labeled]
    weak, strong1, strong2, ignore, box1, box2 = [v.to(device) for v in unlabeled]
    if weak.shape[0] < 2:
        raise ValueError('Cross-image CutMix requires target batch >= 2')
    optimizer.zero_grad(set_to_none=True)
    metrics = {}

    def backward(name, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {name} at step {step}')
        loss.backward()
        metrics[name] = float(loss.detach())

    source_weight = float(cfg.get('source_loss_weight', 1.0)) if source is not None else 0.0
    backward('target_gt', supervised_loss(model(labeled[0]), labeled[1], cfg) / (2 * (1 + source_weight)))
    if source is not None and source_weight:
        source = [v.to(device) for v in source]
        backward('source_gt', supervised_loss(model(source[0]), source[1], cfg) * source_weight / (2 * (1 + source_weight)))

    # Same-network weak targets, as in MCCL; there is no extra EMA teacher.
    logits_w, feature_w = model(weak, return_features=True)
    probability = logits_w.detach().softmax(1)
    confidence, labels = probability.max(1)
    reference_w = feature_w.detach()
    fp_logits = model.classify_context(F.dropout2d(feature_w, 0.5, training=True), weak.shape[-2:])
    backward('weak_fp', 0.25 * masked_ce(fp_logits, labels, confidence, ignore, cfg['conf_thresh']))
    del logits_w, feature_w, fp_logits

    # Prototype compactness uses the uncut strong view, matching official code.
    _, proto_features = model(strong1, return_features=True)
    backward('prototype', cfg['feat_nearest'] * bank(reference_w, proto_features, probability, ignore))
    del proto_features
    bank.update(reference_w, probability, ignore)

    for index, (strong, box) in enumerate(zip((strong1, strong2), (box1, box2)), 1):
        # All fields follow the identical spatial CutMix. Donor is another image
        # in the batch, following the existing controlled UniMatchV2 adaptation.
        weak_mix = mix(weak, weak.roll(1, 0), box)
        strong_mix = mix(strong, strong.roll(1, 0), box)
        ignore_mix = mix(ignore, ignore.roll(1, 0), box)
        probability_mix = mix(probability, probability.roll(1, 0), box)
        conf_mix, labels_mix = probability_mix.max(1)
        with torch.no_grad():
            _, feature_reference = model(weak_mix, return_features=True)
        logits_s, features_s = model(strong_mix, return_features=True)
        loss = 0.125 * masked_ce(logits_s, labels_mix, conf_mix, ignore_mix, cfg['conf_thresh'])
        point_loss = alignment(feature_reference, features_s, ignore_mix)
        loss = loss + cfg['feat_ali_weight'] * point_loss / 2
        metrics[f'point_alignment_{index}'] = float(point_loss.detach())
        noise, dropped = intervene(feature_reference, features_s, ignore_mix)
        weight = perturb_weight(step, cfg['total_iters'], cfg) / 2
        for name, enabled, feature in (('noise', cfg['F-Noise'], noise), ('drop', cfg['F-Drop'], dropped)):
            if enabled:
                prediction = model.classify_context(feature, weak.shape[-2:])
                consistency = weight * masked_mse(prediction, probability_mix, ignore_mix)
                loss = loss + consistency
                metrics[f'{name}_{index}'] = float(consistency.detach())
                del prediction, consistency
        backward(f'strong_{index}', loss)
        del logits_s, features_s, feature_reference, loss, noise, dropped

    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Nonfinite gradient at step {step}')
    optimizer.step()
    terms = ('target_gt', 'source_gt', 'weak_fp', 'prototype', 'strong_1', 'strong_2')
    metrics['total'] = sum(metrics.get(k, 0) for k in terms)
    metrics['mask_ratio'] = float(((confidence >= cfg['conf_thresh']) & (ignore != 255)).sum() / (ignore != 255).sum().clamp_min(1))
    return metrics
