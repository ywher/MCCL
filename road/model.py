"""Expose HRDA context decoder features on a common full-image lattice."""
import torch
import torch.nn.functional as F

from .hrda_base import HRDABase


class MCCLHRDA(HRDABase):
    def forward(self, images, return_features=False):
        if not return_features:
            return super().forward(images)
        if not self.training:
            raise ValueError('Decoder feature extraction is a training-only interface')
        captured = []
        # HRDA calls the shared classifier for context before the detail crop.
        # Context coordinates remain aligned across weak/strong views even when
        # their independently sampled HR crops differ.
        hook = self.head.head.conv_seg.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            logits = self._forward_train(images)
        finally:
            hook.remove()
        if len(captured) != 2:
            raise RuntimeError(f'Expected HRDA context/detail pair, got {len(captured)}')
        logits = F.interpolate(logits, images.shape[-2:], mode='bilinear', align_corners=False)
        return logits, captured[0]

    def classify_context(self, features, output_size):
        # Features were captured after the decoder's ordinary dropout.
        logits = self.head.head.conv_seg(features)
        return F.interpolate(logits, output_size, mode='bilinear', align_corners=False)

    def freeze_backbone_check(self):
        frozen = [p for n, p in self.backbone.named_parameters() if not n.startswith('adapter.')]
        if not frozen or any(p.requires_grad for p in frozen):
            raise RuntimeError('The generic DINOv3 encoder must remain frozen')
