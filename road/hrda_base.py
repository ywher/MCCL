"""DINOv3-B + ReIN + HRDA adapter for UniMatchV2 training."""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .road_vfm import HRDAEncoderDecoder, HRDAHead, ReinsDINOv3


def _extract_logits(output):
    if isinstance(output, dict):
        output = output.get("seg_logits", output)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported HRDA output type: {type(output)!r}")
    return output


def _backbone_config():
    return {
        "dinov3_config": {
            "img_size": 512,
            "patch_size": 16,
            "pos_embed_rope_rescale_coords": 2.0,
            "pos_embed_rope_dtype": "fp32",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "ffn_ratio": 4.0,
            "qkv_bias": True,
            "layerscale_init": 1e-5,
            "ffn_layer": "mlp",
            "ffn_bias": True,
            "proj_bias": True,
            "n_storage_tokens": 4,
            "mask_k_bias": True,
            "out_indices": [2, 5, 8, 11],
        },
        "reins_config": {
            "lora_dim": 16,
            "num_layers": 12,
            "non_adapter_layers": 0,
            "embed_dims": 768,
            "patch_size": 16,
            "token_length": 100,
            "link_token_to_query": False,
        },
    }


def _decoder_config(num_classes):
    return {
        "in_channels": [768, 768, 768, 768],
        "in_index": [0, 1, 2, 3],
        "channels": 256,
        "dropout_ratio": 0.1,
        "num_classes": num_classes,
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "align_corners": False,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
        "single_scale_head": "DAFormerHead",
        "interpolate": False,
        "decoder_params": {
            "embed_dims": 256,
            "embed_cfg": {"type": "mlp", "act_cfg": None, "norm_cfg": None},
            "embed_neck_cfg": {
                "type": "mlp",
                "act_cfg": None,
                "norm_cfg": None,
            },
            "fusion_cfg": {
                "type": "aspp",
                "sep": True,
                "dilations": [1, 6, 12, 18],
                "pool": False,
                "act_cfg": {"type": "ReLU"},
                "norm_cfg": {"type": "BN", "requires_grad": True},
            },
        },
        "lr_loss_weight": 0,
        "hr_loss_weight": 0.1,
        "scales": [0.5, 1.0],
        "attention_embed_dim": 256,
        "attention_classwise": True,
        "enable_hr_crop": True,
        "hr_slide_inference": True,
        "hr_slide_overlapping": True,
        "hr_slide_batch_size": 4,
        "crop_coord_divisible": 8,
        "blur_hr_crop": False,
        "feature_scale": 0.5,
        "fixed_attention": None,
        "debug_output_attention": False,
    }


class HRDABase(nn.Module):
    """Expose the controlled TC-ADA segmentor through UniMatch's model API."""

    def __init__(self, num_classes, pretrained, work_dir="."):
        super().__init__()
        pretrained = Path(pretrained).expanduser().resolve()
        if not pretrained.is_file():
            raise FileNotFoundError(f"DINOv3-B checkpoint does not exist: {pretrained}")

        backbone = ReinsDINOv3(
            backbone_config=_backbone_config(),
            pretrained={"dinov3": str(pretrained)},
        )
        decode_head = HRDAHead(_decoder_config(num_classes))
        self.segmentor = HRDAEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            auxiliary_head=None,
            token_mask_ratio=None,
            train_cfg={"work_dir": str(work_dir)},
            test_cfg={
                "mode": "slide",
                "stride": [512, 512],
                "crop_size": [1024, 1024],
                "batched_slide": False,
            },
        )
        self.is_peft_enabled = True
        self.peft_config = {
            "name": "reins",
            "segmentor": "dinov3_base_rein_hrda",
        }
        self.feature_channels = 768
        self.output_channels = 256

    @property
    def backbone(self):
        return self.segmentor.backbone

    @property
    def head(self):
        return self.segmentor.decode_head

    def _forward_train(self, images, comp_drop=False):
        try:
            features, _ = self.segmentor._forward_train_features(images)
            if comp_drop:
                kept_ratio = 0.5 if isinstance(comp_drop, bool) else float(
                    comp_drop.get("kept_ratio", 0.5)
                )
                features = self.segmentor._apply_complementary_dropout(
                    features, kept_ratio=kept_ratio
                )
            return _extract_logits(self.segmentor.decode_head.forward_train(features))
        finally:
            self.segmentor.decode_head.reset_crop()

    def _forward_inference(self, images):
        try:
            self.segmentor.decode_head.reset_crop()
            return _extract_logits(
                self.segmentor.encode_decode(
                    images, return_feat=False, upscale_pred=True
                )
            )
        finally:
            self.segmentor.decode_head.reset_crop()

    def forward(self, images, comp_drop=False, return_features=False):
        if return_features:
            raise NotImplementedError(
                "Feature-return mode is not used by the controlled UniMatchV2+Source protocol"
            )
        logits = (
            self._forward_train(images, comp_drop=comp_drop)
            if self.training
            else self._forward_inference(images)
        )
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def peft_state_dict(self):
        return {
            "adapter": self.segmentor.backbone.adapter.state_dict(),
            "decoder": self.segmentor.decode_head.state_dict(),
        }

    def load_peft_state_dict(self, state_dict, strict=True):
        missing_adapter, unexpected_adapter = (
            self.segmentor.backbone.adapter.load_state_dict(
                state_dict["adapter"], strict=strict
            )
        )
        missing_decoder, unexpected_decoder = (
            self.segmentor.decode_head.load_state_dict(
                state_dict["decoder"], strict=strict
            )
        )
        missing = [f"adapter.{key}" for key in missing_adapter]
        missing += [f"decoder.{key}" for key in missing_decoder]
        unexpected = [f"adapter.{key}" for key in unexpected_adapter]
        unexpected += [f"decoder.{key}" for key in unexpected_decoder]
        return missing, unexpected

    def optimizer_param_groups(self, base_lr, lr_multi, weight_decay):
        norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
        )
        groups = []
        seen = set()
        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                full_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                lr_scale = lr_multi if "decode_head" in full_name else 1.0
                no_decay = (
                    isinstance(module, norm_types)
                    or parameter.ndim == 1
                    or "learnable_tokens" in full_name
                )
                groups.append(
                    {
                        "params": [parameter],
                        "lr": base_lr * lr_scale,
                        "lr_scale": lr_scale,
                        "weight_decay": 0.0 if no_decay else weight_decay,
                    }
                )
        return groups
