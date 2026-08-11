"""Grad-CAM via captum's LayerGradCam on the last ConvNeXt feature stage."""

from __future__ import annotations

import torch
from torch import nn

from vision_xai.errors import ExplainError


def _last_feature_stage(model: nn.Module) -> nn.Module:
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Sequential) or len(features) == 0:
        raise ExplainError("gradcam expects a ConvNeXt-style model with a .features Sequential")
    return features[len(features) - 1]


def gradcam_attributions(
    model: nn.Module, images: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    from captum.attr import LayerAttribution, LayerGradCam

    layer_gradcam = LayerGradCam(model, _last_feature_stage(model))
    attributions = layer_gradcam.attribute(images, target=targets, relu_attributions=True)
    upsampled = LayerAttribution.interpolate(
        attributions, tuple(images.shape[-2:]), interpolate_mode="bilinear"
    )
    result: torch.Tensor = upsampled.squeeze(1)
    return result
