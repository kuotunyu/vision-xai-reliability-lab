"""Occlusion sensitivity via captum, channel-summed to per-pixel maps.

The occlusion baseline value 0.0 corresponds to the dataset mean in normalized
input space (same convention as Integrated Gradients).
"""

from __future__ import annotations

import torch
from torch import nn


def occlusion_attributions(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    window: int,
    stride: int,
) -> torch.Tensor:
    from captum.attr import Occlusion

    height, width = images.shape[-2:]
    window = min(window, height, width)
    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
        images,
        target=targets,
        sliding_window_shapes=(images.shape[1], window, window),
        strides=(images.shape[1], stride, stride),
        baselines=0.0,
        show_progress=False,
    )
    result: torch.Tensor = attributions.sum(dim=1)
    return result
