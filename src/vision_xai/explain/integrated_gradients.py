"""Integrated Gradients via captum, channel-summed to per-pixel maps.

Baseline is the all-zeros tensor, which in normalized input space corresponds
to the dataset (ImageNet) mean image.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


def ig_attributions(
    model: nn.Module, images: torch.Tensor, targets: torch.Tensor, *, n_steps: int
) -> torch.Tensor:
    from captum.attr import IntegratedGradients

    integrated_gradients = IntegratedGradients(model)
    # captum's attribute() has overloads mypy cannot resolve under this config.
    attributions = cast(
        torch.Tensor,
        integrated_gradients.attribute(
            images,
            target=targets,
            baselines=torch.zeros_like(images),
            n_steps=n_steps,
            # Without this, captum expands the batch to (batch * n_steps) and
            # runs it as ONE forward/backward — 16 images x 32 steps = 512
            # images' worth of backbone activations *with* input gradients,
            # which OOMs a 22 GB GPU. Capping at the incoming batch size keeps
            # peak memory at roughly one ordinary batch, whatever n_steps is;
            # the maths is unchanged, only the chunking.
            internal_batch_size=images.shape[0],
        ),
    )
    return attributions.sum(dim=1)
