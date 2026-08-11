"""Image and mask transforms sharing identical geometry.

The image path uses BILINEAR, the trimap/mask path NEAREST, but both apply the
same resize + center crop, so masks stay pixel-aligned with model inputs.
Stage 1 is deterministic-only; train-time augmentations arrive in Stage 2.
The spurious corner patch is applied *after* these transforms (see patches.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from vision_xai.config import DataConfig
from vision_xai.data.trimap import resize_and_center_crop_trimap, trimap_to_mask


def build_eval_transform(cfg: DataConfig) -> Callable[[Image.Image], torch.Tensor]:
    """Resize (shorter side) -> center crop -> float tensor in [0, 1]."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    pipeline = transforms.Compose(
        [
            transforms.Resize(
                cfg.resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
        ]
    )
    return cast(Callable[[Image.Image], torch.Tensor], pipeline)


def build_train_transform(cfg: DataConfig) -> Callable[[Image.Image], torch.Tensor]:
    """Same deterministic geometry as :func:`build_eval_transform`.

    Train-time augmentation (a horizontal flip) is deliberately *not* baked in
    here as a stateful ``transforms.RandomHorizontalFlip()`` — that draws from
    torch's global RNG, whose state depends on how many prior draws happened
    *in this process*, which breaks resume-equivalence (a resumed run is a new
    process and would replay a different epoch's flip decisions). Instead,
    :class:`~vision_xai.data.datasets.PetClassificationDataset` applies the
    flip itself, deterministically per ``(seed, sample_id)`` — see its
    ``train_augment_seed`` parameter — so the augmentation is a pure function
    of the sample, independent of process/epoch history. The flip still runs
    *before* the spurious patch is applied (patch application lives in the
    dataset, after this transform), so the patch always occupies the same
    fixed corner — that fixed location is the definition of the spurious cue.
    """
    return build_eval_transform(cfg)


def make_normalizer(
    mean: tuple[float, float, float], std: tuple[float, float, float]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Channel-wise normalization applied *after* the spurious patch (post-patch
    transform), so patch pixels are normalized exactly like image pixels."""
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    def normalize(image: torch.Tensor) -> torch.Tensor:
        return (image - mean_tensor) / std_tensor

    return normalize


def build_mask_transform(cfg: DataConfig) -> Callable[[Image.Image], npt.NDArray[np.bool_]]:
    """Same geometry as :func:`build_eval_transform`, NEAREST, then trimap -> bool mask."""

    def transform(trimap: Image.Image) -> npt.NDArray[np.bool_]:
        aligned = resize_and_center_crop_trimap(trimap, cfg.resize_size, cfg.image_size)
        return trimap_to_mask(aligned, cfg.mask_policy)

    return transform
