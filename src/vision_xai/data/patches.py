"""Synthetic spurious corner patch: assignment, pixel application, and mask.

Assignment (who gets a patch, per variant) is decided at prepare time and
persisted for auditability; pixels are applied on the fly by the dataset,
*after* all geometric transforms, so patch edges stay crisp and
:func:`patch_mask` is exactly aligned with model-input coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import torch

from vision_xai.config import SpuriousPatchConfig
from vision_xai.data.manifest import ManifestRecord
from vision_xai.utils.seed import per_sample_rng

CHECKER_CELLS_PER_SIDE = 4


class PatchVariant(StrEnum):
    CORRELATED = "correlated"
    NO_PATCH = "no_patch"
    COUNTER_CORRELATED = "counter_correlated"


def in_target_group(record: ManifestRecord, cfg: SpuriousPatchConfig) -> bool:
    if cfg.target_group == "cats":
        return record.species == "cat"
    if cfg.target_group == "dogs":
        return record.species == "dog"
    return record.class_id in cfg.target_group


def assign_patches(
    records: Sequence[ManifestRecord],
    cfg: SpuriousPatchConfig,
    seed: int,
    variant: PatchVariant,
) -> dict[str, bool]:
    """Decide per sample whether it carries the patch.

    Each draw is a pure function of ``(seed, variant, sample_id)`` — independent of
    iteration order, subset choice, and platform.
    """
    assignment: dict[str, bool] = {}
    for record in records:
        if not cfg.enabled or variant is PatchVariant.NO_PATCH:
            assignment[record.sample_id] = False
            continue
        in_group = in_target_group(record, cfg)
        if variant is PatchVariant.CORRELATED:
            probability = cfg.p_patch_in_group if in_group else cfg.p_patch_out_group
        else:
            probability = cfg.p_patch_out_group if in_group else cfg.p_patch_in_group
        rng = per_sample_rng(seed, record.sample_id, f"patch:{variant.value}")
        assignment[record.sample_id] = bool(rng.random() < probability)
    return assignment


def patch_geometry(image_size: int, cfg: SpuriousPatchConfig) -> tuple[int, int, int, int]:
    """Return the patch bounding box ``(x0, y0, x1, y1)`` in model-input pixels."""
    size = max(1, round(cfg.size_fraction * image_size))
    margin = round(cfg.margin_fraction * image_size)
    if size + 2 * margin > image_size:
        raise ValueError(
            f"patch (size {size} + 2*margin {margin}) does not fit in image of {image_size}px"
        )
    near, far = margin, image_size - margin - size
    x0 = near if cfg.corner in ("top_left", "bottom_left") else far
    y0 = near if cfg.corner in ("top_left", "top_right") else far
    return (x0, y0, x0 + size, y0 + size)


def _patch_pixels(size: int, cfg: SpuriousPatchConfig) -> npt.NDArray[np.uint8]:
    color = np.array(cfg.color, dtype=np.uint8)
    color2 = np.array(cfg.color2, dtype=np.uint8)
    if cfg.pattern == "solid":
        return np.broadcast_to(color, (size, size, 3)).copy()
    cell = max(1, size // CHECKER_CELLS_PER_SIDE)
    yy, xx = np.mgrid[0:size, 0:size]
    checker = ((yy // cell) + (xx // cell)) % 2 == 0
    return np.where(checker[..., None], color, color2).astype(np.uint8)


def apply_corner_patch(image: torch.Tensor, cfg: SpuriousPatchConfig) -> torch.Tensor:
    """Overwrite the configured corner of a CHW image tensor with the patch pattern.

    Accepts float tensors in [0, 1] or uint8 tensors; must be called on square,
    already-transformed inputs (after resize + center crop).
    """
    rgb_channels = 3
    if image.ndim != rgb_channels or image.shape[0] != rgb_channels:
        raise ValueError(f"expected a CHW RGB tensor, got shape {tuple(image.shape)}")
    _, height, width = image.shape
    if height != width:
        raise ValueError(
            f"patch geometry is defined on square inputs (apply after center crop), "
            f"got {height}x{width}"
        )
    x0, y0, x1, y1 = patch_geometry(height, cfg)
    patch = torch.from_numpy(_patch_pixels(x1 - x0, cfg)).permute(2, 0, 1)
    patch = patch.to(image.dtype) / 255.0 if image.is_floating_point() else patch.to(image.dtype)
    out = image.clone()
    out[:, y0:y1, x0:x1] = patch
    return out


def patch_mask(image_size: int, cfg: SpuriousPatchConfig) -> npt.NDArray[np.bool_]:
    """Boolean mask (HxW) of the patch region in model-input coordinates."""
    x0, y0, x1, y1 = patch_geometry(image_size, cfg)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask
