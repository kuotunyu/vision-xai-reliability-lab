from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from conftest import FOREGROUND_RECT, IMG_SIZE, write_trimap
from vision_xai.config import DataConfig
from vision_xai.data.transforms import build_eval_transform, build_mask_transform
from vision_xai.data.trimap import (
    TRIMAP_BACKGROUND,
    TRIMAP_BOUNDARY,
    TRIMAP_FOREGROUND,
    resize_and_center_crop_trimap,
    trimap_to_mask,
    verify_trimap,
)


def _load_trimap(tmp_path: Path, swapped: bool = False) -> Image.Image:
    path = tmp_path / "trimap.png"
    write_trimap(path, swapped=swapped)
    return Image.open(path)


def test_trimap_to_mask_policies() -> None:
    trimap = np.array(
        [[TRIMAP_FOREGROUND, TRIMAP_BACKGROUND], [TRIMAP_BOUNDARY, TRIMAP_BACKGROUND]],
        dtype=np.uint8,
    )
    fg_only = trimap_to_mask(trimap, "foreground")
    assert fg_only.tolist() == [[True, False], [False, False]]
    with_boundary = trimap_to_mask(trimap, "foreground_and_boundary")
    assert with_boundary.tolist() == [[True, False], [True, False]]


def test_nearest_resize_preserves_value_set(tmp_path: Path) -> None:
    with _load_trimap(tmp_path) as trimap:
        aligned = resize_and_center_crop_trimap(trimap, resize_size=40, crop_size=32)
    assert aligned.shape == (32, 32)
    assert set(np.unique(aligned).tolist()) <= {1, 2, 3}


def test_crop_alignment_is_exact_without_scaling() -> None:
    """With an identity resize, image and mask transforms must agree pixel-perfectly."""
    crop = 32
    cfg = DataConfig.model_validate(
        {"download": False, "image_size": crop, "resize_size": IMG_SIZE}
    )
    # A rectangle strictly inside the crop window: [20,40) maps to [4,24) after
    # the 64->32 center crop (offset 16), so the expected mask is non-trivial.
    x0, y0, x1, y1 = 20, 20, 40, 40

    trimap_array = np.full((IMG_SIZE, IMG_SIZE), TRIMAP_BACKGROUND, dtype=np.uint8)
    trimap_array[y0:y1, x0:x1] = TRIMAP_FOREGROUND
    indicator = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    indicator[y0:y1, x0:x1, :] = 255

    image_tensor = build_eval_transform(cfg)(Image.fromarray(indicator, mode="RGB"))
    image_mask = (image_tensor[0] > 0.5).numpy()
    pet_mask = build_mask_transform(cfg)(Image.fromarray(trimap_array, mode="L"))

    assert image_mask.shape == pet_mask.shape == (crop, crop)
    offset = (IMG_SIZE - crop) // 2
    expected = np.zeros((crop, crop), dtype=bool)
    expected[y0 - offset : y1 - offset, x0 - offset : x1 - offset] = True
    assert np.array_equal(pet_mask, expected)
    assert np.array_equal(image_mask, expected)
    assert expected.sum() not in (0, crop * crop)  # guard against a degenerate check


def test_scaled_alignment_is_tight(tmp_path: Path) -> None:
    """Through the scaled path (BILINEAR vs NEAREST), masks must still overlap tightly."""
    cfg = DataConfig.model_validate({"download": False, "image_size": 32, "resize_size": 40})
    x0, y0, x1, y1 = FOREGROUND_RECT
    indicator = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    indicator[y0:y1, x0:x1, :] = 255
    image_tensor = build_eval_transform(cfg)(Image.fromarray(indicator, mode="RGB"))
    image_mask = (image_tensor[0] > 0.5).numpy()

    with _load_trimap(tmp_path) as trimap:
        pet_mask = build_mask_transform(cfg)(trimap)

    intersection = np.logical_and(image_mask, pet_mask).sum()
    union = np.logical_or(image_mask, pet_mask).sum()
    assert union > 0
    assert intersection / union > 0.9


def test_verify_trimap_accepts_official_semantics(tmp_path: Path) -> None:
    with _load_trimap(tmp_path) as trimap:
        result = verify_trimap(np.asarray(trimap, dtype=np.uint8))
    assert result.values_ok
    assert result.border_background_fraction == 1.0


def test_verify_trimap_flags_swapped_semantics(tmp_path: Path) -> None:
    with _load_trimap(tmp_path, swapped=True) as trimap:
        result = verify_trimap(np.asarray(trimap, dtype=np.uint8))
    assert result.values_ok  # values are still within {1,2,3}...
    assert result.border_background_fraction == 0.0  # ...but the heuristic catches the swap


def test_verify_trimap_flags_unexpected_values() -> None:
    bad = np.full((8, 8), 7, dtype=np.uint8)
    result = verify_trimap(bad)
    assert not result.values_ok
    assert result.unique_values == (7,)
