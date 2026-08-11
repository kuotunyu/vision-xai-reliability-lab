from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from conftest import RecordFactory
from vision_xai.config import SpuriousPatchConfig
from vision_xai.data.manifest import ManifestRecord
from vision_xai.data.patches import (
    PatchVariant,
    apply_corner_patch,
    assign_patches,
    in_target_group,
    patch_geometry,
    patch_mask,
)


def _cfg(**overrides: Any) -> SpuriousPatchConfig:
    return SpuriousPatchConfig.model_validate(overrides)


def _mixed_records(record_factory: RecordFactory, per_species: int = 50) -> list[ManifestRecord]:
    records = [
        record_factory(f"Cat_{i:03d}", species="cat", class_id=0) for i in range(per_species)
    ]
    records += [
        record_factory(f"dog_{i:03d}", species="dog", class_id=20) for i in range(per_species)
    ]
    return records


def test_assignment_is_deterministic_and_order_invariant(
    record_factory: RecordFactory,
) -> None:
    records = _mixed_records(record_factory)
    cfg = _cfg()
    first = assign_patches(records, cfg, seed=42, variant=PatchVariant.CORRELATED)
    second = assign_patches(records, cfg, seed=42, variant=PatchVariant.CORRELATED)
    assert first == second
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    assert assign_patches(shuffled, cfg, seed=42, variant=PatchVariant.CORRELATED) == first


def test_no_patch_variant_and_disabled_assign_nothing(record_factory: RecordFactory) -> None:
    records = _mixed_records(record_factory, per_species=10)
    assert not any(assign_patches(records, _cfg(), seed=42, variant=PatchVariant.NO_PATCH).values())
    assert not any(
        assign_patches(
            records, _cfg(enabled=False), seed=42, variant=PatchVariant.CORRELATED
        ).values()
    )


def test_extreme_probabilities_are_exact_and_counter_inverts(
    record_factory: RecordFactory,
) -> None:
    records = _mixed_records(record_factory, per_species=20)
    cfg = _cfg(p_patch_in_group=1.0, p_patch_out_group=0.0, target_group="cats")
    correlated = assign_patches(records, cfg, seed=42, variant=PatchVariant.CORRELATED)
    counter = assign_patches(records, cfg, seed=42, variant=PatchVariant.COUNTER_CORRELATED)
    for record in records:
        assert correlated[record.sample_id] == (record.species == "cat")
        assert counter[record.sample_id] == (record.species == "dog")


def test_default_probabilities_separate_groups(record_factory: RecordFactory) -> None:
    records = _mixed_records(record_factory, per_species=100)
    cfg = _cfg()  # 0.9 in-group (cats) / 0.1 out-group
    assignment = assign_patches(records, cfg, seed=42, variant=PatchVariant.CORRELATED)
    cat_rate = float(np.mean([assignment[r.sample_id] for r in records if r.species == "cat"]))
    dog_rate = float(np.mean([assignment[r.sample_id] for r in records if r.species == "dog"]))
    assert cat_rate > 0.7
    assert dog_rate < 0.3


def test_class_id_list_target_group(record_factory: RecordFactory) -> None:
    cfg = _cfg(target_group=[0, 5])
    assert in_target_group(record_factory("Cat_001", species="cat", class_id=0), cfg)
    assert in_target_group(record_factory("dog_001", species="dog", class_id=5), cfg)
    assert not in_target_group(record_factory("dog_002", species="dog", class_id=20), cfg)


def test_patch_geometry_all_corners() -> None:
    image_size = 32
    cfg_kwargs = {"size_fraction": 0.25, "margin_fraction": 0.0}
    assert patch_geometry(image_size, _cfg(corner="top_left", **cfg_kwargs)) == (0, 0, 8, 8)
    assert patch_geometry(image_size, _cfg(corner="top_right", **cfg_kwargs)) == (24, 0, 32, 8)
    assert patch_geometry(image_size, _cfg(corner="bottom_left", **cfg_kwargs)) == (0, 24, 8, 32)
    assert patch_geometry(image_size, _cfg(corner="bottom_right", **cfg_kwargs)) == (
        24,
        24,
        32,
        32,
    )


def test_apply_writes_exactly_the_patch_bbox_float() -> None:
    cfg = _cfg()
    image = torch.full((3, 32, 32), 0.5)
    out = apply_corner_patch(image, cfg)
    changed = (out != image).any(dim=0).numpy()
    assert np.array_equal(changed, patch_mask(32, cfg))
    assert torch.all(image == 0.5)  # input untouched
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_apply_uint8_and_checker_colors() -> None:
    cfg = _cfg(pattern="checker", color=(255, 0, 255), color2=(0, 0, 0))
    image = torch.full((3, 32, 32), 128, dtype=torch.uint8)
    out = apply_corner_patch(image, cfg)
    assert out.dtype == torch.uint8
    x0, y0, x1, y1 = patch_geometry(32, cfg)
    patch_region = out[:, y0:y1, x0:x1]
    # top-left checker cell carries `color`, and both colors appear
    assert patch_region[:, 0, 0].tolist() == [255, 0, 255]
    flat = patch_region.permute(1, 2, 0).reshape(-1, 3)
    assert [255, 0, 255] in flat.tolist()
    assert [0, 0, 0] in flat.tolist()


def test_solid_pattern_is_uniform() -> None:
    cfg = _cfg(pattern="solid", color=(10, 20, 30))
    image = torch.full((3, 32, 32), 128, dtype=torch.uint8)
    out = apply_corner_patch(image, cfg)
    x0, y0, x1, y1 = patch_geometry(32, cfg)
    region = out[:, y0:y1, x0:x1]
    assert torch.all(region[0] == 10) and torch.all(region[1] == 20) and torch.all(region[2] == 30)


def test_apply_rejects_non_square_and_bad_shape() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="square"):
        apply_corner_patch(torch.zeros(3, 32, 48), cfg)
    with pytest.raises(ValueError, match="CHW"):
        apply_corner_patch(torch.zeros(1, 32, 32), cfg)


def test_patch_mask_area_matches_geometry() -> None:
    cfg = _cfg()
    x0, y0, x1, y1 = patch_geometry(224, cfg)
    mask = patch_mask(224, cfg)
    assert int(mask.sum()) == (x1 - x0) * (y1 - y0)
