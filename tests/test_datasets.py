from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import ConfigFactory
from vision_xai.config import AppConfig
from vision_xai.data.datasets import PetClassificationDataset
from vision_xai.data.manifest import MANIFEST_FILENAME, ManifestRecord, load_manifest
from vision_xai.data.patches import PatchVariant, assign_patches, patch_mask
from vision_xai.data.prepare import prepare_data
from vision_xai.data.transforms import build_eval_transform, build_mask_transform
from vision_xai.paths import manifests_dir, resolve_data_dir


@pytest.fixture()
def prepared(
    app_config: AppConfig,
) -> tuple[AppConfig, list[ManifestRecord], Path]:
    result = prepare_data(app_config)
    assert result.completed
    records = load_manifest(manifests_dir(app_config) / MANIFEST_FILENAME)
    return app_config, records, resolve_data_dir(app_config)


def test_returns_image_and_label(
    prepared: tuple[AppConfig, list[ManifestRecord], Path],
) -> None:
    cfg, records, data_root = prepared
    dataset = PetClassificationDataset(records, data_root, transform=build_eval_transform(cfg.data))
    assert len(dataset) == len(records)
    item = dataset[0]
    assert len(item) == 2
    image, label = item[0], item[1]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, cfg.data.image_size, cfg.data.image_size)
    assert label == records[0].class_id


def test_returns_aligned_mask_when_requested(
    prepared: tuple[AppConfig, list[ManifestRecord], Path],
) -> None:
    cfg, records, data_root = prepared
    dataset = PetClassificationDataset(
        records,
        data_root,
        transform=build_eval_transform(cfg.data),
        mask_transform=build_mask_transform(cfg.data),
        return_mask=True,
    )
    item = dataset[0]
    assert len(item) == 3
    mask = item[2]
    assert mask.shape == (cfg.data.image_size, cfg.data.image_size)
    assert mask.dtype == np.bool_
    assert 0 < int(mask.sum()) < mask.size  # synthetic pet rectangle, not degenerate


def test_patch_applied_only_to_assigned_samples(
    prepared: tuple[AppConfig, list[ManifestRecord], Path],
) -> None:
    cfg, records, data_root = prepared
    patch_cfg = cfg.data.spurious_patch
    test_records = [r for r in records if r.split == "test"]
    assignment = assign_patches(test_records, patch_cfg, cfg.seed, PatchVariant.CORRELATED)
    assert any(assignment.values()) and not all(assignment.values())

    plain = PetClassificationDataset(
        test_records, data_root, transform=build_eval_transform(cfg.data)
    )
    patched = PetClassificationDataset(
        test_records,
        data_root,
        transform=build_eval_transform(cfg.data),
        patch_cfg=patch_cfg,
        patch_assignment=assignment,
    )
    region = patch_mask(cfg.data.image_size, patch_cfg)
    for index, record in enumerate(test_records):
        base_image = plain[index][0]
        patched_image = patched[index][0]
        changed = (patched_image != base_image).any(dim=0).numpy()
        if assignment[record.sample_id]:
            assert changed.any()
            assert not np.logical_and(changed, np.logical_not(region)).any()
        else:
            assert not changed.any()


def test_train_augment_flip_is_deterministic_and_order_invariant(
    prepared: tuple[AppConfig, list[ManifestRecord], Path],
) -> None:
    """The train-time flip must be a pure function of (seed, sample_id): same
    result regardless of call order, and independent of torch's global RNG
    state (unlike torchvision's stateful RandomHorizontalFlip) — required for
    --resume to reproduce an uninterrupted run's augmentation exactly."""
    cfg, records, data_root = prepared
    test_records = [r for r in records if r.split == "test"][:4]

    def build(seed: int, torch_seed: int) -> list[torch.Tensor]:
        torch.manual_seed(torch_seed)  # simulate differing global-RNG history
        dataset = PetClassificationDataset(
            test_records,
            data_root,
            transform=build_eval_transform(cfg.data),
            train_augment_seed=seed,
        )
        return [dataset[i][0] for i in range(len(test_records))]

    first = build(seed=42, torch_seed=0)
    second = build(seed=42, torch_seed=999)  # different global RNG state, same result
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)

    plain_dataset = PetClassificationDataset(
        test_records, data_root, transform=build_eval_transform(cfg.data)
    )
    plain = [plain_dataset[i][0] for i in range(len(test_records))]
    # Every augmented image is either untouched or exactly horizontally flipped.
    for augmented, unaugmented in zip(first, plain, strict=True):
        assert torch.equal(augmented, unaugmented) or torch.equal(
            augmented, torch.flip(unaugmented, dims=[-1])
        )


def test_constructor_validations(synthetic_data_dir: Path, config_factory: ConfigFactory) -> None:
    cfg = config_factory(synthetic_data_dir)
    with pytest.raises(ValueError, match="mask_transform"):
        PetClassificationDataset(
            [], synthetic_data_dir, transform=build_eval_transform(cfg.data), return_mask=True
        )
    with pytest.raises(ValueError, match="together"):
        PetClassificationDataset(
            [],
            synthetic_data_dir,
            transform=build_eval_transform(cfg.data),
            patch_cfg=cfg.data.spurious_patch,
        )
