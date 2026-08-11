from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ConfigFactory, make_synthetic_pet_tree, write_trimap
from vision_xai.config import AppConfig
from vision_xai.data.manifest import (
    MANIFEST_FILENAME,
    PARTIAL_MANIFEST_FILENAME,
    load_manifest,
    load_state,
)
from vision_xai.data.prepare import prepare_data
from vision_xai.errors import DatasetIntegrityError, ResumeStateError
from vision_xai.paths import manifests_dir, raw_results_dir

TOTAL_SAMPLES = 4 * 6 + 4 * 3  # 4 classes, 6 trainval + 3 test each

PATCH_ASSIGNMENT_NAMES = (
    "train_correlated",
    "test_correlated",
    "test_no_patch",
    "test_counter_correlated",
)


def test_full_run_produces_all_artifacts(app_config: AppConfig) -> None:
    result = prepare_data(app_config)
    assert result.completed
    assert result.num_total == TOTAL_SAMPLES
    assert result.num_processed_this_run == TOTAL_SAMPLES
    # per class: 6 trainval -> round(6*0.2)=1 val, 5 train; 3 test
    assert result.counts_by_split == {"test": 12, "train": 20, "val": 4}

    man_dir = manifests_dir(app_config)
    records = load_manifest(man_dir / MANIFEST_FILENAME)
    assert len(records) == TOTAL_SAMPLES
    assert not (man_dir / PARTIAL_MANIFEST_FILENAME).exists()
    assert load_state(man_dir) is None
    for name in PATCH_ASSIGNMENT_NAMES:
        assert (man_dir / f"patch_assignments_{name}.jsonl").is_file()

    out_dir = raw_results_dir(app_config)
    for artifact in ("fingerprint.json", "split_summary.json", "patch_summary.json"):
        assert (out_dir / artifact).is_file()


def test_interrupted_then_resumed_matches_uninterrupted(
    tmp_path: Path, config_factory: ConfigFactory
) -> None:
    tree_fresh = tmp_path / "fresh"
    tree_resumed = tmp_path / "resumed"
    make_synthetic_pet_tree(tree_fresh)
    make_synthetic_pet_tree(tree_resumed)

    fresh = prepare_data(config_factory(tree_fresh))
    assert fresh.completed

    cfg_resumed = config_factory(tree_resumed)
    partial = prepare_data(cfg_resumed, max_items=7)
    assert not partial.completed
    assert partial.num_processed_this_run == 7
    man_dir = manifests_dir(cfg_resumed)
    state = load_state(man_dir)
    assert state is not None and state.stage == "hashing" and state.completed == 7
    partial_records = load_manifest(man_dir / PARTIAL_MANIFEST_FILENAME, require_split=False)
    assert len(partial_records) == 7  # incremental save honored before completion

    resumed = prepare_data(cfg_resumed, resume=True)
    assert resumed.completed
    assert resumed.num_processed_this_run == TOTAL_SAMPLES - 7
    assert resumed.content_sha256 == fresh.content_sha256

    fresh_bytes = (manifests_dir(config_factory(tree_fresh)) / MANIFEST_FILENAME).read_bytes()
    resumed_bytes = (man_dir / MANIFEST_FILENAME).read_bytes()
    assert fresh_bytes == resumed_bytes


def test_resume_without_checkpoint_raises(app_config: AppConfig) -> None:
    with pytest.raises(ResumeStateError, match="no checkpoint"):
        prepare_data(app_config, resume=True)


def test_fresh_run_refuses_over_incomplete_run(app_config: AppConfig) -> None:
    partial = prepare_data(app_config, max_items=5)
    assert not partial.completed
    with pytest.raises(ResumeStateError, match="incomplete run"):
        prepare_data(app_config)


def test_resume_rejects_changed_config(
    synthetic_data_dir: Path, config_factory: ConfigFactory
) -> None:
    partial = prepare_data(config_factory(synthetic_data_dir), max_items=5)
    assert not partial.completed
    changed = config_factory(synthetic_data_dir, split={"val_fraction": 0.3, "seed": 42})
    with pytest.raises(ResumeStateError, match="config changed"):
        prepare_data(changed, resume=True)


def test_max_items_covering_everything_completes(app_config: AppConfig) -> None:
    result = prepare_data(app_config, max_items=TOTAL_SAMPLES)
    assert result.completed


def test_limit_per_class_bounds_processing(
    synthetic_data_dir: Path, config_factory: ConfigFactory
) -> None:
    cfg = config_factory(synthetic_data_dir, limit_per_class=2)
    result = prepare_data(cfg)
    assert result.completed
    assert result.num_total == 4 * (2 + 2)
    # 2 trainval per class -> round(2*0.2)=0 val
    assert result.counts_by_split == {"test": 8, "train": 8}


def test_swapped_trimap_semantics_fail_integrity(
    synthetic_data_dir: Path, config_factory: ConfigFactory
) -> None:
    trimaps = synthetic_data_dir / "oxford-iiit-pet" / "annotations" / "trimaps"
    for path in trimaps.glob("*.png"):
        write_trimap(path, swapped=True)
    with pytest.raises(DatasetIntegrityError, match="swapped"):
        prepare_data(config_factory(synthetic_data_dir))
