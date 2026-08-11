from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ConfigFactory
from vision_xai.config import AppConfig
from vision_xai.data.prepare import prepare_data
from vision_xai.errors import EvaluationError
from vision_xai.eval.runner import evaluate_all
from vision_xai.explain.runner import run_explanations
from vision_xai.paths import raw_eval_dir
from vision_xai.train.loop import train_model
from vision_xai.utils.io import read_jsonl, write_jsonl_atomic

N_EXPLAINED = 5


@pytest.fixture()
def explained_config(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AppConfig:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts"))
    cfg = config_factory(
        synthetic_data_dir,
        top_level={
            "train": {"epochs": 1, "batch_size": 8, "amp": False, "pretrained": False},
            "explain": {
                "split": "test",
                "num_samples": N_EXPLAINED,
                "batch_size": 4,
                "ig_steps": 4,
                "occlusion_window": 16,
                "occlusion_stride": 8,
            },
            "evaluate": {
                "deletion_steps": 4,
                "randomization_samples": 2,
                "consistency_samples": 2,
                "batch_size": 8,
            },
        },
    )
    assert prepare_data(cfg).completed
    train_model(cfg, "cnn")
    train_model(cfg, "cnn", patched=True)
    run_explanations(cfg, "cnn", "integrated_gradients")
    run_explanations(cfg, "cnn", "uniform")
    run_explanations(cfg, "cnn", "integrated_gradients", patched=True, test_variant="correlated")
    return cfg


def test_evaluate_all_families_and_incremental(explained_config: AppConfig) -> None:
    result = evaluate_all(explained_config)
    out = raw_eval_dir(explained_config)

    # localization + faithfulness cover both clean runs (ig + uniform), all samples
    assert result.rows_written["localization"] == 2 * N_EXPLAINED
    assert result.rows_written["faithfulness"] == 2 * N_EXPLAINED
    # randomization/consistency: real methods only (ig), capped at 2 samples
    assert result.rows_written["randomization"] == 2
    assert result.rows_written["consistency"] == 2
    # spurious: the patched-variant run, all samples
    assert result.rows_written["spurious"] == N_EXPLAINED

    loc_rows = read_jsonl(out / "cnn" / "integrated_gradients" / "localization.jsonl")
    assert len(loc_rows) == N_EXPLAINED
    expected_keys = {"sample_id", "correct", "energy_in_mask", "pointing_hit", "topk_iou_0.05"}
    assert expected_keys <= set(loc_rows[0])
    for row in loc_rows:
        assert 0.0 <= row["energy_in_mask"] <= 1.0

    faith_rows = read_jsonl(out / "cnn" / "integrated_gradients" / "faithfulness.jsonl")
    assert {"deletion_auc", "insertion_auc", "deletion_curve"} <= set(faith_rows[0])

    # randomization/consistency rows must carry 'correct' too, so report/build.py
    # can split them into correct-vs-all like every other family (hard project rule).
    rand_rows = read_jsonl(out / "cnn" / "integrated_gradients" / "randomization.jsonl")
    assert {"sample_id", "correct", "abs_spearman"} <= set(rand_rows[0])
    cons_rows = read_jsonl(out / "cnn" / "integrated_gradients" / "consistency.jsonl")
    assert {"sample_id", "correct", "abs_spearman"} <= set(cons_rows[0])

    spurious_rows = read_jsonl(
        out / "cnn_patched" / "integrated_gradients__correlated" / "spurious.jsonl"
    )
    assert len(spurious_rows) == N_EXPLAINED
    assert {"patch_energy", "pet_mask_energy", "patched_input", "correct"} <= set(spurious_rows[0])

    assert (out / "thresholds.json").is_file()

    # Second run is incremental: nothing new to write, no duplicate rows.
    rerun = evaluate_all(explained_config)
    assert all(count == 0 for count in rerun.rows_written.values())
    assert len(read_jsonl(out / "cnn" / "integrated_gradients" / "localization.jsonl")) == (
        N_EXPLAINED
    )


def test_evaluate_all_backfills_a_genuine_partial_family(
    explained_config: AppConfig,
) -> None:
    """Unlike the 'rerun after fully done' check above, this simulates a real
    partial state (some rows already written, some missing) by truncating one
    family's file, and confirms evaluate_all() backfills exactly the missing
    rows without duplicating or corrupting the ones already there."""
    evaluate_all(explained_config)
    out = raw_eval_dir(explained_config)
    loc_path = out / "cnn" / "integrated_gradients" / "localization.jsonl"
    full_rows = read_jsonl(loc_path)
    assert len(full_rows) == N_EXPLAINED

    kept_rows = full_rows[:2]
    write_jsonl_atomic(loc_path, kept_rows)

    result = evaluate_all(explained_config)
    assert result.rows_written["localization"] == N_EXPLAINED - 2

    backfilled_rows = read_jsonl(loc_path)
    assert len(backfilled_rows) == N_EXPLAINED
    assert len({row["sample_id"] for row in backfilled_rows}) == N_EXPLAINED
    # the 2 rows that were already there survive untouched, byte-for-byte
    assert backfilled_rows[:2] == kept_rows


def test_evaluate_without_explanations_raises(
    synthetic_data_dir: Path, config_factory: ConfigFactory
) -> None:
    cfg = config_factory(synthetic_data_dir)
    assert prepare_data(cfg).completed
    with pytest.raises(EvaluationError, match="no explanation runs"):
        evaluate_all(cfg)
