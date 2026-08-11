from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ConfigFactory
from vision_xai.config import AppConfig
from vision_xai.data.prepare import prepare_data
from vision_xai.errors import ReportError
from vision_xai.eval.runner import evaluate_all
from vision_xai.explain.runner import run_explanations
from vision_xai.report.aggregate import aggregate_rows, aggregate_values
from vision_xai.report.build import (
    RESULTS_BEGIN,
    RESULTS_END,
    build_report,
    build_summary,
    inject_results,
)
from vision_xai.train.loop import train_model
from vision_xai.utils.io import read_jsonl


def test_aggregate_values_known() -> None:
    result = aggregate_values([1.0, 2.0, 3.0, 4.0], seed=42)
    assert result["mean"] == pytest.approx(2.5)
    assert result["n"] == 4
    assert result["ci_low"] <= 2.5 <= result["ci_high"]
    # deterministic for a fixed seed
    assert aggregate_values([1.0, 2.0, 3.0, 4.0], seed=42) == result

    empty = aggregate_values([], seed=42)
    assert empty["mean"] is None and empty["n"] == 0
    single = aggregate_values([7.0], seed=42)
    assert single["mean"] == single["ci_low"] == single["ci_high"] == 7.0


def test_aggregate_rows_separates_correct_subset() -> None:
    rows = [
        {"metric": 1.0, "correct": True},
        {"metric": 0.0, "correct": False},
    ]
    result = aggregate_rows(rows, ["metric"], seed=42)
    assert result["all"]["metric"]["n"] == 2
    assert result["correct"]["metric"]["n"] == 1
    assert result["correct"]["metric"]["mean"] == pytest.approx(1.0)


def test_inject_results_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{RESULTS_BEGIN}\nold\n{RESULTS_END}\n\ntail\n", encoding="utf-8")
    inject_results(readme, "NEW BLOCK")
    first = readme.read_text(encoding="utf-8")
    assert "NEW BLOCK" in first and "old" not in first
    assert first.count(RESULTS_BEGIN) == 1 and first.count(RESULTS_END) == 1
    inject_results(readme, "NEW BLOCK")
    assert readme.read_text(encoding="utf-8") == first  # idempotent
    assert first.endswith("tail\n")

    bare = tmp_path / "bare.md"
    bare.write_text("no markers", encoding="utf-8")
    with pytest.raises(ReportError, match="markers"):
        inject_results(bare, "X")


@pytest.fixture()
def evaluated_config(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AppConfig:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts"))
    monkeypatch.chdir(tmp_path)  # assets/figures land in tmp, not the repo
    cfg = config_factory(
        synthetic_data_dir,
        top_level={
            "train": {"epochs": 1, "batch_size": 8, "amp": False, "pretrained": False},
            "explain": {"split": "test", "num_samples": 4, "batch_size": 4, "ig_steps": 4},
            "evaluate": {
                "deletion_steps": 4,
                "randomization_samples": 2,
                "consistency_samples": 2,
            },
        },
    )
    assert prepare_data(cfg).completed
    train_model(cfg, "cnn")
    train_model(cfg, "cnn", patched=True)
    run_explanations(cfg, "cnn", "integrated_gradients")
    run_explanations(cfg, "cnn", "integrated_gradients", patched=True, test_variant="correlated")
    evaluate_all(cfg)
    return cfg


def test_build_report_end_to_end(evaluated_config: AppConfig, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# t\n{RESULTS_BEGIN}\n{RESULTS_END}\n", encoding="utf-8")
    result = build_report(evaluated_config, readme_paths=[readme])

    assert result.summary_path.is_file()
    summary = build_summary(evaluated_config)
    assert summary["localization"]["cnn"]["integrated_gradients"]["all"]["energy_in_mask"]["n"] == 4
    assert "cnn_patched" in summary["spurious"]
    assert summary["train"]["cnn"]["val_accuracy"] is not None
    assert summary["train"]["cnn"]["val_ece"] is not None
    assert "val ECE" in result.summary_path.parent.joinpath("summary.md").read_text(
        encoding="utf-8"
    )
    assert "not causal faithfulness" in " ".join(summary["notes"])
    assert "reported separately" in " ".join(summary["notes"])

    # Every attribution-derived family must expose a genuine correct/all split
    # (randomization, consistency, and spurious used to be flat — regression guard).
    rand_payload = summary["randomization"]["cnn"]["integrated_gradients"]
    assert {"all", "correct"} <= set(rand_payload)
    assert "abs_spearman" in rand_payload["all"]
    cons_payload = summary["consistency"]["cnn"]["integrated_gradients"]
    assert {"all", "correct"} <= set(cons_payload)
    spurious_payload = summary["spurious"]["cnn_patched"]["integrated_gradients"]["correlated"]
    assert {"all", "correct"} <= set(spurious_payload["patch_energy_patched_inputs"])
    assert {"all", "correct"} <= set(spurious_payload["pet_mask_energy"])

    injected = readme.read_text(encoding="utf-8")
    assert "Localization" in injected
    assert "A heatmap is not proof of causal reasoning." in injected
    assert result.figure_paths  # at least one figure produced
    for figure in result.figure_paths:
        assert figure.is_file()
    assert (result.summary_path.parent / "summary.md").is_file()


def test_build_summary_without_eval_raises(
    synthetic_data_dir: Path, config_factory: ConfigFactory
) -> None:
    cfg = config_factory(synthetic_data_dir)
    with pytest.raises(ReportError, match="no evaluation results"):
        build_summary(cfg)


def test_summary_uses_only_raw_rows(evaluated_config: AppConfig) -> None:
    """The aggregated energy mean must equal the mean of the raw per-sample rows."""
    from vision_xai.paths import raw_eval_dir

    rows = read_jsonl(
        raw_eval_dir(evaluated_config) / "cnn" / "integrated_gradients" / "localization.jsonl"
    )
    raw_mean = sum(row["energy_in_mask"] for row in rows) / len(rows)
    summary = build_summary(evaluated_config)
    reported = summary["localization"]["cnn"]["integrated_gradients"]["all"]["energy_in_mask"]
    assert reported["mean"] == pytest.approx(raw_mean, abs=1e-6)
