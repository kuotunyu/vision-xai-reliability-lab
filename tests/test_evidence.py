from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from app.evidence import EvidenceError, load_evidence_dashboard

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_public_evidence(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    summary = root / "results" / "derived" / "summary.json"
    summary.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "results" / "derived" / "summary.json", summary)
    canary = root / "release" / "cuda-resume-canary.json"
    canary.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "release" / "cuda-resume-canary.json", canary)
    shutil.copytree(REPO_ROOT / "assets" / "figures", root / "assets" / "figures")
    return root


def test_full_evidence_extracts_canonical_headlines() -> None:
    dashboard = load_evidence_dashboard(REPO_ROOT)

    assert dashboard.center_pointing == 0.922
    assert dashboard.attribution_samples == 500
    assert dashboard.spurious_patch_energy_max == pytest.approx(0.012622)
    assert dashboard.model("cnn").best_method == "Grad-CAM"
    assert dashboard.model("cnn").best_pointing == 0.864
    assert dashboard.model("vit").best_method == "Occlusion"
    assert dashboard.model("vit").best_pointing == 0.626
    assert dashboard.model("cnn").ig_randomization == pytest.approx(0.480816)
    assert dashboard.model("vit").ig_randomization == pytest.approx(0.471189)


def test_full_evidence_exposes_safe_figures_and_canary_scope() -> None:
    dashboard = load_evidence_dashboard(REPO_ROOT)

    assert dashboard.model("cnn").localization_figure == (
        REPO_ROOT / "assets" / "figures" / "localization_cnn.png"
    )
    assert dashboard.model("vit").spurious_figure == (
        REPO_ROOT / "assets" / "figures" / "spurious_vit_patched.png"
    )
    assert dashboard.canary_exact_states == (
        "final head",
        "optimizer",
        "GradScaler",
        "stable metrics",
    )
    assert dashboard.canary_scheduler_status == "not_applicable"
    assert dashboard.canary_not_full_scale is True
    assert dashboard.canary_checkpoint_hash_equal is True
    assert dashboard.canary_gpu == "NVIDIA GeForce RTX 4090"
    assert dashboard.canary_torch_version == "2.11.0+cu130"


def test_non_full_summary_fails_closed(tmp_path: Path) -> None:
    root = _copy_public_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = "smoke"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(EvidenceError, match="canonical full"):
        load_evidence_dashboard(root)


def test_missing_figure_fails_without_leaking_machine_path(tmp_path: Path) -> None:
    root = _copy_public_evidence(tmp_path)
    missing = root / "assets" / "figures" / "faithfulness_vit.png"
    missing.unlink()

    with pytest.raises(EvidenceError) as exc_info:
        load_evidence_dashboard(root)

    message = str(exc_info.value)
    assert "assets/figures/faithfulness_vit.png" in message
    assert str(tmp_path) not in message


def test_missing_required_metric_fails_closed(tmp_path: Path) -> None:
    root = _copy_public_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["randomization"]["cnn"]["integrated_gradients"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(EvidenceError, match=r"results/derived/summary\.json"):
        load_evidence_dashboard(root)
