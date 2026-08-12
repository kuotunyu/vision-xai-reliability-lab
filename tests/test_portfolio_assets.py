from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from tools.portfolio_assets import (
    PortfolioAssetError,
    extract_portfolio_metrics,
    render_portfolio_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_extract_portfolio_metrics_uses_canonical_full_summary() -> None:
    summary = json.loads(
        (REPO_ROOT / "results" / "derived" / "summary.json").read_text(encoding="utf-8")
    )

    metrics = extract_portfolio_metrics(summary)

    assert metrics.center_pointing == pytest.approx(0.922)
    assert metrics.cnn_best_attribution_pointing == pytest.approx(0.864)
    assert metrics.vit_best_attribution_pointing == pytest.approx(0.626)
    assert metrics.cnn_ig_randomization == pytest.approx(0.480816)
    assert metrics.vit_ig_randomization == pytest.approx(0.471189)
    assert metrics.cnn_spurious_accuracy_spread == pytest.approx(0.012)
    assert metrics.vit_spurious_accuracy_spread == pytest.approx(0.0)
    assert metrics.attribution_subset == 500


def test_extract_portfolio_metrics_rejects_non_full_summary() -> None:
    with pytest.raises(PortfolioAssetError, match="canonical full summary"):
        extract_portfolio_metrics({"schema_version": 1, "experiment": "smoke"})


def test_render_portfolio_assets_writes_expected_pngs(tmp_path: Path) -> None:
    hero, social = render_portfolio_assets(REPO_ROOT, tmp_path)

    assert Image.open(hero).size == (1600, 900)
    assert Image.open(social).size == (1280, 640)
    assert hero.stat().st_size < 1024 * 1024
    assert social.stat().st_size < 1024 * 1024


def test_renderer_refuses_to_write_into_canonical_results() -> None:
    with pytest.raises(PortfolioAssetError, match="protected evidence tree"):
        render_portfolio_assets(REPO_ROOT, REPO_ROOT / "results" / "portfolio")
