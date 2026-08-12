from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from app.api import configure_gradio_environment
from app.demo import build_demo, model_evidence_outputs
from app.evidence import load_evidence_dashboard

from conftest import ConfigFactory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config_text(demo: object) -> str:
    config = demo.get_config_file()  # type: ignore[attr-defined]
    return json.dumps(config, ensure_ascii=False)


def test_demo_has_exactly_two_zh_tw_tabs(
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
) -> None:
    cfg = config_factory(synthetic_data_dir)
    demo = build_demo(cfg, evidence_root=REPO_ROOT)
    config = demo.get_config_file()
    labels = [
        component.get("props", {}).get("label")
        for component in config["components"]
        if component.get("type") == "tabitem"
    ]

    assert labels == ["實驗證據", "本機模型"]
    text = json.dumps(config, ensure_ascii=False)
    assert "Precomputed explorer" not in text
    assert "500-sample attribution subset" in text
    assert "localization 不是 causal faithfulness" in text


def test_model_evidence_outputs_switch_models() -> None:
    dashboard = load_evidence_dashboard(REPO_ROOT)

    cnn = model_evidence_outputs(dashboard, "cnn")
    vit = model_evidence_outputs(dashboard, "vit")

    assert "86.4%" in cnn[0] and "Grad-CAM" in cnn[0]
    assert cnn[1].name == "localization_cnn.png"
    assert cnn[2].name == "faithfulness_cnn.png"
    assert cnn[3].name == "spurious_cnn_patched.png"
    assert "62.6%" in vit[0] and "Occlusion" in vit[0]
    assert vit[1].name == "localization_vit.png"


def test_missing_evidence_renders_error_without_breaking_demo(
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
    tmp_path: Path,
) -> None:
    cfg = config_factory(synthetic_data_dir)

    text = _config_text(build_demo(cfg, evidence_root=tmp_path))

    assert "公開證據無法載入" in text
    assert "請執行 release verifier" in text
    assert str(tmp_path) not in text
    assert "本機模型" in text


def test_demo_without_checkpoints_has_compact_readiness_state(
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "no-checkpoints"))
    cfg = config_factory(synthetic_data_dir)

    text = _config_text(build_demo(cfg, evidence_root=REPO_ROOT))

    assert "尚未偵測到本機 checkpoint" in text
    assert "實驗證據不受影響" in text


def test_gradio_analytics_default_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)

    configure_gradio_environment()

    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_gradio_analytics_explicit_setting_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")

    configure_gradio_environment()

    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "True"
