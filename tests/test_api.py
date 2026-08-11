from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from app.api import create_app
from fastapi.testclient import TestClient
from PIL import Image

from conftest import ConfigFactory
from vision_xai.data.prepare import prepare_data
from vision_xai.train.loop import train_model

PNG_MAGIC = b"\x89PNG"


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 60, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def client(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> TestClient:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts"))
    cfg = config_factory(
        synthetic_data_dir,
        top_level={
            "train": {"epochs": 1, "batch_size": 8, "amp": False, "pretrained": False},
            "explain": {"ig_steps": 4, "occlusion_window": 16, "occlusion_stride": 8},
        },
    )
    assert prepare_data(cfg).completed
    train_model(cfg, "cnn")  # only cnn — vit stays unavailable on purpose
    return TestClient(create_app(cfg))


def test_health_reports_available_models(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models_available"] == ["cnn"]
    assert "causal" in payload["warning"]


def test_methods_compatibility_map(client: TestClient) -> None:
    response = client.get("/methods")
    assert response.status_code == 200
    methods = response.json()["methods"]
    assert methods["gradcam"] == ["cnn"]
    assert set(methods["integrated_gradients"]) == {"cnn", "vit"}


def test_predict_returns_topk(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"model": "cnn"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    top_k = payload["top_k"]
    assert len(top_k) == 4  # synthetic dataset has 4 classes
    probabilities = [entry["probability"] for entry in top_k]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1.0 + 1e-6
    assert payload["warning"]


def test_predict_without_checkpoint_is_503(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"model": "vit"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "checkpoint" in detail
    # The underlying error embeds a local absolute path (useful in server logs,
    # not something to hand to an API client) — must not leak into the response.
    assert ":\\" not in detail and ":/" not in detail and "/home/" not in detail


def test_predict_unknown_model_is_400(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"model": "resnet"},
    )
    assert response.status_code == 400


def test_explain_returns_png_heatmap(client: TestClient) -> None:
    response = client.post(
        "/explain",
        files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"model": "cnn", "method": "gradcam"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    png = base64.b64decode(payload["heatmap_png_base64"])
    assert png.startswith(PNG_MAGIC)
    assert payload["method"] == "gradcam"
    assert payload["explained_class_id"] in range(4)
    assert payload["warning"]


def test_explain_incompatible_method_is_400(client: TestClient) -> None:
    response = client.post(
        "/explain",
        files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"model": "vit", "method": "gradcam"},
    )
    assert response.status_code == 400
    assert "not compatible" in response.json()["detail"]


def test_explain_undecodable_image_is_400(client: TestClient) -> None:
    response = client.post(
        "/explain",
        files={"file": ("pet.jpg", b"not an image", "image/jpeg")},
        data={"model": "cnn", "method": "gradcam"},
    )
    assert response.status_code == 400


def test_predict_decompression_bomb_is_400_not_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PIL.Image.DecompressionBombError does not subclass OSError; a naive
    `except OSError` around Image.open() would let it escape as a raw 500."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # our 64x64 fixture now "explodes"
    response = client.post(
        "/predict",
        files={"file": ("pet.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"model": "cnn"},
    )
    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]
