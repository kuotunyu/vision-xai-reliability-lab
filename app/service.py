"""Shared inference service used by both the FastAPI app and the Gradio demo."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vision_xai.config import AppConfig
from vision_xai.data.manifest import MANIFEST_FILENAME, load_manifest
from vision_xai.data.transforms import build_eval_transform, make_normalizer
from vision_xai.errors import ResumeStateError, TrainingError, VisionXAIError
from vision_xai.explain.base import (
    ALL_METHODS,
    METHOD_COMPATIBILITY,
    check_compatibility,
    explain,
)
from vision_xai.models.factory import IMAGENET_MEAN, IMAGENET_STD, MODEL_NAMES, ModelName
from vision_xai.models.loading import load_trained_model
from vision_xai.paths import manifests_dir, resolve_checkpoints_dir
from vision_xai.train.loop import CHECKPOINT_FILENAME

logger = logging.getLogger(__name__)

WARNING_TEXT = "A heatmap is not proof of causal reasoning."


class ModelUnavailableError(VisionXAIError):
    """The requested model has no usable checkpoint."""


class InferenceService:
    """Lazily loads trained models and exposes predict/explain for single images."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._bundles: dict[str, tuple[Any, torch.device]] = {}
        self._transform = build_eval_transform(cfg.data)
        self._normalize = make_normalizer(IMAGENET_MEAN, IMAGENET_STD)
        self._class_names: dict[int, str] | None = None

    def available_models(self) -> list[str]:
        ckpt_root = resolve_checkpoints_dir(self.cfg)
        return sorted(
            name for name in MODEL_NAMES if (ckpt_root / name / CHECKPOINT_FILENAME).is_file()
        )

    def methods(self) -> dict[str, list[str]]:
        return {method: sorted(METHOD_COMPATIBILITY[method]) for method in ALL_METHODS}

    def class_names(self) -> dict[int, str]:
        if self._class_names is None:
            self._class_names = {}
            manifest_path = manifests_dir(self.cfg) / MANIFEST_FILENAME
            if manifest_path.is_file():
                for record in load_manifest(manifest_path):
                    self._class_names[record.class_id] = record.class_name
            else:
                logger.info("manifest not found at %s; class names unavailable", manifest_path)
        return self._class_names

    def _bundle(self, model_name: ModelName) -> tuple[Any, torch.device]:
        if model_name not in self._bundles:
            # Both exceptions' underlying messages embed a local absolute
            # checkpoint path (useful in CLI/logs, not something an API client
            # should see — it leaks the server's working directory and, on
            # Windows, typically the local username). Log the detail
            # server-side and keep the client-facing message generic — but
            # the two conditions are factually different (no checkpoint file
            # vs. a checkpoint that exists but doesn't match this config), so
            # they get distinct messages rather than being collapsed into one.
            try:
                self._bundles[model_name] = load_trained_model(self.cfg, model_name)
            except TrainingError as exc:
                logger.info("model %r has no usable checkpoint: %s", model_name, exc)
                raise ModelUnavailableError(
                    f"model {model_name!r} has no checkpoint — train it first"
                ) from exc
            except ResumeStateError as exc:
                logger.info("model %r checkpoint does not match this config: %s", model_name, exc)
                raise ModelUnavailableError(
                    f"model {model_name!r} has a checkpoint, but it was trained under a "
                    "different config (experiment_name/seed/data) than this server is "
                    "currently running — align the configs, or retrain under this one"
                ) from exc
        return self._bundles[model_name]

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        return self._normalize(self._transform(image.convert("RGB"))).unsqueeze(0)

    def predict(self, image: Image.Image, model_name: ModelName, top_k: int) -> dict[str, Any]:
        bundle, device = self._bundle(model_name)
        batch = self._prepare(image).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(bundle.model(batch), dim=1)[0]
        k = min(top_k, probabilities.shape[0])
        values, indices = torch.topk(probabilities, k)
        names = self.class_names()
        return {
            "model": model_name,
            "top_k": [
                {
                    "class_id": int(index),
                    "class_name": names.get(int(index)),
                    "probability": round(float(value), 6),
                }
                for value, index in zip(values, indices, strict=True)
            ],
        }

    def explain_image(
        self, image: Image.Image, model_name: ModelName, method: str
    ) -> tuple[bytes, dict[str, Any]]:
        check_compatibility(method, model_name)
        bundle, device = self._bundle(model_name)
        batch = self._prepare(image).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(bundle.model(batch), dim=1)[0]
        prediction = int(probabilities.argmax())
        result = explain(
            bundle.model,
            batch,
            torch.tensor([prediction], device=device),
            method,
            self.cfg.explain,
            model_name,
            seed=self.cfg.seed,
            sample_ids=["upload"],
        )
        attribution = result.attributions[0].numpy()
        png = render_attribution_png(attribution, self._transform(image.convert("RGB")))
        names = self.class_names()
        info = {
            "model": model_name,
            "method": method,
            "explained_class_id": prediction,
            "explained_class_name": names.get(prediction),
            "probability": round(float(probabilities[prediction]), 6),
            "metadata": result.metadata,
        }
        return png, info


def render_attribution_png(
    attribution: np.ndarray, image_tensor: torch.Tensor | None = None
) -> bytes:
    """Render |attribution| as a colormapped PNG, optionally blended on the image.

    This normalization is *visualization only* — metric computations elsewhere
    always use the raw attribution values.
    """
    from matplotlib import colormaps

    magnitude = np.abs(attribution.astype(np.float64))
    peak = magnitude.max()
    normalized = magnitude / peak if peak > 0 else magnitude
    colored = (colormaps["magma"](normalized)[..., :3] * 255).astype(np.uint8)
    if image_tensor is not None:
        base = (image_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        colored = (0.5 * base + 0.5 * colored).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(colored, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def load_image_bytes(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        # DecompressionBombError subclasses Exception, not OSError — Pillow
        # raises it for a crafted header declaring an enormous pixel count,
        # before any real pixel data is decoded, so a tiny file can trigger it.
        raise VisionXAIError(f"could not decode the uploaded image: {exc}") from exc
    return image


def default_config_path() -> Path:
    return Path("configs") / "smoke.yaml"
