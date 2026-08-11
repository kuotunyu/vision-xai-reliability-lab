"""FastAPI service: /health, /methods, /predict, /explain (+ optional /demo).

Models load lazily from checkpoints. When a checkpoint is missing, /predict and
/explain answer 503 with a clear message while /health stays 200 — the service
never fabricates a model.
"""

from __future__ import annotations

import base64
import logging
from typing import Annotated, Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import vision_xai
from app.service import WARNING_TEXT, InferenceService, ModelUnavailableError, load_image_bytes
from vision_xai.config import AppConfig
from vision_xai.errors import ExplainError, VisionXAIError
from vision_xai.models.factory import MODEL_NAMES, ModelName

logger = logging.getLogger(__name__)


def _validated_model(model: str) -> ModelName:
    if model not in MODEL_NAMES:
        raise HTTPException(
            status_code=400, detail=f"model must be one of {', '.join(MODEL_NAMES)}"
        )
    return model  # narrowed by the membership check


def create_app(cfg: AppConfig) -> FastAPI:
    application = FastAPI(
        title="vision-xai-reliability-lab",
        version=vision_xai.__version__,
        description="Attribution methods and their reliability, CNN vs ViT.",
    )
    service = InferenceService(cfg)
    application.state.service = service

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": vision_xai.__version__,
            "torch": torch.__version__,
            "experiment": cfg.experiment_name,
            "models_available": service.available_models(),
            "warning": WARNING_TEXT,
        }

    @application.get("/methods")
    def methods() -> dict[str, Any]:
        return {"methods": service.methods(), "warning": WARNING_TEXT}

    @application.post("/predict")
    def predict(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = cfg.serve.default_model,
    ) -> dict[str, Any]:
        model_name = _validated_model(model)
        try:
            image = load_image_bytes(file.file.read())
            payload = service.predict(image, model_name, cfg.serve.top_k)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VisionXAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {**payload, "warning": WARNING_TEXT}

    @application.post("/explain")
    def explain_endpoint(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = cfg.serve.default_model,
        method: Annotated[str, Form()] = "gradcam",
    ) -> dict[str, Any]:
        model_name = _validated_model(model)
        try:
            image = load_image_bytes(file.file.read())
            png, info = service.explain_image(image, model_name, method)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ExplainError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VisionXAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **info,
            "heatmap_png_base64": base64.b64encode(png).decode("ascii"),
            "warning": WARNING_TEXT,
        }

    _try_mount_demo(application, cfg)
    return application


def _try_mount_demo(application: FastAPI, cfg: AppConfig) -> None:
    try:
        import gradio as gr

        from app.demo import build_demo
    except ImportError:
        logger.info("gradio not installed — the API runs without the /demo UI")
        return
    try:
        gr.mount_gradio_app(application, build_demo(cfg), path="/demo")
        logger.info("Gradio demo mounted at /demo")
    except Exception:  # gradio version quirks must never take down the API
        logger.exception("failed to mount the Gradio demo; API continues without it")
