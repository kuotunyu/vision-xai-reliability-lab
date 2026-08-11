"""Load a trained (head-finetuned) model from its checkpoint for inference."""

from __future__ import annotations

import logging

import torch

from vision_xai.config import AppConfig, config_hash
from vision_xai.models.factory import ModelBundle, ModelName, build_model, head_module

logger = logging.getLogger(__name__)


def variant_name(model_name: ModelName, patched: bool) -> str:
    return f"{model_name}_patched" if patched else model_name


def load_trained_model(
    cfg: AppConfig,
    model_name: ModelName,
    *,
    patched: bool = False,
    device: str | None = None,
) -> tuple[ModelBundle, torch.device]:
    """Rebuild the backbone and load the trained head; returns an eval-mode model."""
    from vision_xai.paths import resolve_checkpoints_dir
    from vision_xai.train.loop import CHECKPOINT_FILENAME, load_head_checkpoint

    variant = variant_name(model_name, patched)
    ckpt_path = resolve_checkpoints_dir(cfg) / variant / CHECKPOINT_FILENAME
    payload = load_head_checkpoint(ckpt_path, expected_config_hash=config_hash(cfg))
    bundle = build_model(
        model_name,
        num_classes=int(payload["num_classes"]),
        pretrained=cfg.train.pretrained,
        head_only=True,
    )
    head_module(bundle.model, model_name).load_state_dict(payload["head_state"])
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle.model.to(resolved_device).eval()
    logger.info(
        "loaded %s from %s (epochs_completed=%s)",
        variant,
        ckpt_path,
        payload["epochs_completed"],
    )
    return bundle, resolved_device
