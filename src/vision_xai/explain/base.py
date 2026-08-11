"""Unified explanation interface.

``explain(model, images, targets, method, ...)`` returns per-pixel attribution
maps of shape (N, H, W). Signed values are preserved as computed by each method;
metric-side handling (e.g. absolute values) is an evaluation decision, and
visualization normalization is separate from metric values by design.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vision_xai.config import ExplainConfig
from vision_xai.errors import ExplainError
from vision_xai.models.factory import ModelName

logger = logging.getLogger(__name__)

REAL_METHODS = ("gradcam", "integrated_gradients", "occlusion")
BASELINE_METHODS = ("random", "uniform", "center")
ALL_METHODS = REAL_METHODS + BASELINE_METHODS

METHOD_COMPATIBILITY: dict[str, frozenset[str]] = {
    "gradcam": frozenset({"cnn"}),
    "integrated_gradients": frozenset({"cnn", "vit"}),
    "occlusion": frozenset({"cnn", "vit"}),
    "random": frozenset({"cnn", "vit"}),
    "uniform": frozenset({"cnn", "vit"}),
    "center": frozenset({"cnn", "vit"}),
}


@dataclass(frozen=True)
class AttributionBatch:
    attributions: torch.Tensor  # (N, H, W), float32, on CPU
    method: str
    metadata: dict[str, Any]


def check_compatibility(method: str, model_name: ModelName) -> None:
    if method not in METHOD_COMPATIBILITY:
        raise ExplainError(f"unknown method {method!r}; expected one of {ALL_METHODS}")
    if model_name not in METHOD_COMPATIBILITY[method]:
        raise ExplainError(
            f"method {method!r} is not compatible with model {model_name!r} "
            f"(supported: {sorted(METHOD_COMPATIBILITY[method])})"
        )


def explain(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    explain_cfg: ExplainConfig,
    model_name: ModelName,
    *,
    seed: int = 42,
    sample_ids: Sequence[str] | None = None,
) -> AttributionBatch:
    """Compute attributions for a batch; images must already be model inputs."""
    check_compatibility(method, model_name)
    nchw_ndim = 4
    if images.ndim != nchw_ndim:
        raise ExplainError(f"expected NCHW images, got shape {tuple(images.shape)}")
    metadata: dict[str, Any] = {"aggregation": "channel_sum"}

    if method == "gradcam":
        from vision_xai.explain.gradcam import gradcam_attributions

        attributions = gradcam_attributions(model, images, targets)
        metadata["aggregation"] = "gradcam_layer"
    elif method == "integrated_gradients":
        from vision_xai.explain.integrated_gradients import ig_attributions

        attributions = ig_attributions(model, images, targets, n_steps=explain_cfg.ig_steps)
        metadata["n_steps"] = explain_cfg.ig_steps
        metadata["baseline"] = "zeros (the normalized-space dataset mean)"
    elif method == "occlusion":
        from vision_xai.explain.occlusion import occlusion_attributions

        attributions = occlusion_attributions(
            model,
            images,
            targets,
            window=explain_cfg.occlusion_window,
            stride=explain_cfg.occlusion_stride,
        )
        metadata["window"] = explain_cfg.occlusion_window
        metadata["stride"] = explain_cfg.occlusion_stride
    else:
        from vision_xai.explain.baselines import baseline_attributions

        if sample_ids is None or len(sample_ids) != images.shape[0]:
            raise ExplainError("baseline methods require one sample_id per image")
        attributions = baseline_attributions(
            method, images.shape[0], images.shape[-2], images.shape[-1], sample_ids, seed
        )
        metadata["aggregation"] = "baseline"

    return AttributionBatch(
        attributions=attributions.detach().float().cpu(),
        method=method,
        metadata=metadata,
    )
