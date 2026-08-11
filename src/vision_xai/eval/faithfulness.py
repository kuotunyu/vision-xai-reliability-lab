"""Deletion / insertion faithfulness curves and their AUCs.

Pixels are removed (or inserted) in descending |attribution| order and replaced
with zeros — which, in normalized input space, is the dataset per-channel mean.
The tracked score is the softmax probability of the explained target class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


@dataclass(frozen=True)
class PerturbationResult:
    fractions: list[float]
    deletion_curve: list[float]
    insertion_curve: list[float]
    deletion_auc: float
    insertion_auc: float


def perturbation_curves(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    attribution: npt.NDArray[np.floating],
    *,
    steps: int,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> PerturbationResult:
    """Compute deletion and insertion curves for one already-normalized CHW image."""
    chw_ndim = 3
    if image.ndim != chw_ndim:
        raise ValueError(f"expected a CHW image, got shape {tuple(image.shape)}")
    if image.shape[-2:] != attribution.shape:
        raise ValueError(
            f"attribution {attribution.shape} does not match image {tuple(image.shape[-2:])}"
        )
    resolved_device = device or torch.device("cpu")
    num_pixels = attribution.size
    order = np.argsort(-np.abs(attribution).ravel(), kind="stable")
    fractions = np.linspace(0.0, 1.0, steps + 1)
    counts = np.round(fractions * num_pixels).astype(int)

    flat = image.reshape(image.shape[0], -1)
    perturbed: list[torch.Tensor] = []
    for count in counts:
        keep = torch.from_numpy(order[:count].copy()).long()
        deleted = flat.clone()
        deleted[:, keep] = 0.0
        inserted = torch.zeros_like(flat)
        inserted[:, keep] = flat[:, keep]
        perturbed.append(deleted.reshape(image.shape))
        perturbed.append(inserted.reshape(image.shape))

    stacked = torch.stack(perturbed)
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, stacked.shape[0], batch_size):
            chunk = stacked[start : start + batch_size].to(resolved_device)
            softmax = torch.softmax(model(chunk), dim=1)
            probabilities.extend(float(p) for p in softmax[:, target].cpu())

    deletion_curve = probabilities[0::2]
    insertion_curve = probabilities[1::2]
    deletion_auc = float(np.trapezoid(deletion_curve, fractions))
    insertion_auc = float(np.trapezoid(insertion_curve, fractions))
    return PerturbationResult(
        fractions=[round(float(f), 6) for f in fractions],
        deletion_curve=[round(p, 6) for p in deletion_curve],
        insertion_curve=[round(p, 6) for p in insertion_curve],
        deletion_auc=round(deletion_auc, 6),
        insertion_auc=round(insertion_auc, 6),
    )
