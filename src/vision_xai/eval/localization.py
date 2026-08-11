"""Localization metrics: where attribution mass lands relative to a target mask.

All metrics operate on the absolute attribution values (metric values are kept
separate from any visualization normalization). Localization measures *where*
attributions land — it is reported as localization, never as causal
faithfulness.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate(attribution: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]) -> None:
    if attribution.shape != mask.shape:
        raise ValueError(f"shape mismatch: attribution {attribution.shape} vs mask {mask.shape}")


def energy_in_mask(attribution: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]) -> float:
    """Fraction of total |attribution| mass inside the mask (0 when the map is empty)."""
    _validate(attribution, mask)
    magnitude = np.abs(attribution.astype(np.float64))
    total = float(magnitude.sum())
    if total == 0.0:
        return 0.0
    return float(magnitude[mask].sum() / total)


def pointing_game_hit(attribution: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]) -> bool:
    """True when the single highest-|attribution| pixel falls inside the mask."""
    _validate(attribution, mask)
    if not mask.any():
        return False
    flat_index = int(np.argmax(np.abs(attribution)))
    return bool(mask.flat[flat_index])


def topk_iou(
    attribution: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    fraction: float,
) -> float:
    """IoU between the top-``fraction`` |attribution| pixel set and the mask."""
    _validate(attribution, mask)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not mask.any():
        return 0.0
    k = max(1, round(fraction * attribution.size))
    magnitude = np.abs(attribution).ravel()
    top_indices = np.argpartition(-magnitude, k - 1)[:k]
    top_set = np.zeros(attribution.size, dtype=bool)
    top_set[top_indices] = True
    mask_flat = mask.ravel()
    intersection = int(np.logical_and(top_set, mask_flat).sum())
    union = int(np.logical_or(top_set, mask_flat).sum())
    return intersection / union if union else 0.0
