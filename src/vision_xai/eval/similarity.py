"""Rank-based similarity between attribution maps (no scipy dependency)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def tie_averaged_ranks(values: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Ranks with ties assigned their average rank (Spearman convention)."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    mean_ranks = (starts + ends - 1) / 2.0
    result: npt.NDArray[np.float64] = mean_ranks[inverse]
    return result


def abs_spearman(first: npt.NDArray[np.floating], second: npt.NDArray[np.floating]) -> float:
    """|Spearman rank correlation| between |first| and |second| (flattened).

    Returns 0.0 when either map is constant (correlation undefined).
    """
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {first.shape} vs {second.shape}")
    ranks_a = tie_averaged_ranks(np.abs(first.astype(np.float64)).ravel())
    ranks_b = tie_averaged_ranks(np.abs(second.astype(np.float64)).ravel())
    std_a = ranks_a.std()
    std_b = ranks_b.std()
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    covariance = float(np.mean((ranks_a - ranks_a.mean()) * (ranks_b - ranks_b.mean())))
    return float(abs(covariance / (std_a * std_b)))
