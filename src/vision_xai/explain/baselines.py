"""Reference attribution baselines: random, uniform, and center prior.

The random baseline is deterministic per ``(seed, sample_id)`` so repeated runs
and resumed runs produce identical maps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from vision_xai.errors import ExplainError
from vision_xai.utils.seed import per_sample_rng

CENTER_PRIOR_SIGMA_FRACTION = 0.25


def _random_map(height: int, width: int, sample_id: str, seed: int) -> np.ndarray:
    rng = per_sample_rng(seed, sample_id, "baseline:random")
    return rng.random((height, width), dtype=np.float64)


def _uniform_map(height: int, width: int) -> np.ndarray:
    return np.full((height, width), 1.0 / (height * width))


def _center_prior_map(height: int, width: int) -> np.ndarray:
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    sigma = CENTER_PRIOR_SIGMA_FRACTION * min(height, width)
    squared_distance = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
    result: np.ndarray = np.exp(-squared_distance / (2.0 * sigma**2))
    return result


def baseline_attributions(
    method: str,
    batch_size: int,
    height: int,
    width: int,
    sample_ids: Sequence[str],
    seed: int,
) -> torch.Tensor:
    maps: list[np.ndarray] = []
    for index in range(batch_size):
        if method == "random":
            maps.append(_random_map(height, width, sample_ids[index], seed))
        elif method == "uniform":
            maps.append(_uniform_map(height, width))
        elif method == "center":
            maps.append(_center_prior_map(height, width))
        else:
            raise ExplainError(f"unknown baseline method {method!r}")
    return torch.from_numpy(np.stack(maps)).float()
