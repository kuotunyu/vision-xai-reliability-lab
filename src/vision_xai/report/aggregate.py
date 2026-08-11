"""Aggregation of raw per-sample eval rows into summary statistics.

Every reported number is a mean with a seeded bootstrap 95% CI — no value is
ever entered by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 1000


def aggregate_values(values: Sequence[float], seed: int) -> dict[str, Any]:
    """Mean, seeded bootstrap 95% CI, and sample count for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if n == 1:
        return {"mean": round(mean, 6), "ci_low": round(mean, 6), "ci_high": round(mean, 6), "n": 1}
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    boot_means = array[resamples].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": round(mean, 6),
        "ci_low": round(float(ci_low), 6),
        "ci_high": round(float(ci_high), 6),
        "n": n,
    }


def aggregate_rows(
    rows: Sequence[dict[str, Any]], metric_keys: Sequence[str], seed: int
) -> dict[str, dict[str, Any]]:
    """Aggregate the given metric keys over ``all`` rows and the ``correct`` subset.

    Correctly-predicted and all-prediction subsets are always reported separately.
    """
    subsets = {
        "all": list(rows),
        "correct": [row for row in rows if row.get("correct", False)],
    }
    result: dict[str, dict[str, Any]] = {}
    for subset_name, subset_rows in subsets.items():
        metrics: dict[str, Any] = {}
        for key in metric_keys:
            values = [float(row[key]) for row in subset_rows if key in row]
            metrics[key] = aggregate_values(values, seed)
        result[subset_name] = metrics
    return result
