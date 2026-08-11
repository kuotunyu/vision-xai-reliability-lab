"""Deterministic stratified train/val split of the official trainval samples."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Literal

from vision_xai.utils.seed import per_sample_rng

logger = logging.getLogger(__name__)


def stratified_train_val_split(
    labels_by_id: Mapping[str, int], val_fraction: float, seed: int
) -> dict[str, Literal["train", "val"]]:
    """Assign each sample id to train or val, stratified per class.

    Deterministic and independent of the input mapping's iteration order: ids are
    sorted per class and each class draws from its own hash-derived RNG. Per class,
    ``round(n * val_fraction)`` ids go to val (clamped so at least one train sample
    remains whenever the class has more than one).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    by_class: dict[int, list[str]] = defaultdict(list)
    for sample_id in sorted(labels_by_id):
        by_class[labels_by_id[sample_id]].append(sample_id)

    assignment: dict[str, Literal["train", "val"]] = {}
    for class_id in sorted(by_class):
        ids = by_class[class_id]
        n_val = round(len(ids) * val_fraction)
        if len(ids) > 1:
            n_val = min(n_val, len(ids) - 1)
        rng = per_sample_rng(seed, f"class:{class_id}", "split")
        val_positions = set(rng.permutation(len(ids))[:n_val].tolist())
        for position, sample_id in enumerate(ids):
            assignment[sample_id] = "val" if position in val_positions else "train"
        logger.debug("class %d: %d train / %d val", class_id, len(ids) - n_val, n_val)
    return assignment
