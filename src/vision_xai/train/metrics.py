"""Classification metrics implemented on plain tensors (no sklearn dependency)."""

from __future__ import annotations

import itertools

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct predictions; 0.0 on empty input."""
    if predictions.numel() == 0:
        return 0.0
    return float((predictions == targets).float().mean())


def macro_f1(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Unweighted mean of per-class F1 over all ``num_classes`` classes.

    Classes with no true and no predicted samples contribute 0 (sklearn's
    ``zero_division=0`` convention).
    """
    if predictions.numel() == 0:
        return 0.0
    f1_sum = 0.0
    for class_id in range(num_classes):
        predicted = predictions == class_id
        actual = targets == class_id
        true_positive = float((predicted & actual).sum())
        denominator = (
            2 * true_positive
            + float((predicted & ~actual).sum())
            + float((~predicted & actual).sum())
        )
        f1_sum += (2 * true_positive / denominator) if denominator > 0 else 0.0
    return f1_sum / num_classes


def expected_calibration_error(
    confidences: torch.Tensor, correct: torch.Tensor, num_bins: int = 10
) -> float:
    """Binned ECE (Guo et al. 2017): weighted mean |bin accuracy - bin confidence|.

    ``confidences`` is the predicted class's softmax probability per sample;
    ``correct`` is whether that prediction matched the target. Bins are
    equal-width over [0, 1]; empty bins contribute nothing (0-weighted).
    """
    if confidences.numel() == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, num_bins + 1)
    total = confidences.numel()
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        in_bin = (
            (confidences > lo) & (confidences <= hi)
            if lo > 0
            else (confidences >= lo) & (confidences <= hi)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct[in_bin].float().mean())
        bin_confidence = float(confidences[in_bin].mean())
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
    return ece
