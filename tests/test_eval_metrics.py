from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from vision_xai.eval.faithfulness import perturbation_curves
from vision_xai.eval.localization import energy_in_mask, pointing_game_hit, topk_iou
from vision_xai.eval.randomization import randomize_head
from vision_xai.eval.similarity import abs_spearman, tie_averaged_ranks
from vision_xai.models.factory import build_model, head_module


def test_energy_in_mask_exact() -> None:
    attribution = np.array([[1.0, 1.0], [1.0, 1.0]])
    mask = np.array([[True, False], [False, False]])
    assert energy_in_mask(attribution, mask) == pytest.approx(0.25)
    signed = np.array([[-2.0, 0.0], [1.0, 1.0]])
    assert energy_in_mask(signed, mask) == pytest.approx(0.5)  # uses |values|
    assert energy_in_mask(np.zeros((2, 2)), mask) == 0.0


def test_pointing_game_exact() -> None:
    attribution = np.array([[0.1, 0.9], [0.2, 0.3]])
    assert pointing_game_hit(attribution, np.array([[False, True], [False, False]]))
    assert not pointing_game_hit(attribution, np.array([[True, False], [False, False]]))
    assert not pointing_game_hit(attribution, np.zeros((2, 2), dtype=bool))
    negative = np.array([[-5.0, 0.9], [0.2, 0.3]])  # |-5| dominates
    assert pointing_game_hit(negative, np.array([[True, False], [False, False]]))


def test_topk_iou_exact() -> None:
    attribution = np.arange(16, dtype=float).reshape(4, 4)
    top4 = np.zeros((4, 4), dtype=bool)
    top4.flat[[15, 14, 13, 12]] = True
    assert topk_iou(attribution, top4, 0.25) == pytest.approx(1.0)
    disjoint = np.zeros((4, 4), dtype=bool)
    disjoint.flat[[0, 1, 2, 3]] = True
    # intersection 0, union 8
    assert topk_iou(attribution, disjoint, 0.25) == pytest.approx(0.0)
    assert topk_iou(attribution, np.zeros((4, 4), dtype=bool), 0.25) == 0.0
    with pytest.raises(ValueError, match="fraction"):
        topk_iou(attribution, top4, 0.0)


def test_tie_averaged_ranks() -> None:
    ranks = tie_averaged_ranks(np.array([1.0, 1.0, 2.0]))
    assert ranks.tolist() == [0.5, 0.5, 2.0]


def test_abs_spearman_known_values() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs_spearman(a, a * 10.0) == pytest.approx(1.0)
    assert abs_spearman(a, a[::-1].copy()) == pytest.approx(1.0)  # |rho| of reversal
    assert abs_spearman(a, np.full(4, 7.0)) == 0.0  # constant map -> undefined -> 0


class _MeanLogitModel(nn.Module):
    """Logit 0 = mean of channel 0; logit 1 = 0 — probability is monotone in
    how much of channel 0 remains."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = x[:, 0].flatten(1).mean(dim=1, keepdim=True)
        return torch.cat([first, torch.zeros_like(first)], dim=1)


def test_perturbation_curves_monotone_model() -> None:
    torch.manual_seed(0)
    image = torch.rand(3, 8, 8) * 4.0
    attribution = image[0].numpy()  # attribute exactly what drives the logit
    result = perturbation_curves(
        _MeanLogitModel(), image, target=0, attribution=attribution, steps=8
    )
    # Deleting everything -> all-zero input -> softmax([0,0]) = 0.5
    assert result.deletion_curve[-1] == pytest.approx(0.5, abs=1e-6)
    # Inserting everything reproduces the original prediction
    assert result.insertion_curve[-1] == pytest.approx(result.deletion_curve[0], abs=1e-6)
    # A faithful ranking deletes probability quickly and restores it quickly
    assert result.deletion_auc < result.insertion_auc
    # Deletion curve decreases monotonically for this exactly-faithful attribution
    assert all(
        later <= earlier + 1e-6
        for earlier, later in zip(result.deletion_curve, result.deletion_curve[1:], strict=False)
    )


def test_perturbation_curves_shape_validation() -> None:
    with pytest.raises(ValueError, match="CHW"):
        perturbation_curves(_MeanLogitModel(), torch.rand(1, 3, 8, 8), 0, np.zeros((8, 8)), steps=4)
    with pytest.raises(ValueError, match="does not match"):
        perturbation_curves(_MeanLogitModel(), torch.rand(3, 8, 8), 0, np.zeros((4, 4)), steps=4)


def test_randomize_head_is_seeded_and_changes_weights() -> None:
    bundle = build_model("cnn", pretrained=False)
    original = [p.clone() for p in head_module(bundle.model, "cnn").parameters()]
    randomize_head(bundle.model, "cnn", seed=7)
    changed = list(head_module(bundle.model, "cnn").parameters())
    assert any(not torch.equal(a, b) for a, b in zip(original, changed, strict=True))

    other = build_model("cnn", pretrained=False)
    randomize_head(other.model, "cnn", seed=7)
    for a, b in zip(changed, head_module(other.model, "cnn").parameters(), strict=True):
        assert torch.equal(a, b)  # same seed -> identical randomization


def test_randomize_head_never_draws_into_the_parameter_itself() -> None:
    """Values must be drawn into a CPU scratch tensor and copied in.

    ``torch.Generator()`` is a CPU generator: handing it to ``normal_`` on a CUDA
    parameter raises "Expected a 'cuda' device type for generator", which is
    invisible to a CPU-only test suite. Switching to a device-side generator
    would run, but silently change the numbers (a CUDA generator yields a
    different sequence for the same seed), making this sanity check's result
    depend on where it ran. Only the copy-based path satisfies both, and only
    this structural assertion can catch a regression without a GPU.
    """
    bundle = build_model("cnn", pretrained=False)
    head_parameter_ids = {id(p) for p in head_module(bundle.model, "cnn").parameters()}
    drawn_into: list[int] = []
    original_normal = torch.Tensor.normal_

    def spy(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        drawn_into.append(id(self))
        return original_normal(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor, "normal_", spy)
        randomize_head(bundle.model, "cnn", seed=7)

    assert drawn_into, "normal_ was never called"
    assert not head_parameter_ids & set(drawn_into), (
        "randomize_head drew straight into a head parameter; on CUDA that raises "
        "\"Expected a 'cuda' device type for generator\""
    )
