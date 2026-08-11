from __future__ import annotations

import pytest
import torch

from vision_xai.models.factory import build_model, head_module
from vision_xai.train.metrics import accuracy, expected_calibration_error, macro_f1


def test_cnn_output_shape_and_freeze() -> None:
    bundle = build_model("cnn", pretrained=False)
    logits = bundle.model(torch.rand(2, 3, 64, 64))
    assert logits.shape == (2, 37)
    head = head_module(bundle.model, "cnn")
    trainable = [p for p in bundle.model.parameters() if p.requires_grad]
    assert len(trainable) == len(list(head.parameters()))
    assert all(p.requires_grad for p in head.parameters())


def test_vit_output_shape_and_freeze() -> None:
    bundle = build_model("vit", pretrained=False)
    logits = bundle.model(torch.rand(1, 3, 224, 224))
    assert logits.shape == (1, 37)
    head = head_module(bundle.model, "vit")
    trainable = [p for p in bundle.model.parameters() if p.requires_grad]
    assert len(trainable) == len(list(head.parameters()))


def test_custom_num_classes_and_full_finetune_flag() -> None:
    bundle = build_model("cnn", num_classes=4, pretrained=False, head_only=False)
    assert bundle.model(torch.rand(1, 3, 64, 64)).shape == (1, 4)
    assert all(p.requires_grad for p in bundle.model.parameters())


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("resnet", pretrained=False)  # type: ignore[arg-type]


def test_accuracy_and_macro_f1_hand_computed() -> None:
    predictions = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])
    assert accuracy(predictions, targets) == pytest.approx(0.75)
    # class 0: f1=1; class 1: f1=2/3; class 2: f1=2/3 -> macro = 7/9
    assert macro_f1(predictions, targets, num_classes=3) == pytest.approx(7 / 9)


def test_metrics_empty_input() -> None:
    empty = torch.tensor([], dtype=torch.long)
    assert accuracy(empty, empty) == 0.0
    assert macro_f1(empty, empty, num_classes=3) == 0.0


def test_macro_f1_counts_absent_classes_as_zero() -> None:
    predictions = torch.tensor([0, 0])
    targets = torch.tensor([0, 0])
    # class 0 perfect, classes 1 and 2 absent -> (1 + 0 + 0) / 3
    assert macro_f1(predictions, targets, num_classes=3) == pytest.approx(1 / 3)


def test_expected_calibration_error_hand_computed() -> None:
    # bin (0.9, 1.0]: two samples at conf=0.95, both correct -> acc=1.0,
    #   |1.0-0.95| weighted 2/4 = 0.025
    # bin (0.5, 0.6]: two samples at conf=0.55, one correct -> acc=0.5,
    #   |0.5-0.55| weighted 2/4 = 0.025
    confidences = torch.tensor([0.95, 0.95, 0.55, 0.55])
    correct = torch.tensor([True, True, True, False])
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.05)


def test_expected_calibration_error_perfectly_calibrated_is_zero() -> None:
    confidences = torch.tensor([1.0, 1.0, 1.0])
    correct = torch.tensor([True, True, True])
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)


def test_expected_calibration_error_empty_input() -> None:
    empty_conf = torch.tensor([], dtype=torch.float32)
    empty_correct = torch.tensor([], dtype=torch.bool)
    assert expected_calibration_error(empty_conf, empty_correct) == 0.0
