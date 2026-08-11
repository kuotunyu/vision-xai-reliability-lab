"""Model construction: torchvision ConvNeXt-Tiny (cnn) and ViT-B/16 (vit) with a
replaced 37-class head. Head-only training freezes everything except that head."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

from torch import nn

logger = logging.getLogger(__name__)

ModelName = Literal["cnn", "vit"]
MODEL_NAMES: tuple[ModelName, ...] = ("cnn", "vit")
NUM_CLASSES = 37

# Both torchvision weight sets use the standard ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ARCH_BY_NAME = {"cnn": "convnext_tiny", "vit": "vit_b_16"}

# Index of the Linear head inside each model's classifier container:
# convnext_tiny.classifier = Sequential(LayerNorm2d, Flatten, Linear)
# vit_b_16.heads           = Sequential(head=Linear)
_CNN_HEAD_INDEX = 2
_VIT_HEAD_INDEX = -1


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    model_name: ModelName
    arch: str
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]


def _head_container(model: nn.Module, name: ModelName) -> tuple[nn.Sequential, int]:
    attribute = model.classifier if name == "cnn" else model.heads
    container = cast(nn.Sequential, attribute)
    return container, (_CNN_HEAD_INDEX if name == "cnn" else _VIT_HEAD_INDEX)


def build_model(
    name: ModelName,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    head_only: bool = True,
) -> ModelBundle:
    """Build a classifier; ``pretrained=False`` avoids any weight download (tests/CI)."""
    from torchvision import models as tv_models

    if name == "cnn":
        model: nn.Module = tv_models.convnext_tiny(
            weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif name == "vit":
        model = tv_models.vit_b_16(
            weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError(f"unknown model name {name!r}; expected one of {MODEL_NAMES}")

    container, head_index = _head_container(model, name)
    old_head = container[head_index]
    if not isinstance(old_head, nn.Linear):
        raise TypeError(f"expected a Linear head in {name}, got {type(old_head)}")
    container[head_index] = nn.Linear(old_head.in_features, num_classes)

    if head_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head_module(model, name).parameters():
            parameter.requires_grad = True
    logger.info(
        "built %s (%s): pretrained=%s head_only=%s num_classes=%d",
        name,
        ARCH_BY_NAME[name],
        pretrained,
        head_only,
        num_classes,
    )
    return ModelBundle(
        model=model,
        model_name=name,
        arch=ARCH_BY_NAME[name],
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
    )


def head_module(model: nn.Module, name: ModelName) -> nn.Module:
    """Return the trainable classification head of a model built by :func:`build_model`."""
    container, head_index = _head_container(model, name)
    return container[head_index]
