"""Model-parameter randomization (sanity check, Adebayo et al. 2018).

The trained classification head is re-initialized with a seeded RNG; a reliable
attribution method should produce substantially different maps afterwards, so a
*low* similarity to the original maps is the healthy outcome.
"""

from __future__ import annotations

import torch
from torch import nn

from vision_xai.models.factory import ModelName, head_module

INIT_STD = 0.02


def _fill_normal(parameter: torch.Tensor, generator: torch.Generator) -> None:
    """Draw on the CPU, then copy onto whatever device the parameter lives on.

    ``torch.Generator()`` is a CPU generator, and handing one to ``normal_`` on a
    CUDA tensor raises "Expected a 'cuda' device type for generator". Switching
    to a device-side generator would work but silently change the numbers — for
    the same seed a CUDA generator yields a different sequence from a CPU one,
    so this sanity check's result would depend on where it happened to run.
    Drawing on the CPU and copying keeps it identical on every device.
    """
    values = torch.empty(parameter.shape, dtype=torch.float32)
    values.normal_(0.0, INIT_STD, generator=generator)
    parameter.copy_(values)


def randomize_head(model: nn.Module, model_name: ModelName, seed: int) -> None:
    """Re-initialize every Linear layer in the classification head, in place."""
    generator = torch.Generator().manual_seed(seed)
    head = head_module(model, model_name)
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, nn.Linear):
                _fill_normal(module.weight, generator)
                if module.bias is not None:
                    _fill_normal(module.bias, generator)
