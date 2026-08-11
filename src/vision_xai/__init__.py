"""Vision XAI Reliability Lab.

Compare attribution methods between a CNN (ConvNeXt-Tiny) and a ViT (ViT-B/16)
on Oxford-IIIT Pet, and quantitatively evaluate explanation reliability.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

MIN_TORCH_VERSION = (2, 6)
MIN_TORCHVISION_VERSION = (0, 21)


def _parse_version(raw: str) -> tuple[int, int]:
    parts = raw.split("+", maxsplit=1)[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


def check_runtime() -> None:
    """Log torch/torchvision versions and warn (never fail) below the tested floor.

    On Colab this package is installed with ``pip install . --no-deps`` on top of
    the preinstalled CUDA torch, so version enforcement must stay advisory.
    """
    import torch
    import torchvision

    # torchvision defines __version__ dynamically; its stubs do not re-export it.
    torchvision_version: str = getattr(torchvision, "__version__", "0.0")
    torch_ok = _parse_version(torch.__version__) >= MIN_TORCH_VERSION
    tv_ok = _parse_version(torchvision_version) >= MIN_TORCHVISION_VERSION
    logger.info(
        "runtime: torch=%s torchvision=%s cuda_available=%s",
        torch.__version__,
        torchvision_version,
        torch.cuda.is_available(),
    )
    if not torch_ok:
        logger.warning(
            "torch %s is below the tested floor %s.%s; behaviour is not guaranteed",
            torch.__version__,
            *MIN_TORCH_VERSION,
        )
    if not tv_ok:
        logger.warning(
            "torchvision %s is below the tested floor %s.%s; behaviour is not guaranteed",
            torchvision_version,
            *MIN_TORCHVISION_VERSION,
        )
