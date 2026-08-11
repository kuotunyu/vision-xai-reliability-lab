"""Trimap semantics and geometry-aligned mask handling.

Official Oxford-IIIT Pet annotation semantics (annotations README on
https://www.robots.ox.ac.uk/~vgg/data/pets/): 1 = foreground (pet),
2 = background, 3 = not classified (border/ambiguous). Because some third-party
copies remap these values, :func:`verify_trimap` additionally checks the
documented semantics empirically (image-border pixels should be mostly
background); the aggregate check lives in the prepare pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from PIL import Image

from vision_xai.config import MaskPolicy

TRIMAP_FOREGROUND: Final[int] = 1
TRIMAP_BACKGROUND: Final[int] = 2
TRIMAP_BOUNDARY: Final[int] = 3
VALID_TRIMAP_VALUES: Final[frozenset[int]] = frozenset(
    {TRIMAP_FOREGROUND, TRIMAP_BACKGROUND, TRIMAP_BOUNDARY}
)


def trimap_to_mask(
    trimap: Image.Image | npt.NDArray[np.uint8], policy: MaskPolicy
) -> npt.NDArray[np.bool_]:
    """Convert a trimap to a boolean pet mask under the given boundary policy."""
    array: npt.NDArray[np.uint8] = np.asarray(trimap, dtype=np.uint8)
    mask: npt.NDArray[np.bool_] = array == TRIMAP_FOREGROUND
    if policy == "foreground_and_boundary":
        mask = mask | (array == TRIMAP_BOUNDARY)
    return mask


def resize_and_center_crop_trimap(
    trimap: Image.Image, resize_size: int, crop_size: int
) -> npt.NDArray[np.uint8]:
    """Apply the same geometry as the image transform, with NEAREST interpolation.

    NEAREST guarantees output values stay within the original trimap value set.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    # torchvision's functional ops accept PIL images at runtime; the annotations
    # only mention Tensor, hence the targeted ignore.
    resized = tvf.resize(
        trimap,  # type: ignore[arg-type]
        [resize_size],
        interpolation=InterpolationMode.NEAREST,
    )
    cropped = tvf.center_crop(resized, [crop_size])
    result: npt.NDArray[np.uint8] = np.asarray(cropped, dtype=np.uint8)
    return result


@dataclass(frozen=True)
class TrimapCheckResult:
    values_ok: bool
    unique_values: tuple[int, ...]
    border_background_fraction: float


def verify_trimap(trimap: npt.NDArray[np.uint8]) -> TrimapCheckResult:
    """Check a single trimap against the documented value set and border heuristic.

    ``border_background_fraction`` is the share of image-border pixels equal to
    the documented background value; per-image values are aggregated by the
    prepare pipeline to detect a global 1<->2 semantic swap.
    """
    unique_values = tuple(int(v) for v in np.unique(trimap))
    values_ok = set(unique_values) <= VALID_TRIMAP_VALUES
    border = np.concatenate([trimap[0, :], trimap[-1, :], trimap[1:-1, 0], trimap[1:-1, -1]])
    fraction = float(np.mean(border == TRIMAP_BACKGROUND)) if border.size else 0.0
    return TrimapCheckResult(
        values_ok=values_ok,
        unique_values=unique_values,
        border_background_fraction=fraction,
    )
