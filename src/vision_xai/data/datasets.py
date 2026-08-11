"""Manifest-driven classification dataset, optionally patch-aware and mask-returning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset

from vision_xai.config import SpuriousPatchConfig
from vision_xai.data.manifest import ManifestRecord
from vision_xai.data.patches import apply_corner_patch
from vision_xai.utils.seed import per_sample_rng

PetItem = tuple[torch.Tensor, int] | tuple[torch.Tensor, int, npt.NDArray[np.bool_]]
FLIP_PROBABILITY = 0.5


class PetClassificationDataset(Dataset[PetItem]):
    """Yields ``(image, class_id)`` or ``(image, class_id, pet_mask)`` per manifest record.

    The spurious corner patch (if assigned to a sample) is applied after the image
    transform, in final model-input coordinates.
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        data_root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], npt.NDArray[np.bool_]] | None = None,
        patch_cfg: SpuriousPatchConfig | None = None,
        patch_assignment: Mapping[str, bool] | None = None,
        post_patch_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_mask: bool = False,
        train_augment_seed: int | None = None,
    ) -> None:
        if return_mask and mask_transform is None:
            raise ValueError("return_mask=True requires a mask_transform")
        if (patch_cfg is None) != (patch_assignment is None):
            raise ValueError("patch_cfg and patch_assignment must be provided together")
        self._records = list(records)
        self._data_root = data_root
        self._transform = transform
        self._mask_transform = mask_transform
        self._patch_cfg = patch_cfg
        self._patch_assignment = patch_assignment
        self._post_patch_transform = post_patch_transform
        self._return_mask = return_mask
        self._train_augment_seed = train_augment_seed

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> PetItem:
        record = self._records[index]
        with Image.open(self._data_root / record.image_relpath) as raw:
            image = self._transform(raw.convert("RGB"))
        if self._train_augment_seed is not None:
            # A horizontal flip, decided deterministically per (seed, sample_id)
            # rather than drawn from torch's global RNG — a pure function of the
            # sample, independent of process/epoch history, so --resume sees the
            # exact same augmentation an uninterrupted run would have.
            rng = per_sample_rng(self._train_augment_seed, record.sample_id, "train_flip")
            if rng.random() < FLIP_PROBABILITY:
                image = torch.flip(image, dims=[-1])
        if (
            self._patch_cfg is not None
            and self._patch_assignment is not None
            and self._patch_assignment.get(record.sample_id, False)
        ):
            image = apply_corner_patch(image, self._patch_cfg)
        if self._post_patch_transform is not None:
            image = self._post_patch_transform(image)
        if not self._return_mask:
            return image, record.class_id
        assert self._mask_transform is not None  # guaranteed by __init__
        with Image.open(self._data_root / record.trimap_relpath) as trimap:
            mask = self._mask_transform(trimap)
        return image, record.class_id, mask
