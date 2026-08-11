"""Shared fixtures: a tiny synthetic dataset in the Oxford-IIIT Pet layout.

Everything is PIL-generated with seeded RNGs — tests never touch the real
dataset, so the suite runs offline and in CI.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vision_xai.config import AppConfig
from vision_xai.data.manifest import ManifestRecord
from vision_xai.data.trimap import TRIMAP_BACKGROUND, TRIMAP_BOUNDARY, TRIMAP_FOREGROUND
from vision_xai.utils.seed import per_sample_rng

IMG_SIZE = 64
# Known trimap geometry: boundary ring [15,49), foreground rectangle [16,48).
FOREGROUND_RECT = (16, 16, 48, 48)  # x0, y0, x1, y1

# (class_name, class_id_1based, species_code 1=cat|2=dog) — mirrors the real
# naming convention (cat breeds capitalized, dog breeds lowercase).
SYNTH_CLASSES = [
    ("Fakecat", 1, 1),
    ("Mockcat", 2, 1),
    ("fakedog", 3, 2),
    ("mockdog", 4, 2),
]
N_TRAINVAL_PER_CLASS = 6
N_TEST_PER_CLASS = 3

ConfigFactory = Callable[..., AppConfig]
RecordFactory = Callable[..., ManifestRecord]


@pytest.fixture(autouse=True)
def _no_data_dir_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISION_XAI_DATA_DIR", raising=False)


def _write_image(path: Path, sample_id: str) -> None:
    rng = per_sample_rng(0, sample_id, "fixture-image")
    base = rng.integers(0, 216, size=3).astype(np.int16)
    noise = rng.integers(0, 40, size=(IMG_SIZE, IMG_SIZE, 3)).astype(np.int16)
    array = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=90)


def write_trimap(path: Path, swapped: bool = False) -> None:
    fg, bg = (
        (TRIMAP_BACKGROUND, TRIMAP_FOREGROUND)
        if swapped
        else (
            TRIMAP_FOREGROUND,
            TRIMAP_BACKGROUND,
        )
    )
    array = np.full((IMG_SIZE, IMG_SIZE), bg, dtype=np.uint8)
    array[15:49, 15:49] = TRIMAP_BOUNDARY
    x0, y0, x1, y1 = FOREGROUND_RECT
    array[y0:y1, x0:x1] = fg
    Image.fromarray(array, mode="L").save(path, format="PNG")


def make_synthetic_pet_tree(data_dir: Path) -> None:
    """Create a miniature dataset in the exact torchvision on-disk layout."""
    base = data_dir / "oxford-iiit-pet"
    images = base / "images"
    annotations = base / "annotations"
    trimaps = annotations / "trimaps"
    images.mkdir(parents=True, exist_ok=True)
    trimaps.mkdir(parents=True, exist_ok=True)
    lines: dict[str, list[str]] = {"trainval": [], "test": []}
    for class_name, class_id_1based, species_code in SYNTH_CLASSES:
        for split, count, offset in (
            ("trainval", N_TRAINVAL_PER_CLASS, 100),
            ("test", N_TEST_PER_CLASS, 200),
        ):
            for index in range(count):
                sample_id = f"{class_name}_{offset + index}"
                _write_image(images / f"{sample_id}.jpg", sample_id)
                write_trimap(trimaps / f"{sample_id}.png")
                lines[split].append(
                    f"{sample_id} {class_id_1based} {species_code} {class_id_1based}"
                )
    for split, split_lines in lines.items():
        (annotations / f"{split}.txt").write_text("\n".join(split_lines) + "\n", encoding="utf-8")


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    make_synthetic_pet_tree(data_dir)
    return data_dir


@pytest.fixture()
def config_factory(tmp_path: Path) -> ConfigFactory:
    """Build an AppConfig aimed at a synthetic tree; kwargs override the data section."""

    def factory(
        data_dir: Path,
        top_level: dict[str, Any] | None = None,
        **data_overrides: Any,
    ) -> AppConfig:
        data: dict[str, Any] = {
            "download": False,
            "image_size": 32,
            "resize_size": 40,
            "incremental_save_every": 5,
        }
        data.update(data_overrides)
        raw: dict[str, Any] = {
            "experiment_name": "testexp",
            "seed": 42,
            "paths": {
                "data_dir": str(data_dir),
                "results_dir": str(tmp_path / "results"),
            },
            "data": data,
        }
        raw.update(top_level or {})
        return AppConfig.model_validate(raw)

    return factory


@pytest.fixture()
def app_config(synthetic_data_dir: Path, config_factory: ConfigFactory) -> AppConfig:
    return config_factory(synthetic_data_dir)


@pytest.fixture()
def record_factory() -> RecordFactory:
    def factory(
        sample_id: str,
        species: str = "cat",
        class_id: int = 0,
        split: str | None = None,
    ) -> ManifestRecord:
        return ManifestRecord.model_validate(
            {
                "sample_id": sample_id,
                "official_split": "trainval",
                "split": split,
                "class_id": class_id,
                "class_name": sample_id.rsplit("_", 1)[0],
                "species": species,
                "image_relpath": f"oxford-iiit-pet/images/{sample_id}.jpg",
                "trimap_relpath": f"oxford-iiit-pet/annotations/trimaps/{sample_id}.png",
                "image_sha256": "0" * 64,
                "trimap_sha256": "1" * 64,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "trimap_values": [1, 2, 3],
                "border_background_fraction": 1.0,
            }
        )

    return factory
