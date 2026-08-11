"""Dataset sources.

``OxfordPetSource`` handles the real Oxford-IIIT Pet layout (as created by
torchvision); tests point it at a tiny synthetic tree with the same structure,
so the parsing path under test is the production one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

from vision_xai.errors import DataPreparationError, DatasetIntegrityError

logger = logging.getLogger(__name__)

NUM_CLASSES = 37
EXPECTED_COUNTS = {"trainval": 3680, "test": 3669}

Species = Literal["cat", "dog"]
OfficialSplit = Literal["trainval", "test"]

_SPECIES_BY_CODE: dict[int, Species] = {1: "cat", 2: "dog"}


class SampleRef(NamedTuple):
    sample_id: str
    class_id: int  # zero-based, 0..36 (torchvision convention)
    class_name: str
    species: Species
    official_split: OfficialSplit
    image_path: Path
    trimap_path: Path


class DatasetSource(Protocol):
    def ensure_available(self) -> None: ...

    def list_samples(self) -> list[SampleRef]: ...


class OxfordPetSource:
    """Oxford-IIIT Pet under ``root/oxford-iiit-pet/`` (torchvision layout).

    List files (``annotations/trainval.txt`` / ``test.txt``) have lines of
    ``<sample_id> <class_id 1..37> <species 1=cat|2=dog> <breed_id>``.
    """

    _BASE = "oxford-iiit-pet"

    def __init__(self, root: Path, download: bool = True) -> None:
        self._root = root
        self._download = download
        self._base = root / self._BASE
        self._images_dir = self._base / "images"
        self._annotations_dir = self._base / "annotations"
        self._trimaps_dir = self._annotations_dir / "trimaps"

    def ensure_available(self) -> None:
        if self._download:
            # torchvision downloads both archives and verifies their md5 checksums.
            from torchvision.datasets import OxfordIIITPet

            logger.info("ensuring Oxford-IIIT Pet is available under %s", self._base)
            for split in ("trainval", "test"):
                OxfordIIITPet(
                    root=str(self._root),
                    split=split,
                    target_types="category",
                    download=True,
                )
            return
        missing = [
            path
            for path in (
                self._images_dir,
                self._trimaps_dir,
                self._annotations_dir / "trainval.txt",
                self._annotations_dir / "test.txt",
            )
            if not path.exists()
        ]
        if missing:
            raise DataPreparationError(
                "dataset not found and download is disabled; missing: "
                + ", ".join(str(p) for p in missing)
            )

    def list_samples(self) -> list[SampleRef]:
        samples: list[SampleRef] = []
        seen: dict[str, OfficialSplit] = {}
        for split in ("trainval", "test"):
            list_file = self._annotations_dir / f"{split}.txt"
            for line_no, line in enumerate(
                list_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                sample = self._parse_line(stripped, split, list_file, line_no)
                if sample.sample_id in seen:
                    raise DatasetIntegrityError(
                        f"sample {sample.sample_id!r} listed in both "
                        f"{seen[sample.sample_id]!r} and {split!r}"
                    )
                seen[sample.sample_id] = split
                samples.append(sample)
        self._check_files_exist(samples)
        samples.sort(key=lambda s: s.sample_id)
        for split in ("trainval", "test"):
            count = sum(1 for s in samples if s.official_split == split)
            logger.info("listed %d samples for official split %r", count, split)
        return samples

    def _parse_line(
        self, line: str, split: OfficialSplit, list_file: Path, line_no: int
    ) -> SampleRef:
        expected_fields = 4
        parts = line.split()
        where = f"{list_file}:{line_no}"
        if len(parts) != expected_fields:
            raise DatasetIntegrityError(f"{where}: expected 4 fields, got {len(parts)}: {line!r}")
        sample_id, raw_class, raw_species, _breed = parts
        try:
            class_id_1based = int(raw_class)
            species_code = int(raw_species)
        except ValueError as exc:
            raise DatasetIntegrityError(f"{where}: non-integer field in {line!r}") from exc
        if not 1 <= class_id_1based <= NUM_CLASSES:
            raise DatasetIntegrityError(f"{where}: class id {class_id_1based} outside 1..37")
        species = _SPECIES_BY_CODE.get(species_code)
        if species is None:
            raise DatasetIntegrityError(f"{where}: species code {species_code} outside {{1, 2}}")
        return SampleRef(
            sample_id=sample_id,
            class_id=class_id_1based - 1,
            class_name=sample_id.rsplit("_", 1)[0],
            species=species,
            official_split=split,
            image_path=self._images_dir / f"{sample_id}.jpg",
            trimap_path=self._trimaps_dir / f"{sample_id}.png",
        )

    def _check_files_exist(self, samples: list[SampleRef]) -> None:
        missing = [
            str(path)
            for sample in samples
            for path in (sample.image_path, sample.trimap_path)
            if not path.is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise DatasetIntegrityError(
                f"{len(missing)} listed files are missing on disk (first few: {preview})"
            )
