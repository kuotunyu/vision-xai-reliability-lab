"""Manifest records (JSONL) and the resume checkpoint for ``data prepare``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vision_xai.errors import DataPreparationError, ResumeStateError
from vision_xai.utils.io import append_jsonl, atomic_write_json, read_jsonl, write_jsonl_atomic

MANIFEST_FILENAME = "manifest.jsonl"
PARTIAL_MANIFEST_FILENAME = "manifest_partial.jsonl"
STATE_FILENAME = "prepare_state.json"

PrepareStage = Literal["hashing", "splitting", "patching", "done"]


class ManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    official_split: Literal["trainval", "test"]
    split: Literal["train", "val", "test"] | None = None  # None only in the partial manifest
    class_id: int = Field(ge=0)
    class_name: str
    species: Literal["cat", "dog"]
    image_relpath: str
    trimap_relpath: str
    image_sha256: str
    trimap_sha256: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    trimap_values: list[int]
    border_background_fraction: float = Field(ge=0.0, le=1.0)

    @field_validator("image_relpath", "trimap_relpath")
    @classmethod
    def _posix_relative(cls, value: str) -> str:
        if "\\" in value or ":" in value or value.startswith("/"):
            raise ValueError(f"paths must be POSIX-style and relative, got {value!r}")
        return value


class PrepareState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    stage: PrepareStage
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    config_hash: str
    updated_at: str


def append_records(path: Path, records: Sequence[ManifestRecord]) -> None:
    append_jsonl(path, [record.model_dump(mode="json") for record in records])


def write_manifest(path: Path, records: Sequence[ManifestRecord]) -> None:
    """Write a complete manifest atomically (temp file + replace)."""
    write_jsonl_atomic(path, [record.model_dump(mode="json") for record in records])


def load_manifest(path: Path, *, require_split: bool = True) -> list[ManifestRecord]:
    """Load and validate a manifest; rejects schema violations and duplicate ids."""
    try:
        records = [ManifestRecord.model_validate(row) for row in read_jsonl(path)]
    except ValidationError as exc:
        raise DataPreparationError(f"invalid manifest {path}: {exc}") from exc
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise DataPreparationError(f"duplicate sample_id {record.sample_id!r} in {path}")
        seen.add(record.sample_id)
        if require_split and record.split is None:
            raise DataPreparationError(f"record {record.sample_id!r} in {path} has no split")
    return records


def save_state(directory: Path, state: PrepareState) -> None:
    atomic_write_json(directory / STATE_FILENAME, state.model_dump(mode="json"))


def load_state(directory: Path) -> PrepareState | None:
    path = directory / STATE_FILENAME
    if not path.is_file():
        return None
    try:
        return PrepareState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ResumeStateError(f"corrupt resume state {path}: {exc}") from exc


def clear_state(directory: Path) -> None:
    (directory / STATE_FILENAME).unlink(missing_ok=True)
