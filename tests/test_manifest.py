from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import RecordFactory
from vision_xai.data.manifest import (
    ManifestRecord,
    PrepareState,
    append_records,
    clear_state,
    load_manifest,
    load_state,
    save_state,
    write_manifest,
)
from vision_xai.errors import DataPreparationError, ResumeStateError


def test_manifest_round_trip(tmp_path: Path, record_factory: RecordFactory) -> None:
    records = [record_factory(f"Cat_{i:03d}", split="train") for i in range(5)]
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, records)
    assert load_manifest(path) == records


def test_append_then_load_partial(tmp_path: Path, record_factory: RecordFactory) -> None:
    path = tmp_path / "partial.jsonl"
    append_records(path, [record_factory("Cat_001")])
    append_records(path, [record_factory("Cat_002")])
    loaded = load_manifest(path, require_split=False)
    assert [r.sample_id for r in loaded] == ["Cat_001", "Cat_002"]
    assert all(r.split is None for r in loaded)


def test_duplicate_sample_id_rejected(tmp_path: Path, record_factory: RecordFactory) -> None:
    path = tmp_path / "manifest.jsonl"
    append_records(path, [record_factory("Cat_001", split="train")] * 2)
    with pytest.raises(DataPreparationError, match="duplicate"):
        load_manifest(path)


def test_missing_split_rejected_when_required(
    tmp_path: Path, record_factory: RecordFactory
) -> None:
    path = tmp_path / "manifest.jsonl"
    append_records(path, [record_factory("Cat_001")])
    with pytest.raises(DataPreparationError, match="no split"):
        load_manifest(path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "oxford-iiit-pet\\images\\Cat_001.jpg",  # backslashes
        "C:/data/images/Cat_001.jpg",  # drive letter
        "/data/images/Cat_001.jpg",  # absolute
    ],
)
def test_relpaths_must_be_posix_relative(bad_path: str, record_factory: RecordFactory) -> None:
    good = record_factory("Cat_001")
    with pytest.raises(ValidationError, match="POSIX"):
        ManifestRecord.model_validate({**good.model_dump(), "image_relpath": bad_path})


def test_state_round_trip_and_clear(tmp_path: Path) -> None:
    state = PrepareState(
        stage="hashing", completed=7, total=36, config_hash="abc123", updated_at="t"
    )
    save_state(tmp_path, state)
    assert load_state(tmp_path) == state
    clear_state(tmp_path)
    assert load_state(tmp_path) is None
    clear_state(tmp_path)  # idempotent


def test_corrupt_state_raises(tmp_path: Path) -> None:
    (tmp_path / "prepare_state.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ResumeStateError, match="corrupt"):
        load_state(tmp_path)
