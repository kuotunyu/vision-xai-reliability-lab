from __future__ import annotations

import hashlib
from pathlib import Path

from conftest import RecordFactory
from vision_xai.data.fingerprint import build_fingerprint, content_fingerprint, sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_one_byte_change_changes_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00" * 1024)
    original = sha256_file(path)
    path.write_bytes(b"\x00" * 1023 + b"\x01")
    assert sha256_file(path) != original


def test_content_fingerprint_is_order_invariant(record_factory: RecordFactory) -> None:
    records = [record_factory(f"Cat_{i:03d}") for i in range(10)]
    assert content_fingerprint(records) == content_fingerprint(list(reversed(records)))


def test_content_fingerprint_sensitive_to_file_hashes(record_factory: RecordFactory) -> None:
    records = [record_factory(f"Cat_{i:03d}") for i in range(3)]
    tampered = [records[0].model_copy(update={"image_sha256": "f" * 64}), *records[1:]]
    assert content_fingerprint(records) != content_fingerprint(tampered)


def test_build_fingerprint_counts_and_meta(record_factory: RecordFactory) -> None:
    records = [
        record_factory("Cat_001", class_id=0, split="train"),
        record_factory("Cat_002", class_id=0, split="val"),
        record_factory("dog_001", class_id=1, split="test"),
    ]
    fingerprint = build_fingerprint(records, {"dataset": "oxford-iiit-pet", "config_hash": "x"})
    assert fingerprint["num_samples"] == {"test": 1, "train": 1, "val": 1}
    assert fingerprint["num_classes"] == 2
    assert fingerprint["dataset"] == "oxford-iiit-pet"
    assert fingerprint["config_hash"] == "x"
    assert fingerprint["content_sha256"] == content_fingerprint(records)
