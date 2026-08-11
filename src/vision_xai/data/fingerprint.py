"""Dataset fingerprinting: per-file sha256 and an order-independent content hash."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vision_xai.data.manifest import ManifestRecord

FINGERPRINT_ALGORITHM = "sha256-of-sorted(sample_id:image_sha256:trimap_sha256)-v1"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_fingerprint(records: Sequence[ManifestRecord]) -> str:
    """Order-independent hash over sample ids and file hashes (resume-order-proof)."""
    lines = sorted(f"{r.sample_id}:{r.image_sha256}:{r.trimap_sha256}" for r in records)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def build_fingerprint(records: Sequence[ManifestRecord], meta: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        if record.split is not None:
            counts[record.split] = counts.get(record.split, 0) + 1
    return {
        "schema_version": 1,
        "algorithm": FINGERPRINT_ALGORITHM,
        "content_sha256": content_fingerprint(records),
        "num_samples": dict(sorted(counts.items())),
        "num_classes": len({record.class_id for record in records}),
        **dict(meta),
    }
