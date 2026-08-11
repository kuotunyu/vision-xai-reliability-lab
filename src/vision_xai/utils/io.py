"""File I/O helpers: atomic JSON writes and UTF-8 JSONL append/read."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    """Write JSON via a temp file + ``os.replace`` so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append one JSON object per line; creates the file and parents if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a complete JSONL file atomically (temp file + ``os.replace``)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    append_jsonl(tmp, rows)
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
