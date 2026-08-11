"""Audit built wheel and sdist contents without extracting either archive."""

from __future__ import annotations

import argparse
import re
import stat
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_MEMBER_BYTES = 1024 * 1024
FORBIDDEN_PARTS = {
    ".artifacts",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "notebooks",
}
FORBIDDEN_NAMES = {"PROGRESS.md", "RELEASE_AUDIT.md"}
FORBIDDEN_SUFFIXES = {".ckpt", ".pt", ".pth", ".pyc"}
ALLOWED_RAW_RESULTS = {
    "results/raw/data_prepare/full/fingerprint.json",
    "results/raw/data_prepare/full/patch_summary.json",
    "results/raw/data_prepare/full/split_summary.json",
}
TEXT_SUFFIXES = {
    "",
    ".example",
    ".gitignore",
    ".gitattributes",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class DistributionAuditError(RuntimeError):
    """A built distribution crosses the public release boundary."""


@dataclass(frozen=True)
class Member:
    name: str
    size: int
    payload: bytes


def _pure_member(name: str) -> PurePosixPath:
    if "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise DistributionAuditError(f"unsafe archive path: {name}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise DistributionAuditError(f"unsafe archive path: {name}")
    return pure


def _wheel_members(path: Path) -> list[Member]:
    try:
        with zipfile.ZipFile(path) as archive:
            members: list[Member] = []
            for info in archive.infolist():
                _pure_member(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DistributionAuditError(f"archive link is not allowed: {info.filename}")
                if info.is_dir():
                    continue
                members.append(Member(info.filename, info.file_size, archive.read(info)))
            return members
    except (OSError, zipfile.BadZipFile) as exc:
        raise DistributionAuditError(f"cannot read wheel {path}: {exc}") from exc


def _sdist_members(path: Path) -> list[Member]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members: list[Member] = []
            for info in archive.getmembers():
                _pure_member(info.name)
                if info.issym() or info.islnk():
                    raise DistributionAuditError(f"archive link is not allowed: {info.name}")
                if info.isdir():
                    continue
                if not info.isfile():
                    raise DistributionAuditError(f"unsupported archive member: {info.name}")
                stream = archive.extractfile(info)
                if stream is None:
                    raise DistributionAuditError(f"cannot read archive member: {info.name}")
                members.append(Member(info.name, info.size, stream.read()))
            return members
    except (OSError, tarfile.TarError) as exc:
        raise DistributionAuditError(f"cannot read sdist {path}: {exc}") from exc


def _relative_sdist_names(
    members: Iterable[Member],
) -> tuple[str, list[tuple[PurePosixPath, Member]]]:
    entries = [(_pure_member(member.name), member) for member in members]
    roots = {pure.parts[0] for pure, _ in entries}
    if len(roots) != 1:
        raise DistributionAuditError("sdist must have exactly one top-level directory")
    root = roots.pop()
    relative: list[tuple[PurePosixPath, Member]] = []
    for pure, member in entries:
        parts = pure.parts[1:]
        if not parts:
            continue
        relative.append((PurePosixPath(*parts), member))
    return root, relative


def _check_public_member(relative: PurePosixPath, member: Member) -> None:
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        raise DistributionAuditError(f"forbidden distribution member: {member.name}")
    if relative.name in FORBIDDEN_NAMES or relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise DistributionAuditError(f"forbidden distribution member: {member.name}")
    if relative.parts[0] == "data":
        raise DistributionAuditError(f"raw dataset directory in distribution: {member.name}")
    relative_text = relative.as_posix()
    if relative_text.startswith("results/raw/") and relative_text not in ALLOWED_RAW_RESULTS:
        raise DistributionAuditError(f"runtime result in distribution: {member.name}")
    if member.size > MAX_MEMBER_BYTES:
        raise DistributionAuditError(f"distribution member exceeds 1 MiB: {member.name}")
    if relative.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = member.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DistributionAuditError(f"distribution text is not UTF-8: {member.name}") from exc
    private_paths = (
        re.compile(r"(?i)[a-z]:[\\/]Users[\\/][A-Za-z0-9._-]+[\\/]"),
        re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    )
    secret_patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?" + "PRIVATE KEY"),
    )
    if any(pattern.search(text) for pattern in private_paths):
        raise DistributionAuditError(f"private absolute path in distribution: {member.name}")
    if any(pattern.search(text) for pattern in secret_patterns):
        raise DistributionAuditError(f"possible secret in distribution: {member.name}")


def audit_wheel(path: Path) -> int:
    members = _wheel_members(path)
    if not members:
        raise DistributionAuditError("wheel is empty")
    for member in members:
        relative = _pure_member(member.name)
        top = relative.parts[0]
        if top != "vision_xai" and not top.startswith("vision_xai-"):
            raise DistributionAuditError(f"unexpected wheel member: {member.name}")
        if top.startswith("vision_xai-") and not top.endswith(".dist-info"):
            raise DistributionAuditError(f"unexpected wheel metadata directory: {member.name}")
        _check_public_member(relative, member)
    names = {member.name for member in members}
    required_suffixes = ("/METADATA", "/WHEEL", "/entry_points.txt", "/licenses/LICENSE")
    if "vision_xai/__init__.py" not in names:
        raise DistributionAuditError("wheel package is missing vision_xai/__init__.py")
    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in names):
            raise DistributionAuditError(f"wheel metadata is missing {suffix.removeprefix('/')}")
    return len(members)


def audit_sdist(path: Path) -> int:
    members = _sdist_members(path)
    if not members:
        raise DistributionAuditError("sdist is empty")
    _, entries = _relative_sdist_names(members)
    for relative, member in entries:
        _check_public_member(relative, member)
    names = {relative.as_posix() for relative, _ in entries}
    required = {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src/vision_xai/__init__.py",
        "release/artifact-manifest.json",
        "release/cuda-resume-canary.json",
        "schemas/artifact-manifest.schema.json",
        "schemas/cuda-resume-canary.schema.json",
        "schemas/full-summary.schema.json",
    }
    missing = sorted(required - names)
    if missing:
        raise DistributionAuditError(
            f"sdist is missing required public files: {', '.join(missing)}"
        )
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    args = parser.parse_args()
    try:
        wheel_count = audit_wheel(args.wheel)
        sdist_count = audit_sdist(args.sdist)
    except DistributionAuditError as exc:
        sys.stderr.write(f"FAIL {exc}\n")
        return 1
    sys.stdout.write(f"PASS wheel boundary ({wheel_count} files)\n")
    sys.stdout.write(f"PASS sdist boundary ({sdist_count} files)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
