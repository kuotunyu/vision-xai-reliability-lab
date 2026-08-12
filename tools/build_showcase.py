"""Build and audit the weight-free static results showcase."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 1024 * 1024
FORBIDDEN_SUFFIXES = {".ckpt", ".npz", ".pt", ".pth"}
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".txt"}
SAFE_FILES = {
    "showcase/index.html": "index.html",
    "showcase/styles.css": "styles.css",
    "showcase/app.js": "app.js",
    "results/derived/summary.json": "data/summary.json",
    "release/cuda-resume-canary.json": "data/cuda-resume-canary.json",
    "release/artifact-manifest.json": "data/artifact-manifest.json",
    "assets/portfolio/showcase-demo-2026-08-12.png": (
        "assets/portfolio/showcase-demo-2026-08-12.png"
    ),
    "assets/portfolio/social-preview.png": "assets/portfolio/social-preview.png",
    "assets/figures/faithfulness_cnn.png": "assets/figures/faithfulness_cnn.png",
    "assets/figures/faithfulness_vit.png": "assets/figures/faithfulness_vit.png",
    "assets/figures/localization_cnn.png": "assets/figures/localization_cnn.png",
    "assets/figures/localization_vit.png": "assets/figures/localization_vit.png",
    "assets/figures/spurious_cnn_patched.png": "assets/figures/spurious_cnn_patched.png",
    "assets/figures/spurious_vit_patched.png": "assets/figures/spurious_vit_patched.png",
    "ARTIFACTS.md": "docs/ARTIFACTS.md",
    "DATA_CARD.md": "docs/DATA_CARD.md",
    "MODEL_CARD.md": "docs/MODEL_CARD.md",
    "LICENSE": "docs/LICENSE.txt",
}


class ShowcaseError(RuntimeError):
    """The showcase would cross the approved public artifact boundary."""


def _safe_destination(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ShowcaseError(f"unsafe showcase path: {relative}")
    destination = root.joinpath(*pure.parts).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ShowcaseError(f"showcase path escapes output: {relative}") from exc
    return destination


def _audit_file(path: Path, relative: str) -> None:
    if path.is_symlink():
        raise ShowcaseError(f"showcase symlink is not allowed: {relative}")
    if not path.is_file():
        raise ShowcaseError(f"missing showcase file: {relative}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ShowcaseError(f"forbidden showcase file: {relative}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ShowcaseError(f"showcase file exceeds 1 MiB: {relative}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ShowcaseError(f"showcase text is not UTF-8: {relative}") from exc
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
        raise ShowcaseError(f"private absolute path in showcase: {relative}")
    if any(pattern.search(text) for pattern in secret_patterns):
        raise ShowcaseError(f"possible secret in showcase: {relative}")


def audit_showcase(output: Path) -> list[str]:
    """Return sorted safe file paths or raise without modifying the directory."""
    output = output.resolve()
    if not output.is_dir():
        raise ShowcaseError(f"showcase output is not a directory: {output}")
    audited: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_dir():
            continue
        try:
            relative = path.relative_to(output).as_posix()
        except ValueError as exc:
            raise ShowcaseError(f"showcase file escapes output: {path}") from exc
        _audit_file(path, relative)
        audited.append(relative)
    if not audited:
        raise ShowcaseError("showcase output contains no files")
    return audited


def build_showcase(root: Path, output: Path) -> list[Path]:
    """Copy the explicit public allowlist into an empty output directory."""
    root = root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ShowcaseError("output directory must be empty")

    transfers: list[tuple[Path, Path]] = []
    for source_relative, output_relative in SAFE_FILES.items():
        source = _safe_destination(root, source_relative)
        _audit_file(source, source_relative)
        destination = _safe_destination(output, output_relative)
        transfers.append((source, destination))

    output.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for source, destination in transfers:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        exported.append(destination)
    audit_showcase(output)
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    try:
        if args.audit is not None:
            audited = audit_showcase(args.audit)
            sys.stdout.write(f"PASS showcase boundary ({len(audited)} files)\n")
            return 0
        if args.output is None:
            parser.error("--output is required unless --audit is used")
        exported = build_showcase(args.root, args.output)
    except ShowcaseError as exc:
        sys.stderr.write(f"FAIL {exc}\n")
        return 1
    sys.stdout.write(f"PASS built showcase ({len(exported)} files)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
