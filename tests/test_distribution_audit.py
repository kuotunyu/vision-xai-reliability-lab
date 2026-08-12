from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_wheel(path: Path, extra: dict[str, bytes] | None = None) -> None:
    members = {
        "vision_xai/__init__.py": b'__version__ = "0.1.0"\n',
        "vision_xai-0.1.0.dist-info/METADATA": b"Name: vision-xai\nVersion: 0.1.0\n",
        "vision_xai-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        "vision_xai-0.1.0.dist-info/entry_points.txt": b"[console_scripts]\n",
        "vision_xai-0.1.0.dist-info/licenses/LICENSE": b"MIT\n",
        "vision_xai-0.1.0.dist-info/RECORD": b"",
    }
    members.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _write_sdist(
    path: Path,
    extra: dict[str, bytes] | None = None,
    *,
    include_english_readme: bool = True,
) -> None:
    members = {
        "vision_xai-0.1.0/pyproject.toml": b"[project]\nname = 'vision-xai'\n",
        "vision_xai-0.1.0/LICENSE": b"MIT\n",
        "vision_xai-0.1.0/README.md": b"# release\n",
        "vision_xai-0.1.0/src/vision_xai/__init__.py": b"",
        "vision_xai-0.1.0/release/artifact-manifest.json": b"{}\n",
        "vision_xai-0.1.0/release/cuda-resume-canary.json": b"{}\n",
        "vision_xai-0.1.0/schemas/artifact-manifest.schema.json": b"{}\n",
        "vision_xai-0.1.0/schemas/cuda-resume-canary.schema.json": b"{}\n",
        "vision_xai-0.1.0/schemas/full-summary.schema.json": b"{}\n",
    }
    if include_english_readme:
        members["vision_xai-0.1.0/README_en.md"] = b"# English release\n"
    members.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _run(wheel: Path, sdist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "audit_distribution.py"),
            "--wheel",
            str(wheel),
            "--sdist",
            str(sdist),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_distribution_audit_accepts_minimal_public_archives(tmp_path: Path) -> None:
    wheel = tmp_path / "vision_xai-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "vision_xai-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    result = _run(wheel, sdist)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS wheel boundary" in result.stdout
    assert "PASS sdist boundary" in result.stdout


def test_distribution_audit_requires_complete_bilingual_readmes(tmp_path: Path) -> None:
    wheel = tmp_path / "vision_xai-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "vision_xai-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist, include_english_readme=False)

    result = _run(wheel, sdist)

    assert result.returncode == 1
    assert "README_en.md" in result.stderr


def test_distribution_audit_rejects_archive_traversal(tmp_path: Path) -> None:
    wheel = tmp_path / "vision_xai-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "vision_xai-0.1.0.tar.gz"
    _write_wheel(wheel, {"../outside.txt": b"escape\n"})
    _write_sdist(sdist)

    result = _run(wheel, sdist)

    assert result.returncode == 1
    assert "unsafe archive path" in result.stderr


def test_distribution_audit_rejects_weights_and_private_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "vision_xai-0.1.0-py3-none-any.whl"
    sdist = tmp_path / "vision_xai-0.1.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(
        sdist,
        {
            "vision_xai-0.1.0/checkpoints/model.pt": b"weights",
            "vision_xai-0.1.0/notes.txt": b"C:\\Users\\private-user\\dataset\n",
        },
    )

    result = _run(wheel, sdist)

    assert result.returncode == 1
    assert "forbidden distribution member" in result.stderr
