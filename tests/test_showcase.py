from __future__ import annotations

from pathlib import Path

import pytest
from tools.build_showcase import ShowcaseError, audit_showcase, build_showcase

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_showcase_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    foreign = output / "foreign.txt"
    foreign.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(ShowcaseError, match="output directory must be empty"):
        build_showcase(REPO_ROOT, output)

    assert foreign.read_text(encoding="utf-8") == "do not overwrite\n"


def test_audit_showcase_rejects_private_absolute_path(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "index.html").write_text(
        "<p>C:\\Users\\private-user\\dataset</p>\n", encoding="utf-8"
    )

    with pytest.raises(ShowcaseError, match="private absolute path"):
        audit_showcase(output)


def test_audit_showcase_rejects_model_weights(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "model.pt").write_bytes(b"not really weights")

    with pytest.raises(ShowcaseError, match="forbidden showcase file"):
        audit_showcase(output)


def test_audit_showcase_accepts_small_static_site(tmp_path: Path) -> None:
    output = tmp_path / "public"
    assets = output / "assets"
    assets.mkdir(parents=True)
    (output / "index.html").write_text("<!doctype html><title>Evidence</title>\n", encoding="utf-8")
    (assets / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    audited = audit_showcase(output)

    assert audited == ["assets/figure.png", "index.html"]
