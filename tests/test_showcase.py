from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest
from tools.build_showcase import ShowcaseError, audit_showcase, build_showcase

REPO_ROOT = Path(__file__).resolve().parents[1]


class _ShowcaseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.buttons: list[dict[str, str | None]] = []
        self.images: list[dict[str, str | None]] = []
        self.local_resources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        if tag == "button":
            self.buttons.append(attributes)
        if tag == "img":
            self.images.append(attributes)
        if tag == "script" and attributes.get("src"):
            self.local_resources.append(str(attributes["src"]))
        if tag == "link" and attributes.get("rel") == "stylesheet" and attributes.get("href"):
            self.local_resources.append(str(attributes["href"]))


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


def test_showcase_source_has_accessible_evidence_structure() -> None:
    html = (REPO_ROOT / "showcase" / "index.html").read_text(encoding="utf-8")
    parser = _ShowcaseParser()
    parser.feed(html)

    assert {"main", "findings", "evidence", "cuda", "boundary"} <= parser.ids
    assert {button.get("data-model") for button in parser.buttons} == {"cnn", "vit"}
    assert all(image.get("alt") for image in parser.images)
    assert parser.local_resources == ["styles.css", "app.js"]
    assert "not a live inference demo" in html
    assert "fixed 500-sample" in html
    assert "localization is not causal faithfulness" in html.lower()


def test_build_showcase_exports_only_the_public_allowlist(tmp_path: Path) -> None:
    output = tmp_path / "public"

    exported = build_showcase(REPO_ROOT, output)
    names = {path.relative_to(output).as_posix() for path in exported}

    assert len(names) == 18
    assert {"index.html", "styles.css", "app.js"} <= names
    assert {"data/summary.json", "data/cuda-resume-canary.json"} <= names
    assert "assets/portfolio/hero.png" in names
    assert not any(name.endswith((".ckpt", ".npz", ".pt", ".pth")) for name in names)
