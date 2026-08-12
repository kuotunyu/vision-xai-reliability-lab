from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml
from tools.build_showcase import ShowcaseError, audit_showcase, build_showcase

REPO_ROOT = Path(__file__).resolve().parents[1]


class _ShowcaseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.buttons: list[dict[str, str | None]] = []
        self.images: list[dict[str, str | None]] = []
        self.local_resources: list[str] = []
        self.html_lang: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.html_lang = attributes.get("lang")
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
    assert parser.html_lang == "zh-TW"
    assert {button.get("data-model") for button in parser.buttons} == {"cnn", "vit"}
    assert all(image.get("alt") for image in parser.images)
    assert parser.local_resources == ["styles.css", "app.js"]
    assert "不是 live inference demo" in html
    assert "固定 500 筆 attribution subset" in html
    assert "localization 不是 causal faithfulness" in html
    assert 'data-metric="spurious-patch-energy"' in html


def test_showcase_uses_zh_tw_for_interface_scaffolding() -> None:
    html = (REPO_ROOT / "showcase" / "index.html").read_text(encoding="utf-8")

    assert "完整規模 / 2026.07.25" in html
    assert "<dt>訓練</dt>" in html
    assert "<dt>信賴區間</dt>" in html
    assert "Localization 與 baseline" in html
    assert "Model-randomization sanity check" in html
    assert "Spurious-patch experiment" in html
    assert "已提交的完整規模彙總" in html
    assert "公開 artifact 的驗證範圍" in html
    assert "實驗結果、證據邊界與限制" in html
    assert "開啟正式結果 JSON" in html
    assert "Localization trap" not in html
    assert "Sanity failure" not in html
    assert "Negative result" not in html
    assert "Committed full-scale aggregates" not in html
    assert "Public evidence boundary" not in html
    assert "開啟 CANONICAL JSON" not in html


def test_showcase_uses_research_report_copy_and_readable_density() -> None:
    html = (REPO_ROOT / "showcase" / "index.html").read_text(encoding="utf-8")
    css = (REPO_ROOT / "showcase" / "styles.css").read_text(encoding="utf-8")

    headings = [
        " ".join(re.sub(r"<[^>]+>", "", heading).split())
        for heading in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html, flags=re.DOTALL)
    ]

    assert headings[0] == "Vision XAI 的可靠性評估"
    assert "主要實驗結果與解讀限制" in headings
    assert "各項 metric 回答不同的評估問題" in headings
    assert "CUDA resume canary 的 state-equivalence 檢查" in headings
    assert all("\uff1f" not in heading and "\uff01" not in heading for heading in headings)
    assert "三個可被推翻的結果" not in html
    assert "好看\uff0c不是一項 metric" not in html
    assert "<strong>Google Colab · NVIDIA L4</strong>" in html

    assert re.search(r"html\s*\{[^}]*font-size:\s*18px", css, flags=re.DOTALL)
    assert re.search(r"--max:\s*76rem", css)
    assert re.search(
        r"h1\s*\{[^}]*font-size:\s*clamp\(2rem,\s*3vw,\s*2\.7rem\)",
        css,
        flags=re.DOTALL,
    )
    assert re.search(
        r"h2\s*\{[^}]*font-size:\s*clamp\(1\.45rem,\s*1\.9vw,\s*1\.85rem\)",
        css,
        flags=re.DOTALL,
    )
    assert re.search(
        r"\.findings,\s*\.evidence,\s*\.cuda,\s*\.boundary\s*"
        r"\{\s*padding:\s*1\.75rem 0;\s*\}",
        css,
    )
    assert re.search(r"\.masthead\s*\{[^}]*align-items:\s*start", css, flags=re.DOTALL)
    assert re.search(
        r"\.run-stamp\s*\{[^}]*display:\s*grid[^}]*grid-template-columns:\s*1fr auto",
        css,
        flags=re.DOTALL,
    )


def test_showcase_avoids_decorative_scaffolding_and_system_display_type() -> None:
    html = (REPO_ROOT / "showcase" / "index.html").read_text(encoding="utf-8")
    css = (REPO_ROOT / "showcase" / "styles.css").read_text(encoding="utf-8")

    assert 'class="eyebrow"' not in html
    assert 'class="page-grid"' not in html
    assert "Bahnschrift" not in css
    assert "DIN Alternate" not in css
    assert "mask-image" not in css
    assert "01 / Localization trap" not in html
    assert "02 / Sanity failure" not in html
    assert "03 / Negative result" not in html


def test_build_showcase_exports_only_the_public_allowlist(tmp_path: Path) -> None:
    output = tmp_path / "public"

    exported = build_showcase(REPO_ROOT, output)
    names = {path.relative_to(output).as_posix() for path in exported}

    assert len(names) == 18
    assert {"index.html", "styles.css", "app.js"} <= names
    assert {"data/summary.json", "data/cuda-resume-canary.json"} <= names
    assert "assets/portfolio/hero.png" in names
    assert not any(name.endswith((".ckpt", ".npz", ".pt", ".pth")) for name in names)


def test_docker_copies_only_safe_dashboard_evidence() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "COPY results/derived/summary.json results/derived/summary.json" in dockerfile
    assert "COPY release/cuda-resume-canary.json release/cuda-resume-canary.json" in dockerfile
    assert "COPY assets/figures/ assets/figures/" in dockerfile
    assert "COPY data/" not in dockerfile
    assert "COPY checkpoints/" not in dockerfile
    assert "!results/derived/summary.json" in dockerignore
    assert "!assets/figures/" in dockerignore


def test_pages_workflow_builds_then_deploys_without_project_secrets() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "pages.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    build_steps = jobs["build"]["steps"]
    build_runs = "\n".join(step.get("run", "") for step in build_steps)
    build_actions = {step.get("uses", "") for step in build_steps}
    assert "tools/build_showcase.py" in build_runs
    assert any(action.startswith("actions/upload-pages-artifact@") for action in build_actions)
    assert not any(action.startswith("actions/configure-pages@") for action in build_actions)
    assert jobs["build"]["permissions"] == {"contents": "read"}

    deploy = jobs["deploy"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert "github.event_name == 'push'" in deploy["if"]
    deploy_actions = {step.get("uses", "") for step in deploy["steps"]}
    assert "actions/configure-pages@v6" in deploy_actions
    assert "actions/deploy-pages@v5" in deploy_actions
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "secrets" not in workflow_path.read_text(encoding="utf-8")


def test_primary_ci_builds_showcase_without_deployment() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    assert workflow["permissions"] == {"contents": "read"}
    quality_steps = workflow["jobs"]["quality"]["steps"]
    commands = "\n".join(step.get("run", "") for step in quality_steps)

    assert "tools/build_showcase.py" in commands
    assert not any(
        step.get("uses", "").startswith("actions/deploy-pages@") for step in quality_steps
    )


def test_workflows_pin_setup_uv_to_resolvable_immutable_commit() -> None:
    for filename in ("ci.yml", "pages.yml"):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
        )
        setup_refs = [
            step["uses"].partition("@")[2]
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if step.get("uses", "").startswith("astral-sh/setup-uv@")
        ]

        assert len(setup_refs) == 1
        assert re.fullmatch(r"[0-9a-f]{40}", setup_refs[0])
