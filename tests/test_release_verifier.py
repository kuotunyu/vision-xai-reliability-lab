from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BEGIN = "<!-- RESULTS:BEGIN -->"
RESULTS_END = "<!-- RESULTS:END -->"


def _run_verifier(root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "verify_release.py"),
            "--root",
            str(root),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _copy_evidence(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    for directory in ("assets", "release", "results", "schemas", "showcase"):
        shutil.copytree(REPO_ROOT / directory, root / directory)
    for filename in ("README.md", "README_en.md"):
        shutil.copy2(REPO_ROOT / filename, root / filename)
    for filename in (
        "ARTIFACTS.md",
        "DATA_CARD.md",
        "FAILURES.md",
        "MODEL_CARD.md",
        "OWNER_ACTIONS.md",
    ):
        shutil.copy2(REPO_ROOT / filename, root / filename)
    return root


def _refresh_summary_manifest(root: Path) -> None:
    summary_path = root / "results" / "derived" / "summary.json"
    payload = summary_path.read_bytes().replace(b"\r\n", b"\n")
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["artifacts"] if item["path"] == "results/derived/summary.json"
    )
    entry["bytes"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _initialize_candidate_git(
    root: Path, *, name: str = "kuotunyu", email: str = "61350295+kuotunyu@users.noreply.github.com"
) -> None:
    shutil.copy2(REPO_ROOT / "LICENSE", root / "LICENSE")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", name)
    _git(root, "config", "user.email", email)
    _git(root, "add", "--all")
    _git(root, "commit", "-m", "test fixture")


def test_release_verifier_accepts_committed_evidence() -> None:
    result = _run_verifier(REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "artifact hashes" in result.stdout
    assert "claim invariants" in result.stdout
    assert "README synchronization" in result.stdout
    assert "CUDA resume canary evidence" in result.stdout


def test_release_manifest_covers_portfolio_visuals() -> None:
    manifest = json.loads(
        (REPO_ROOT / "release" / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    paths = {entry["path"] for entry in manifest["artifacts"]}

    assert "assets/portfolio/hero.png" in paths
    assert "assets/portfolio/social-preview.png" in paths


def test_readmes_lead_with_portfolio_evidence() -> None:
    headings = {"README.md": "## 專案進度", "README_en.md": "## Project status"}
    for filename, status_heading in headings.items():
        text = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert "![Vision XAI reliability evidence](assets/portfolio/hero.png)" in text
        assert text.index("assets/portfolio/hero.png") < text.index(status_heading)
        assert "github.com/kuotunyu/vision-xai-reliability-lab/actions/workflows/ci.yml" in text
        assert "kuotunyu.github.io/vision-xai-reliability-lab/" in text
        assert "(showcase/)" in text

    primary = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    english = (REPO_ROOT / "README_en.md").read_text(encoding="utf-8")
    assert "[English version](README_en.md)" in primary
    assert "[正體中文](README.md)" in english


def test_owner_actions_capture_github_portfolio_handoff() -> None:
    text = (REPO_ROOT / "OWNER_ACTIONS.md").read_text(encoding="utf-8")
    expected = (
        "Repository description",
        "以可靠性為核心的 XAI benchmark",
        "computer-vision",
        "trustworthy-ai",
        "assets/portfolio/social-preview.png",
        "GitHub Actions",
        "push `main`",
        "Pin the repository",
    )
    for phrase in expected:
        assert phrase in text


def test_generated_result_tables_are_collapsible() -> None:
    for filename in ("README.md", "README_en.md"):
        text = (REPO_ROOT / filename).read_text(encoding="utf-8")
        details_start = text.rfind("<details>", 0, text.index(RESULTS_BEGIN))
        assert details_start >= 0
        assert text.index(RESULTS_END) < text.index("</details>", details_start)


def test_release_verifier_accepts_clean_candidate_repository(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    _initialize_candidate_git(root)
    result = _run_verifier(root, "--git")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Git identity and history" in result.stdout
    assert "privacy and tracked-file boundary" in result.stdout
    assert "Markdown local links" in result.stdout


def test_release_verifier_rejects_tampered_artifact(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    payload = summary_path.read_bytes()
    summary_path.write_bytes(payload.replace(b'"full"', b'"null"', 1))

    result = _run_verifier(root)

    assert result.returncode == 1
    assert "sha256 mismatch: results/derived/summary.json" in result.stderr


def test_release_verifier_accepts_lf_checkout_of_text_artifacts(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    for relative in (
        "results/derived/summary.json",
        "results/derived/summary.md",
        "results/raw/data_prepare/full/fingerprint.json",
        "results/raw/data_prepare/full/patch_summary.json",
        "results/raw/data_prepare/full/split_summary.json",
    ):
        path = root / relative
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))

    result = _run_verifier(root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_release_verifier_rejects_false_center_prior_claim(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["localization"]["cnn"]["center"]["all"]["pointing_rate"]["mean"] = 0.0
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_summary_manifest(root)

    result = _run_verifier(root)

    assert result.returncode == 1
    assert "center-prior pointing-game claim is false for cnn" in result.stderr


def test_release_verifier_rejects_false_ig_randomization_claim(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["randomization"]["vit"]["integrated_gradients"]["all"]["abs_spearman"]["mean"] = 0.0
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_summary_manifest(root)

    result = _run_verifier(root)

    assert result.returncode == 1
    assert "IG randomization-sanity conclusion is false for vit" in result.stderr


def test_release_verifier_rejects_false_spurious_negative_result(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["spurious"]["cnn_patched"]["gradcam"]["correlated"]["accuracy"]["mean"] = 1.0
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_summary_manifest(root)

    result = _run_verifier(root)

    assert result.returncode == 1
    assert "spurious-patch negative-result accuracy gap changed" in result.stderr


def test_release_verifier_rejects_readme_drift(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    readme_path = root / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8").replace(
            "**Experiment `full`**", "**Experiment `smoke`**", 1
        ),
        encoding="utf-8",
    )

    result = _run_verifier(root)

    assert result.returncode == 1
    assert "README.md result block differs from summary.md" in result.stderr


def test_release_verifier_rejects_tracked_private_progress(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    _initialize_candidate_git(root)
    (root / "PROGRESS.md").write_text("private handoff\n", encoding="utf-8")
    _git(root, "add", "PROGRESS.md")

    result = _run_verifier(root, "--git", "--allow-dirty")

    assert result.returncode == 1
    assert "forbidden tracked path: PROGRESS.md" in result.stderr


def test_release_verifier_rejects_unexpected_commit_identity(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    _initialize_candidate_git(root, name="Wrong User", email="wrong@example.com")

    result = _run_verifier(root, "--git")

    assert result.returncode == 1
    assert "unexpected author or committer" in result.stderr


def test_release_verifier_rejects_broken_local_markdown_link(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    readme = root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\n[missing](docs/not-present.md)\n",
        encoding="utf-8",
    )
    _initialize_candidate_git(root)

    result = _run_verifier(root, "--git")

    assert result.returncode == 1
    assert "broken local Markdown link" in result.stderr
