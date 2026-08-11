from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_verifier(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "verify_release.py"), "--root", str(root)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _copy_evidence(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    for directory in ("assets", "release", "results", "schemas"):
        shutil.copytree(REPO_ROOT / directory, root / directory)
    for filename in ("README.md", "README_zh-TW.md"):
        shutil.copy2(REPO_ROOT / filename, root / filename)
    return root


def _refresh_summary_manifest(root: Path) -> None:
    summary_path = root / "results" / "derived" / "summary.json"
    payload = summary_path.read_bytes()
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["artifacts"] if item["path"] == "results/derived/summary.json"
    )
    entry["bytes"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_release_verifier_accepts_committed_evidence() -> None:
    result = _run_verifier(REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "artifact hashes" in result.stdout
    assert "claim invariants" in result.stdout
    assert "README synchronization" in result.stdout


def test_release_verifier_rejects_tampered_artifact(tmp_path: Path) -> None:
    root = _copy_evidence(tmp_path)
    summary_path = root / "results" / "derived" / "summary.json"
    payload = summary_path.read_bytes()
    summary_path.write_bytes(payload.replace(b'"full"', b'"null"', 1))

    result = _run_verifier(root)

    assert result.returncode == 1
    assert "sha256 mismatch: results/derived/summary.json" in result.stderr


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
