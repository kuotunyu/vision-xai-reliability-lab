"""Read-only verification of committed release evidence.

This verifier intentionally uses only the Python standard library so it can run
before installing the project. It validates the narrow public evidence boundary;
it does not regenerate the full experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

RESULTS_BEGIN = "<!-- RESULTS:BEGIN -->"
RESULTS_END = "<!-- RESULTS:END -->"
BASELINES = {"center", "random", "uniform"}
ATTRIBUTION_SUBSET_SIZE = 500
MAX_SPURIOUS_ACCURACY_SPREAD = 0.02


class VerificationError(RuntimeError):
    """A release invariant is missing or contradicted."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object in {path}")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise VerificationError(f"unsafe artifact path: {relative!r}")
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VerificationError(f"artifact escapes release root: {relative!r}") from exc
    return path


def verify_artifact_hashes(root: Path) -> int:
    manifest = _load_json(root / "release" / "artifact-manifest.json")
    if manifest.get("schema_version") != 1:
        raise VerificationError("unsupported artifact manifest schema_version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise VerificationError("artifact manifest has no artifacts")
    seen: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise VerificationError("artifact manifest entry is not an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise VerificationError(f"invalid or duplicate artifact path: {relative!r}")
        seen.add(relative)
        path = _safe_path(root, relative)
        if not path.is_file():
            raise VerificationError(f"missing artifact: {relative}")
        payload = path.read_bytes()
        if len(payload) != entry.get("bytes"):
            raise VerificationError(f"size mismatch: {relative}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("sha256"):
            raise VerificationError(f"sha256 mismatch: {relative}")
    return len(artifacts)


def verify_summary_schema(root: Path) -> dict[str, Any]:
    schema = _load_json(root / "schemas" / "full-summary.schema.json")
    manifest_schema = _load_json(root / "schemas" / "artifact-manifest.schema.json")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise VerificationError("full summary schema must use JSON Schema 2020-12")
    if manifest_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise VerificationError("artifact manifest schema must use JSON Schema 2020-12")

    summary = _load_json(root / "results" / "derived" / "summary.json")
    required = {
        "schema_version",
        "experiment",
        "config_hash",
        "generated_at",
        "scale_note",
        "notes",
        "train",
        "localization",
        "faithfulness",
        "randomization",
        "consistency",
        "spurious",
    }
    missing = sorted(required - summary.keys())
    if missing:
        raise VerificationError(f"summary is missing required keys: {', '.join(missing)}")
    if summary["schema_version"] != 1 or summary["experiment"] != "full":
        raise VerificationError("summary is not schema v1 for experiment 'full'")
    scale_note = summary.get("scale_note")
    if not isinstance(scale_note, str) or "first 500 samples" not in scale_note:
        raise VerificationError("summary does not disclose the fixed 500-sample subset")
    if "not the entire split" not in scale_note:
        raise VerificationError("summary does not distinguish the subset from the test split")
    return summary


def _mean(payload: dict[str, Any]) -> float:
    value = payload.get("mean")
    if not isinstance(value, (int, float)):
        raise VerificationError("expected a numeric aggregate mean")
    return float(value)


def verify_claim_invariants(summary: dict[str, Any]) -> None:
    localization = summary["localization"]
    for variant in ("cnn", "vit"):
        methods = localization[variant]
        center = _mean(methods["center"]["all"]["pointing_rate"])
        attribution_scores = [
            _mean(payload["all"]["pointing_rate"])
            for method, payload in methods.items()
            if method not in BASELINES
        ]
        if not attribution_scores or center <= max(attribution_scores):
            raise VerificationError(f"center-prior pointing-game claim is false for {variant}")
        if methods["center"]["all"]["pointing_rate"].get("n") != ATTRIBUTION_SUBSET_SIZE:
            raise VerificationError(f"unexpected localization subset size for {variant}")

    randomization = summary["randomization"]
    for variant in ("cnn", "vit"):
        methods = randomization[variant]
        ig = _mean(methods["integrated_gradients"]["all"]["abs_spearman"])
        comparators = [
            _mean(payload["all"]["abs_spearman"])
            for method, payload in methods.items()
            if method != "integrated_gradients"
        ]
        if not comparators or ig <= max(comparators):
            raise VerificationError(f"IG randomization-sanity conclusion is false for {variant}")

    spurious = summary["spurious"]
    for variant, methods in spurious.items():
        for method, test_variants in methods.items():
            accuracies = [_mean(payload["accuracy"]) for payload in test_variants.values()]
            if max(accuracies) - min(accuracies) > MAX_SPURIOUS_ACCURACY_SPREAD:
                raise VerificationError(
                    f"spurious-patch negative-result accuracy gap changed for {variant}/{method}"
                )


def _result_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if RESULTS_BEGIN not in text or RESULTS_END not in text:
        raise VerificationError(f"missing result markers in {path.name}")
    return text.split(RESULTS_BEGIN, 1)[1].split(RESULTS_END, 1)[0].strip()


def verify_readme_synchronization(root: Path) -> None:
    generated = (root / "results" / "derived" / "summary.md").read_text(encoding="utf-8").strip()
    for name in ("README.md", "README_zh-TW.md"):
        if _result_block(root / name) != generated:
            raise VerificationError(f"{name} result block differs from summary.md")


def verify(root: Path) -> list[str]:
    root = root.resolve()
    artifact_count = verify_artifact_hashes(root)
    summary = verify_summary_schema(root)
    verify_claim_invariants(summary)
    verify_readme_synchronization(root)
    return [
        f"PASS artifact hashes ({artifact_count} files)",
        "PASS JSON schemas and full-summary shape",
        "PASS claim invariants",
        "PASS README synchronization",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args()
    try:
        checks = verify(args.root)
    except VerificationError as exc:
        sys.stderr.write(f"FAIL {exc}\n")
        return 1
    sys.stdout.write("\n".join(checks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
