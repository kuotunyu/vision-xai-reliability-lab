"""Read-only verification of committed release evidence.

This verifier intentionally uses only the Python standard library so it can run
before installing the project. It validates the narrow public evidence boundary;
it does not regenerate the full experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

RESULTS_BEGIN = "<!-- RESULTS:BEGIN -->"
RESULTS_END = "<!-- RESULTS:END -->"
BASELINES = {"center", "random", "uniform"}
ATTRIBUTION_SUBSET_SIZE = 500
MAX_SPURIOUS_ACCURACY_SPREAD = 0.02
MAX_TRACKED_FILE_BYTES = 1024 * 1024
GIT_LOG_FIELD_COUNT = 4
EXPECTED_GIT_IDENTITY = "kuotunyu <61350295+kuotunyu@users.noreply.github.com>"
FORBIDDEN_ROOT_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "checkpoints",
    "data",
    "notebooks",
}
FORBIDDEN_EXACT_PATHS = {"PROGRESS.md", "RELEASE_AUDIT.md"}
FORBIDDEN_WEIGHT_SUFFIXES = {".ckpt", ".pt", ".pth"}
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
        hash_mode = entry.get("hash_mode")
        if hash_mode == "text-lf":
            payload = payload.replace(b"\r\n", b"\n")
        elif hash_mode != "binary":
            raise VerificationError(f"unsupported hash mode for {relative}: {hash_mode!r}")
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


def verify_cuda_canary_evidence(root: Path) -> None:
    schema = _load_json(root / "schemas" / "cuda-resume-canary.schema.json")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise VerificationError("CUDA canary schema must use JSON Schema 2020-12")
    evidence = _load_json(root / "release" / "cuda-resume-canary.json")
    if evidence.get("schema_version") != 1 or evidence.get("status") != "PASS":
        raise VerificationError("CUDA resume canary is not a schema-v1 PASS")
    scope = evidence.get("scope")
    if not isinstance(scope, dict) or scope.get("not_full_scale") is not True:
        raise VerificationError("CUDA canary does not disclose its non-full-scale scope")
    execution = evidence.get("execution")
    if not isinstance(execution, dict) or execution.get("external_compute_processes_at_start") != 0:
        raise VerificationError("CUDA canary did not start with zero external compute processes")
    if execution.get("process_isolation") != "three sequential fresh Python worker processes":
        raise VerificationError("CUDA canary phases were not process-isolated")
    comparisons = evidence.get("comparisons")
    if not isinstance(comparisons, dict):
        raise VerificationError("CUDA canary comparisons are missing")
    for key in (
        "head_state_exact",
        "optimizer_state_exact",
        "grad_scaler_state_exact",
        "stable_metrics_exact",
    ):
        if comparisons.get(key) is not True:
            raise VerificationError(f"CUDA resume canary comparison failed: {key}")
    if comparisons.get("differences") != []:
        raise VerificationError("CUDA resume canary reports state or metric differences")
    scheduler = comparisons.get("scheduler_state")
    if not isinstance(scheduler, dict) or scheduler.get("status") != "not_applicable":
        raise VerificationError("CUDA canary scheduler boundary is not explicit")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )
    if result.returncode != 0:
        raise VerificationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _tracked_paths(root: Path) -> list[str]:
    return [path for path in _git(root, "ls-files", "-z").split("\0") if path]


def verify_git_identity_and_history(root: Path, *, allow_dirty: bool) -> None:
    if _git(root, "branch", "--show-current").strip() != "main":
        raise VerificationError("release candidate must be on branch main")
    if _git(root, "remote").strip():
        raise VerificationError("release candidate must not have a Git remote")
    if _git(root, "tag", "--list").strip():
        raise VerificationError("release candidate must not have tags")
    if not allow_dirty and _git(root, "status", "--porcelain=v1").strip():
        raise VerificationError("release candidate working tree is not clean")

    records = _git(
        root,
        "log",
        "--format=%H%x1f%an <%ae>%x1f%cn <%ce>%x1f%B%x1e",
    )
    if not records.strip():
        raise VerificationError("release candidate has no commits")
    trailer_pattern = re.compile(
        r"(?im)^(?:co-authored-by|signed-off-by|reviewed-by|tested-by|assisted-by):"
    )
    for record in records.split("\x1e"):
        if not record.strip():
            continue
        fields = record.strip().split("\x1f", 3)
        if len(fields) != GIT_LOG_FIELD_COUNT:
            raise VerificationError("could not parse Git history record")
        commit, author, committer, body = fields
        if author != EXPECTED_GIT_IDENTITY or committer != EXPECTED_GIT_IDENTITY:
            raise VerificationError(f"unexpected author or committer in {commit}")
        if trailer_pattern.search(body):
            raise VerificationError(f"forbidden contributor trailer in {commit}")


def verify_privacy_and_tracked_boundary(root: Path) -> int:
    tracked = _tracked_paths(root)
    if not tracked:
        raise VerificationError("release candidate tracks no files")
    windows_user_path = re.compile(r"(?i)[a-z]:[\\/]Users[\\/][A-Za-z0-9._-]+[\\/]")
    posix_user_path = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")
    secret_patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?" + "PRIVATE KEY"),
    )
    for relative in tracked:
        pure = PurePosixPath(relative)
        if relative in FORBIDDEN_EXACT_PATHS or pure.parts[0] in FORBIDDEN_ROOT_NAMES:
            raise VerificationError(f"forbidden tracked path: {relative}")
        if pure.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES:
            raise VerificationError(f"tracked model/checkpoint file: {relative}")
        if relative.startswith("results/raw/") and relative not in ALLOWED_RAW_RESULTS:
            raise VerificationError(f"tracked raw runtime result: {relative}")
        path = _safe_path(root, relative)
        size = path.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            raise VerificationError(f"tracked file exceeds 1 MiB: {relative}")
        if pure.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise VerificationError(f"tracked text is not UTF-8: {relative}") from exc
        if windows_user_path.search(text) or posix_user_path.search(text):
            raise VerificationError(f"private absolute path in tracked file: {relative}")
        if any(pattern.search(text) for pattern in secret_patterns):
            raise VerificationError(f"possible secret in tracked file: {relative}")
    if not (root / "LICENSE").is_file():
        raise VerificationError("LICENSE is missing")
    return len(tracked)


def verify_markdown_links(root: Path) -> int:
    """Require every tracked local Markdown link to resolve inside the repository."""
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    checked = 0
    for relative in _tracked_paths(root):
        if PurePosixPath(relative).suffix.lower() != ".md":
            continue
        source = _safe_path(root, relative)
        for raw_target in link_pattern.findall(source.read_text(encoding="utf-8")):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split(maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            local = unquote(parsed.path)
            if not local:
                continue
            destination = (source.parent / Path(local)).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise VerificationError(
                    f"local Markdown link escapes repository: {relative} -> {target}"
                ) from exc
            if not destination.exists():
                raise VerificationError(f"broken local Markdown link: {relative} -> {target}")
            checked += 1
    return checked


def verify(root: Path, *, git: bool = False, allow_dirty: bool = False) -> list[str]:
    root = root.resolve()
    artifact_count = verify_artifact_hashes(root)
    summary = verify_summary_schema(root)
    verify_claim_invariants(summary)
    verify_readme_synchronization(root)
    verify_cuda_canary_evidence(root)
    checks = [
        f"PASS artifact hashes ({artifact_count} files)",
        "PASS JSON schemas and full-summary shape",
        "PASS claim invariants",
        "PASS README synchronization",
        "PASS CUDA resume canary evidence",
    ]
    if git:
        verify_git_identity_and_history(root, allow_dirty=allow_dirty)
        tracked_count = verify_privacy_and_tracked_boundary(root)
        link_count = verify_markdown_links(root)
        checks.extend(
            [
                "PASS Git identity and history",
                f"PASS privacy and tracked-file boundary ({tracked_count} files)",
                f"PASS Markdown local links ({link_count} links)",
            ]
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--git", action="store_true", help="also audit local Git state/history")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dirty tree during development (final release checks must omit this)",
    )
    args = parser.parse_args()
    try:
        checks = verify(args.root, git=args.git, allow_dirty=args.allow_dirty)
    except VerificationError as exc:
        sys.stderr.write(f"FAIL {exc}\n")
        return 1
    sys.stdout.write("\n".join(checks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
