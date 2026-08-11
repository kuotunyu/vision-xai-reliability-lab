"""Build ``results/derived/summary.json``, ``summary.md``, figures, and inject
the generated results block into the READMEs.

Every number comes from raw per-sample results; nothing is entered by hand.
Localization is reported as localization — never as causal faithfulness.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vision_xai.config import AppConfig, config_hash
from vision_xai.errors import ReportError
from vision_xai.paths import derived_dir, raw_eval_dir, resolve_results_dir
from vision_xai.report.aggregate import aggregate_rows, aggregate_values
from vision_xai.report.figures import build_figures
from vision_xai.utils.io import atomic_write_json, read_jsonl

logger = logging.getLogger(__name__)

RESULTS_BEGIN = "<!-- RESULTS:BEGIN -->"
RESULTS_END = "<!-- RESULTS:END -->"


@dataclass(frozen=True)
class ReportResult:
    summary_path: Path
    figure_paths: list[Path]
    readmes_updated: list[Path]
    elapsed_seconds: float


def _split_method_dir(method_dir: str) -> tuple[str, str]:
    if "__" in method_dir:
        method, test_variant = method_dir.split("__", maxsplit=1)
        return method, test_variant
    return method_dir, "clean"


def _collect_rows(cfg: AppConfig) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    eval_root = raw_eval_dir(cfg)
    mapping: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(eval_root.glob("*/*/*.jsonl")):
        variant = path.parent.parent.name
        mapping[(variant, path.parent.name, path.stem)] = read_jsonl(path)
    return mapping


def _train_summaries(cfg: AppConfig) -> dict[str, Any]:
    root = resolve_results_dir(cfg) / "raw" / "train" / cfg.experiment_name
    summaries: dict[str, Any] = {}
    if root.is_dir():
        for final_path in sorted(root.glob("*/final.json")):
            payload = json.loads(final_path.read_text(encoding="utf-8"))
            summaries[str(payload["variant"])] = payload
    return summaries


def _mean_curves(rows: list[dict[str, Any]]) -> dict[str, list[float]] | None:
    if not rows or "fractions" not in rows[0]:
        return None
    fractions = rows[0]["fractions"]
    deletion = np.mean([row["deletion_curve"] for row in rows], axis=0)
    insertion = np.mean([row["insertion_curve"] for row in rows], axis=0)
    return {
        "fractions": [float(f) for f in fractions],
        "deletion_mean": [round(float(v), 6) for v in deletion],
        "insertion_mean": [round(float(v), 6) for v in insertion],
    }


def _scale_note(cfg: AppConfig) -> str:
    """State honestly what scale produced these numbers.

    The shipped configs differ in kind, not merely in size: ``smoke`` trains on a
    handful of images per class and only demonstrates that the mechanics work,
    while ``full`` trains on the entire dataset yet still computes attribution
    metrics over a fixed subset of the evaluation split, because attribution
    methods are expensive. A single fixed sentence would misdescribe one of them
    — claiming "not a benchmark claim" for a full-dataset run understates it,
    and claiming benchmark status for a 3-images-per-class run overstates it.
    """
    if cfg.data.limit_per_class is None:
        training = "trained on the full dataset"
        caveat = (
            "Attribution and reliability metrics therefore describe that fixed "
            "subset, not the entire split."
        )
    else:
        training = f"trained on {cfg.data.limit_per_class} images per class"
        caveat = (
            "Subset-scale numbers validate the pipeline mechanics; they are not benchmark claims."
        )
    return (
        f"experiment '{cfg.experiment_name}': {training}; explanations computed "
        f"on the first {cfg.explain.num_samples} samples of the "
        f"{cfg.explain.split} split. {caveat}"
    )


def build_summary(cfg: AppConfig) -> dict[str, Any]:
    rows_map = _collect_rows(cfg)
    if not rows_map:
        raise ReportError("no evaluation results found — run `evaluate` first")
    seed = cfg.seed
    topk_keys = [f"topk_iou_{fraction}" for fraction in cfg.evaluate.topk_fractions]

    localization: dict[str, dict[str, Any]] = {}
    faithfulness: dict[str, dict[str, Any]] = {}
    randomization: dict[str, dict[str, Any]] = {}
    consistency: dict[str, dict[str, Any]] = {}
    spurious: dict[str, dict[str, Any]] = {}

    for (variant, method_dir, family), rows in rows_map.items():
        method, test_variant = _split_method_dir(method_dir)
        if family == "localization" and test_variant == "clean":
            augmented = [
                {**row, "pointing_rate": 1.0 if row["pointing_hit"] else 0.0} for row in rows
            ]
            localization.setdefault(variant, {})[method] = aggregate_rows(
                augmented, ["energy_in_mask", "pointing_rate", *topk_keys], seed
            )
        elif family == "faithfulness" and test_variant == "clean":
            payload = aggregate_rows(rows, ["deletion_auc", "insertion_auc"], seed)
            curves = _mean_curves(rows)
            faithfulness.setdefault(variant, {})[method] = {**payload, "curves": curves}
        elif family == "randomization":
            randomization.setdefault(variant, {})[method] = aggregate_rows(
                rows, ["abs_spearman"], seed
            )
        elif family == "consistency":
            consistency.setdefault(variant, {})[method] = aggregate_rows(
                rows, ["abs_spearman"], seed
            )
        elif family == "spurious":
            # 'accuracy' IS the correct/all indicator itself (mean of the per-row
            # 'correct' flag), so it has no separate all/correct split to report.
            accuracy = aggregate_values([1.0 if row["correct"] else 0.0 for row in rows], seed)
            patched_rows = [row for row in rows if row["patched_input"]]
            spurious.setdefault(variant, {}).setdefault(method, {})[test_variant] = {
                "accuracy": accuracy,
                "patch_energy_patched_inputs": aggregate_rows(patched_rows, ["patch_energy"], seed),
                "pet_mask_energy": aggregate_rows(rows, ["pet_mask_energy"], seed),
                "num_patched_inputs": len(patched_rows),
            }

    return {
        "schema_version": 1,
        "experiment": cfg.experiment_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_hash": config_hash(cfg),
        "scale_note": _scale_note(cfg),
        "notes": [
            "Localization measures where attributions land; it is not causal faithfulness.",
            "Correct-prediction and all-prediction subsets are reported separately.",
            "Top-k fractions are pre-registered in the config; nothing is tuned on test.",
            "Randomization similarity should be LOW (sensitivity to parameters is healthy).",
        ],
        "train": _train_summaries(cfg),
        "localization": localization,
        "faithfulness": faithfulness,
        "randomization": randomization,
        "consistency": consistency,
        "spurious": spurious,
    }


def _fmt(aggregate: dict[str, Any]) -> str:
    if aggregate["mean"] is None:
        return "n/a"
    return f"{aggregate['mean']:.3f} [{aggregate['ci_low']:.3f}, {aggregate['ci_high']:.3f}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def results_block(summary: dict[str, Any], topk_key: str) -> str:
    parts: list[str] = [
        f"**Experiment `{summary['experiment']}`** — generated {summary['generated_at']}",
        "",
        f"_{summary['scale_note']}_",
        "",
    ]
    if summary["train"]:
        rows = [
            [
                variant,
                f"{payload['val_accuracy']:.3f}",
                f"{payload['val_macro_f1']:.3f}",
                f"{payload['val_ece']:.3f}",
                str(payload["device"]),
                f"{payload['elapsed_seconds']:.0f}s",
                "n/a" if payload["peak_vram_mb"] is None else f"{payload['peak_vram_mb']}MB",
            ]
            for variant, payload in sorted(summary["train"].items())
        ]
        parts += [
            "**Classification (clean validation split)**",
            "",
            _table(
                ["variant", "val acc", "val macro-F1", "val ECE", "device", "time", "peak VRAM"],
                rows,
            ),
            "",
        ]
    if summary["localization"]:
        rows = [
            [
                variant,
                method,
                _fmt(payload["all"]["energy_in_mask"]),
                _fmt(payload["all"]["pointing_rate"]),
                _fmt(payload["all"][topk_key]),
            ]
            for variant, methods in sorted(summary["localization"].items())
            for method, payload in sorted(methods.items())
        ]
        parts += [
            "**Localization (all predictions; mean [95% bootstrap CI]) — not causal faithfulness**",
            "",
            _table(
                [
                    "variant",
                    "method",
                    "energy in mask",
                    "pointing game",
                    topk_key.replace("_", " "),
                ],
                rows,
            ),
            "",
        ]
    if summary["faithfulness"]:
        rows = [
            [
                variant,
                method,
                _fmt(payload["all"]["deletion_auc"]),
                _fmt(payload["all"]["insertion_auc"]),
            ]
            for variant, methods in sorted(summary["faithfulness"].items())
            for method, payload in sorted(methods.items())
        ]
        parts += [
            "**Faithfulness (deletion lower / insertion higher = better)**",
            "",
            _table(["variant", "method", "deletion AUC", "insertion AUC"], rows),
            "",
        ]
    if summary["randomization"] or summary["consistency"]:
        rows = []
        variants = sorted(set(summary["randomization"]) | set(summary["consistency"]))
        for variant in variants:
            methods = sorted(
                set(summary["randomization"].get(variant, {}))
                | set(summary["consistency"].get(variant, {}))
            )
            for method in methods:
                rand = summary["randomization"].get(variant, {}).get(method)
                cons = summary["consistency"].get(variant, {}).get(method)
                rows.append(
                    [
                        variant,
                        method,
                        _fmt(rand["all"]["abs_spearman"]) if rand else "n/a",
                        _fmt(cons["all"]["abs_spearman"]) if cons else "n/a",
                    ]
                )
        parts += [
            "**Sanity & stability (|Spearman|): randomization LOW, flip consistency HIGH**",
            "",
            _table(["variant", "method", "randomization sim.", "flip consistency"], rows),
            "",
        ]
    if summary["spurious"]:
        rows = [
            [
                variant,
                method,
                test_variant,
                _fmt(payload["accuracy"]),
                _fmt(payload["patch_energy_patched_inputs"]["all"]["patch_energy"]),
                _fmt(payload["pet_mask_energy"]["all"]["pet_mask_energy"]),
            ]
            for variant, methods in sorted(summary["spurious"].items())
            for method, by_tv in sorted(methods.items())
            for test_variant, payload in sorted(by_tv.items())
        ]
        parts += [
            "**Spurious corner patch (patched-trained models on three test variants)**",
            "",
            _table(
                [
                    "variant",
                    "method",
                    "test variant",
                    "accuracy",
                    "patch energy (patched inputs)",
                    "pet-mask energy",
                ],
                rows,
            ),
            "",
        ]
    parts.append("_A heatmap is not proof of causal reasoning._")
    return "\n".join(parts)


def inject_results(readme_path: Path, block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    begin = text.find(RESULTS_BEGIN)
    end = text.find(RESULTS_END)
    if begin == -1 or end == -1 or end < begin:
        raise ReportError(f"{readme_path} is missing the {RESULTS_BEGIN}/{RESULTS_END} markers")
    updated = text[: begin + len(RESULTS_BEGIN)] + "\n" + block + "\n" + text[end:]
    readme_path.write_text(updated, encoding="utf-8")


def build_report(cfg: AppConfig, readme_paths: Sequence[Path] | None = None) -> ReportResult:
    start = time.monotonic()
    summary = build_summary(cfg)
    derived = derived_dir(cfg)
    derived.mkdir(parents=True, exist_ok=True)
    summary_path = derived / "summary.json"
    atomic_write_json(summary_path, summary)

    figures = build_figures(summary, derived / "figures", Path("assets") / "figures")

    topk_key = (
        f"topk_iou_{cfg.evaluate.topk_fractions[min(1, len(cfg.evaluate.topk_fractions) - 1)]}"
    )
    block = results_block(summary, topk_key)
    (derived / "summary.md").write_text(block + "\n", encoding="utf-8")

    resolved_readmes = list(
        readme_paths if readme_paths is not None else (Path("README.md"), Path("README_zh-TW.md"))
    )
    updated: list[Path] = []
    for readme in resolved_readmes:
        if readme.is_file():
            inject_results(readme, block)
            updated.append(readme)
        else:
            logger.warning("README %s not found; skipping injection", readme)
    elapsed = time.monotonic() - start
    logger.info(
        "report complete: %s, %d figures, %d READMEs updated (%.1fs)",
        summary_path,
        len(figures),
        len(updated),
        elapsed,
    )
    return ReportResult(
        summary_path=summary_path,
        figure_paths=figures,
        readmes_updated=updated,
        elapsed_seconds=elapsed,
    )
