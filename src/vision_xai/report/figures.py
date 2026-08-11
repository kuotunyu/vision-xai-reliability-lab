"""Matplotlib figures generated from the aggregated summary (Agg backend).

Figures are written to ``results/derived/figures`` and mirrored into
``assets/figures`` so the README can reference committed images; both copies
come from the same raw results.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _faithfulness_figure(plt: Any, variant: str, methods: dict[str, Any], path: Path) -> None:
    figure, (ax_del, ax_ins) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for method, payload in sorted(methods.items()):
        curves = payload.get("curves")
        if not curves:
            continue
        ax_del.plot(curves["fractions"], curves["deletion_mean"], label=method)
        ax_ins.plot(curves["fractions"], curves["insertion_mean"], label=method)
    ax_del.set_title(f"{variant}: deletion (lower AUC = more faithful)")
    ax_ins.set_title(f"{variant}: insertion (higher AUC = more faithful)")
    for axis in (ax_del, ax_ins):
        axis.set_xlabel("fraction of pixels")
        axis.set_ylabel("target-class probability")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _localization_figure(plt: Any, variant: str, methods: dict[str, Any], path: Path) -> None:
    names, means, errors = [], [], []
    for method, payload in sorted(methods.items()):
        aggregate = payload["all"]["energy_in_mask"]
        if aggregate["mean"] is None:
            continue
        names.append(method)
        means.append(aggregate["mean"])
        errors.append(
            [
                max(0.0, aggregate["mean"] - aggregate["ci_low"]),
                max(0.0, aggregate["ci_high"] - aggregate["mean"]),
            ]
        )
    if not names:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, means, yerr=list(zip(*errors, strict=True)), capsize=4)
    axis.set_ylabel("energy inside pet mask")
    axis.set_title(f"{variant}: localization (all predictions, 95% bootstrap CI)")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _spurious_figure(plt: Any, variant: str, methods: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    drawn = False
    for method_index, (method, by_test_variant) in enumerate(sorted(methods.items())):
        variants = sorted(by_test_variant)
        patch_means = [
            by_test_variant[tv]["patch_energy_patched_inputs"]["all"]["patch_energy"]["mean"] or 0.0
            for tv in variants
        ]
        pet_means = [
            by_test_variant[tv]["pet_mask_energy"]["all"]["pet_mask_energy"]["mean"] or 0.0
            for tv in variants
        ]
        positions = [i + method_index * 0.0 for i in range(len(variants))]
        axis.bar(
            [p - bar_width / 2 for p in positions],
            patch_means,
            bar_width,
            label=f"{method}: patch energy",
        )
        axis.bar(
            [p + bar_width / 2 for p in positions],
            pet_means,
            bar_width,
            label=f"{method}: pet-mask energy",
        )
        axis.set_xticks(list(range(len(variants))))
        axis.set_xticklabels(variants)
        drawn = True
        break  # one method per figure keeps the chart readable; others are in summary.md
    if not drawn:
        plt.close(figure)
        return
    axis.set_ylabel("attribution energy fraction")
    axis.set_title(f"{variant}: spurious-patch energy by test variant")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_figures(summary: dict[str, Any], figures_dir: Path, assets_dir: Path) -> list[Path]:
    plt = _plt()
    figures_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for variant, methods in summary.get("faithfulness", {}).items():
        path = figures_dir / f"faithfulness_{variant}.png"
        _faithfulness_figure(plt, variant, methods, path)
        produced.append(path)
    for variant, methods in summary.get("localization", {}).items():
        path = figures_dir / f"localization_{variant}.png"
        _localization_figure(plt, variant, methods, path)
        produced.append(path)
    for variant, methods in summary.get("spurious", {}).items():
        path = figures_dir / f"spurious_{variant}.png"
        _spurious_figure(plt, variant, methods, path)
        produced.append(path)

    produced = [path for path in produced if path.is_file()]
    assets_dir.mkdir(parents=True, exist_ok=True)
    for path in produced:
        shutil.copy2(path, assets_dir / path.name)
    logger.info("wrote %d figures to %s (mirrored to %s)", len(produced), figures_dir, assets_dir)
    return produced
