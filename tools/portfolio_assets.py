"""Generate portfolio visuals from the canonical full-scale aggregate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

BASELINES = {"center", "random", "uniform"}
INK = "#07131D"
PANEL = "#0C1D29"
IVORY = "#F4F0E6"
MUTED = "#8EA3AF"
CYAN = "#42D9E8"
LIME = "#C7F464"
CORAL = "#FF8066"
GRID = "#193340"


class PortfolioAssetError(RuntimeError):
    """The canonical evidence cannot safely drive a portfolio visual."""


@dataclass(frozen=True)
class PortfolioMetrics:
    """Headline values derived from the full aggregate, before display rounding."""

    center_pointing: float
    cnn_best_attribution_pointing: float
    vit_best_attribution_pointing: float
    cnn_ig_randomization: float
    vit_ig_randomization: float
    cnn_spurious_accuracy_spread: float
    vit_spurious_accuracy_spread: float
    attribution_subset: int


def _mean(payload: dict[str, Any]) -> float:
    value = payload.get("mean")
    if not isinstance(value, (int, float)):
        raise PortfolioAssetError("canonical aggregate is missing a numeric mean")
    return float(value)


def _best_attribution_pointing(summary: dict[str, Any], variant: str) -> float:
    methods = summary["localization"][variant]
    scores = [
        _mean(payload["all"]["pointing_rate"])
        for method, payload in methods.items()
        if method not in BASELINES
    ]
    if not scores:
        raise PortfolioAssetError(f"canonical aggregate has no attribution methods for {variant}")
    return max(scores)


def _spurious_accuracy_spread(summary: dict[str, Any], variant: str) -> float:
    methods = summary["spurious"][variant]
    accuracies = [
        _mean(test_payload["accuracy"])
        for method_payload in methods.values()
        for test_payload in method_payload.values()
    ]
    if not accuracies:
        raise PortfolioAssetError(f"canonical aggregate has no spurious results for {variant}")
    return max(accuracies) - min(accuracies)


def extract_portfolio_metrics(summary: dict[str, Any]) -> PortfolioMetrics:
    """Extract headline metrics from schema-v1 experiment ``full`` evidence."""
    if summary.get("schema_version") != 1 or summary.get("experiment") != "full":
        raise PortfolioAssetError("expected the canonical full summary")
    try:
        cnn_center = summary["localization"]["cnn"]["center"]["all"]["pointing_rate"]
        vit_center = summary["localization"]["vit"]["center"]["all"]["pointing_rate"]
        center_mean = _mean(cnn_center)
        if center_mean != _mean(vit_center) or cnn_center["n"] != vit_center["n"]:
            raise PortfolioAssetError("center-prior evidence differs between model families")
        return PortfolioMetrics(
            center_pointing=center_mean,
            cnn_best_attribution_pointing=_best_attribution_pointing(summary, "cnn"),
            vit_best_attribution_pointing=_best_attribution_pointing(summary, "vit"),
            cnn_ig_randomization=_mean(
                summary["randomization"]["cnn"]["integrated_gradients"]["all"]["abs_spearman"]
            ),
            vit_ig_randomization=_mean(
                summary["randomization"]["vit"]["integrated_gradients"]["all"]["abs_spearman"]
            ),
            cnn_spurious_accuracy_spread=_spurious_accuracy_spread(summary, "cnn_patched"),
            vit_spurious_accuracy_spread=_spurious_accuracy_spread(summary, "vit_patched"),
            attribution_subset=int(cnn_center["n"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PortfolioAssetError("canonical full summary has an unexpected shape") from exc


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    properties = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(properties), size=size)


def _metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    index: str,
    value: str,
    title: str,
    detail: tuple[str, ...],
    accent: str,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=22, fill=PANEL, outline=GRID, width=2)
    draw.text((left + 28, top + 24), index, font=_font(17, mono=True), fill=accent)
    draw.text((left + 28, top + 58), value, font=_font(62, bold=True), fill=IVORY)
    draw.text((left + 28, top + 134), title, font=_font(21, bold=True), fill=accent)
    y = top + 174
    for line in detail:
        draw.text((left + 28, y), line, font=_font(17), fill=MUTED)
        y += 28
    draw.line((left + 28, bottom - 26, right - 28, bottom - 26), fill=accent, width=3)


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int, *, spacing: int) -> None:
    for x in range(0, width, spacing):
        draw.line((x, 0, x, height), fill=GRID, width=1)
    for y in range(0, height, spacing):
        draw.line((0, y, width, y), fill=GRID, width=1)


def _render_hero(metrics: PortfolioMetrics) -> Image.Image:
    image = Image.new("RGB", (1600, 900), INK)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, 1600, 900, spacing=80)
    draw.rectangle((0, 0, 18, 900), fill=CYAN)
    draw.text((74, 58), "TRUSTWORTHY VISION / FIELD NOTE 01", font=_font(17, mono=True), fill=CYAN)
    draw.text((70, 102), "VISION XAI", font=_font(92, bold=True), fill=IVORY)
    draw.text((72, 202), "RELIABILITY LAB", font=_font(48), fill=MUTED)
    draw.text(
        (73, 270),
        "A heatmap can look right and still explain the wrong thing.",
        font=_font(26),
        fill=IVORY,
    )
    draw.line((74, 326, 1528, 326), fill=CYAN, width=3)

    spurious_spread = max(
        metrics.cnn_spurious_accuracy_spread, metrics.vit_spurious_accuracy_spread
    )
    _metric_card(
        draw,
        (72, 366, 558, 714),
        index="01 / LOCALIZATION TRAP",
        value=f"{metrics.center_pointing:.3f}",
        title="CENTER PRIOR WINS",
        detail=(
            f"CNN best method  {metrics.cnn_best_attribution_pointing:.3f}",
            f"ViT best method  {metrics.vit_best_attribution_pointing:.3f}",
            "Location is not causality.",
        ),
        accent=CYAN,
    )
    _metric_card(
        draw,
        (572, 366, 1058, 714),
        index="02 / SANITY CHECK",
        value=f"{metrics.cnn_ig_randomization:.3f}",
        title="IG RETAINS SIMILARITY",
        detail=(
            "after CNN head randomization",
            f"ViT retains       {metrics.vit_ig_randomization:.3f}",
            "Low was the healthy expectation.",
        ),
        accent=CORAL,
    )
    _metric_card(
        draw,
        (1072, 366, 1528, 714),
        index="03 / NEGATIVE RESULT",
        value=f"≤{spurious_spread:.3f}",
        title="ACCURACY SPREAD",
        detail=(
            "across patch test variants",
            "Shortcut was not learned.",
            "Reported without spin.",
        ),
        accent=LIME,
    )

    draw.text((74, 776), "FULL L4 TRAINING", font=_font(18, bold=True), fill=IVORY)
    draw.text(
        (74, 811),
        f"ATTRIBUTION METRICS · FIXED {metrics.attribution_subset}-SAMPLE TEST SUBSET",
        font=_font(17, mono=True),
        fill=MUTED,
    )
    draw.text((1236, 802), "LOCALIZATION ≠ CAUSALITY", font=_font(16, bold=True), fill=CYAN)
    return image


def _render_social(metrics: PortfolioMetrics) -> Image.Image:
    image = Image.new("RGB", (1280, 640), INK)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, 1280, 640, spacing=64)
    draw.rectangle((0, 0, 14, 640), fill=CYAN)
    draw.text((62, 48), "VISION XAI / RELIABILITY LAB", font=_font(25, bold=True), fill=CYAN)
    draw.text((58, 94), "DOES THE HEATMAP", font=_font(58, bold=True), fill=IVORY)
    draw.text((58, 158), "SURVIVE CONTACT", font=_font(58, bold=True), fill=IVORY)
    draw.text((58, 222), "WITH EVIDENCE?", font=_font(58, bold=True), fill=IVORY)

    spurious_spread = max(
        metrics.cnn_spurious_accuracy_spread, metrics.vit_spurious_accuracy_spread
    )
    cards = (
        (58, CYAN, f"{metrics.center_pointing:.3f}", "CENTER PRIOR", "beats learned attribution"),
        (
            470,
            CORAL,
            f"{metrics.cnn_ig_randomization:.3f}",
            "IG / RANDOMIZED",
            "similarity remains high",
        ),
        (
            882,
            LIME,
            f"≤{spurious_spread:.3f}",
            "PATCH ΔACC",
            "negative result",
        ),
    )
    for left, accent, value, title, detail in cards:
        draw.rounded_rectangle(
            (left, 350, left + 350, 536), radius=18, fill=PANEL, outline=GRID, width=2
        )
        draw.text((left + 22, 369), value, font=_font(45, bold=True), fill=accent)
        draw.text((left + 22, 429), title, font=_font(18, bold=True), fill=IVORY)
        draw.text((left + 22, 466), detail, font=_font(15), fill=MUTED)
    draw.text(
        (60, 580),
        f"FULL L4 RUN  /  FIXED {metrics.attribution_subset}-SAMPLE ATTRIBUTION SUBSET"
        "  /  AUDITABLE ARTIFACTS",
        font=_font(15, mono=True),
        fill=MUTED,
    )
    return image


def render_portfolio_assets(root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Render README and social-preview PNGs without touching evidence trees."""
    root = root.resolve()
    output_dir = output_dir.resolve()
    protected = tuple(
        (root / name).resolve() for name in ("results", "release", "data", "checkpoints")
    )
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise PortfolioAssetError("refusing to write portfolio art into a protected evidence tree")
    summary_path = root / "results" / "derived" / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortfolioAssetError(f"cannot read canonical summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise PortfolioAssetError("canonical summary must be a JSON object")
    metrics = extract_portfolio_metrics(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    hero = output_dir / "hero.png"
    social = output_dir / "social-preview.png"
    _render_hero(metrics).save(hero, format="PNG", optimize=True, compress_level=9)
    _render_social(metrics).save(social, format="PNG", optimize=True, compress_level=9)
    return hero, social


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        hero, social = render_portfolio_assets(args.root, args.output)
    except PortfolioAssetError as exc:
        parser.exit(1, f"FAIL {exc}\n")
    sys.stdout.write(f"WROTE {hero}\nWROTE {social}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
