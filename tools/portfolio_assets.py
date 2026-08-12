"""Generate the social preview from the canonical full-scale aggregate."""

from __future__ import annotations

import argparse
import contextlib
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
MUTED = "#A9BBC4"
CYAN = "#42D9E8"
LIME = "#C7F464"
CORAL = "#FF8066"
GRID = "#193340"
CJK_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf"),
)


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


def _font(
    size: int, *, bold: bool = False, mono: bool = False, cjk: bool = False
) -> ImageFont.FreeTypeFont:
    if cjk:
        for path in CJK_FONT_CANDIDATES:
            if path.is_file():
                font = ImageFont.truetype(path, size=size)
                if bold:
                    with contextlib.suppress(OSError):
                        font.set_variation_by_name("Bold")
                return font
        family = "Noto Sans CJK TC"
        properties = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
        resolved = font_manager.findfont(properties, fallback_to_default=True)
        return ImageFont.truetype(resolved, size=size)
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    properties = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(properties), size=size)


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int, *, spacing: int) -> None:
    for x in range(0, width, spacing):
        draw.line((x, 0, x, height), fill=GRID, width=1)
    for y in range(0, height, spacing):
        draw.line((0, y, width, y), fill=GRID, width=1)


def _render_social(metrics: PortfolioMetrics) -> Image.Image:
    image = Image.new("RGB", (1280, 640), INK)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, 1280, 640, spacing=64)
    draw.rectangle((0, 0, 14, 640), fill=CYAN)
    draw.text((60, 42), "VISION XAI / RELIABILITY LAB", font=_font(22, bold=True), fill=CYAN)
    draw.text(
        (944, 48),
        "FULL-SCALE EVIDENCE / 01",
        font=_font(15, mono=True),
        fill=MUTED,
    )
    draw.text(
        (56, 92),
        "視覺解釋的可靠性驗證",
        font=_font(62, bold=True, cjk=True),
        fill=IVORY,
    )
    draw.text(
        (60, 176),
        "ConvNeXt-Tiny \N{MULTIPLICATION SIGN} ViT-B/16",
        font=_font(27, bold=True),
        fill=IVORY,
    )
    draw.text(
        (458, 183),
        "Localization · Faithfulness · Model Randomization · Spurious Patch",
        font=_font(17),
        fill=MUTED,
    )
    draw.line((60, 238, 1220, 238), fill=CYAN, width=2)

    spurious_spread = max(
        metrics.cnn_spurious_accuracy_spread, metrics.vit_spurious_accuracy_spread
    )
    cards = (
        (
            58,
            CYAN,
            "LOCALIZATION",
            f"{metrics.center_pointing:.3f}",
            "Center Prior 勝出",
            "Best Attribution",
            (
                f"CNN {metrics.cnn_best_attribution_pointing:.3f}"
                f" · ViT {metrics.vit_best_attribution_pointing:.3f}"
            ),
        ),
        (
            458,
            CORAL,
            "SANITY CHECK",
            f"{metrics.cnn_ig_randomization:.3f}",
            "IG 未通過 Model Randomization",
            "Randomization 後仍相似",
            f"CNN {metrics.cnn_ig_randomization:.3f} · ViT {metrics.vit_ig_randomization:.3f}",
        ),
        (
            858,
            LIME,
            "NEGATIVE RESULT",
            f"≤{spurious_spread:.3f}",
            "Spurious Patch 負結果",
            "Accuracy spread",
            (
                f"CNN {metrics.cnn_spurious_accuracy_spread:.3f}"
                f" · ViT {metrics.vit_spurious_accuracy_spread:.3f}"
            ),
        ),
    )
    for index, (left, accent, category, value, title, detail, detail_value) in enumerate(
        cards, start=1
    ):
        draw.rounded_rectangle(
            (left, 278, left + 364, 530), radius=16, fill=PANEL, outline=GRID, width=2
        )
        draw.text(
            (left + 24, 300),
            f"0{index} / {category}",
            font=_font(14, mono=True),
            fill=accent,
        )
        draw.text((left + 22, 334), value, font=_font(54, bold=True), fill=accent)
        draw.text(
            (left + 24, 407),
            title,
            font=_font(21, bold=True, cjk=True),
            fill=IVORY,
        )
        draw.text(
            (left + 24, 453),
            detail,
            font=_font(16, cjk=True),
            fill=MUTED,
        )
        draw.text(
            (left + 24, 477),
            detail_value,
            font=_font(15),
            fill=MUTED,
        )
        draw.line((left + 24, 510, left + 340, 510), fill=accent, width=2)
    draw.text(
        (60, 576),
        f"完整 L4 run  /  固定 {metrics.attribution_subset}-sample Attribution subset"
        "  /  SHA-256 可驗證 artifacts",
        font=_font(16, cjk=True),
        fill=MUTED,
    )
    return image


def render_social_preview(root: Path, output_dir: Path) -> Path:
    """Render the social-preview PNG without touching protected evidence trees."""
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
    social = output_dir / "social-preview.png"
    _render_social(metrics).save(social, format="PNG", optimize=True, compress_level=9)
    return social


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        social = render_social_preview(args.root, args.output)
    except PortfolioAssetError as exc:
        parser.exit(1, f"FAIL {exc}\n")
    sys.stdout.write(f"WROTE {social}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
