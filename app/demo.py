"""Gradio demo: a precomputed-results explorer (default) plus a live tab.

The explorer only reads files under results/ — it works without a GPU and
without model weights. The live tab needs a trained checkpoint.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.service import WARNING_TEXT, InferenceService, render_attribution_png
from vision_xai.config import AppConfig
from vision_xai.errors import VisionXAIError
from vision_xai.models.factory import ModelName
from vision_xai.paths import resolve_data_dir, resolve_results_dir

logger = logging.getLogger(__name__)


def _explanation_runs(cfg: AppConfig) -> dict[str, Path]:
    root = resolve_results_dir(cfg) / "raw" / "explanations" / cfg.experiment_name
    runs: dict[str, Path] = {}
    for run_json in sorted(root.glob("*/*/run.json")):
        directory = run_json.parent
        runs[f"{directory.parent.name} / {directory.name}"] = directory
    return runs


def _samples_for(run_dir: Path) -> list[str]:
    attr_dir = run_dir / "attr"
    return sorted(path.stem for path in attr_dir.glob("*.npz")) if attr_dir.is_dir() else []


def build_demo(cfg: AppConfig) -> Any:
    import gradio as gr

    service = InferenceService(cfg)
    runs = _explanation_runs(cfg)

    def show_precomputed(run_label: str, sample_id: str) -> tuple[Image.Image | None, str]:
        if not run_label or not sample_id:
            return None, "Pick a run and a sample."
        if run_label not in runs:
            return None, "Unknown run."
        run_dir = runs[run_label]
        # Gradio's Dropdown only restricts values in the browser UI — a client
        # calling the auto-generated API directly can pass any string. Validate
        # against the real sample list before building any filesystem path, so
        # a crafted sample_id (e.g. containing '..' or an absolute path) can
        # never escape this run's own attr/ directory (CWE-22).
        if sample_id not in _samples_for(run_dir):
            return None, "Unknown sample id for this run."
        attribution = np.load(run_dir / "attr" / f"{sample_id}.npz")["attribution"].astype(
            np.float32
        )
        meta_rows = {
            str(row["sample_id"]): row
            for row in (
                json.loads(line)
                for line in (run_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        info = meta_rows.get(sample_id, {})
        image_tensor = None
        # Best effort: overlay on the original image when the dataset is present.
        from vision_xai.data.manifest import MANIFEST_FILENAME, load_manifest
        from vision_xai.data.transforms import build_eval_transform
        from vision_xai.paths import manifests_dir

        manifest_path = manifests_dir(cfg) / MANIFEST_FILENAME
        if manifest_path.is_file():
            records = {r.sample_id: r for r in load_manifest(manifest_path)}
            record = records.get(sample_id)
            if record is not None:
                image_path = resolve_data_dir(cfg) / record.image_relpath
                if image_path.is_file():
                    with Image.open(image_path) as raw:
                        image_tensor = build_eval_transform(cfg.data)(raw.convert("RGB"))
        png = render_attribution_png(attribution, image_tensor)
        details = json.dumps(info, indent=2)
        return Image.open(io.BytesIO(png)), f"```json\n{details}\n```\n\n**{WARNING_TEXT}**"

    def update_samples(run_label: str) -> Any:
        choices = _samples_for(runs[run_label]) if run_label else []
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def live_explain(
        image: Image.Image | None, model: str, method: str
    ) -> tuple[Image.Image | None, str]:
        if image is None:
            return None, "Upload an image first."
        try:
            model_name: ModelName = "cnn" if model == "cnn" else "vit"
            png, info = service.explain_image(image, model_name, method)
        except VisionXAIError as exc:
            return None, f"Error: {exc}"
        return (
            Image.open(io.BytesIO(png)),
            f"```json\n{json.dumps(info, indent=2)}\n```\n\n**{WARNING_TEXT}**",
        )

    with gr.Blocks(title="vision-xai-reliability-lab") as demo:
        gr.Markdown(f"# vision-xai-reliability-lab\n**{WARNING_TEXT}**")
        with gr.Tab("Precomputed explorer"):
            run_dropdown = gr.Dropdown(choices=sorted(runs), label="explanation run")
            sample_dropdown = gr.Dropdown(choices=[], label="sample id")
            show_button = gr.Button("Show")
            heatmap_output = gr.Image(label="attribution (visualization-normalized)")
            info_output = gr.Markdown()
            run_dropdown.change(update_samples, run_dropdown, sample_dropdown)
            show_button.click(
                show_precomputed, [run_dropdown, sample_dropdown], [heatmap_output, info_output]
            )
        with gr.Tab("Live (needs a trained checkpoint)"):
            image_input = gr.Image(type="pil", label="upload an image")
            model_dropdown = gr.Dropdown(
                choices=["cnn", "vit"], value=cfg.serve.default_model, label="model"
            )
            method_dropdown = gr.Dropdown(
                choices=["gradcam", "integrated_gradients", "occlusion"],
                value="gradcam",
                label="method (gradcam is CNN-only)",
            )
            explain_button = gr.Button("Explain")
            live_heatmap = gr.Image(label="attribution (visualization-normalized)")
            live_info = gr.Markdown()
            explain_button.click(
                live_explain,
                [image_input, model_dropdown, method_dropdown],
                [live_heatmap, live_info],
            )
    return demo
