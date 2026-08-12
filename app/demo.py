"""Gradio evidence workbench with an optional local-model layer."""

from __future__ import annotations

import html
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from PIL import Image

from app.evidence import EvidenceDashboard, EvidenceError, ModelName, load_evidence_dashboard
from app.service import InferenceService
from vision_xai.config import AppConfig
from vision_xai.errors import VisionXAIError
from vision_xai.models.factory import ModelName as InferenceModelName

logger = logging.getLogger(__name__)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

METHOD_LABELS = {
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated Gradients",
    "occlusion": "Occlusion",
}
DEMO_CSS_PATH = Path(__file__).with_name("demo.css")


def _percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def _hero_html() -> str:
    return """
    <header class="vx-hero">
      <div class="vx-hero-copy">
        <h1>證據工作台</h1>
        <p>用已提交的完整規模實驗證據檢查 XAI，而不是只挑看似合理的 heatmap。</p>
      </div>
      <div class="vx-hero-meta" aria-label="證據範圍">
        <span>L4 完整結果</span>
        <span>固定 500 筆 attribution subset</span>
        <span>版本化證據</span>
      </div>
    </header>
    """


def _findings_html(dashboard: EvidenceDashboard) -> str:
    return f"""
    <section class="vx-findings" aria-label="三個核心結論">
      <article>
        <strong>{dashboard.center_pointing:.3f}</strong>
        <h2>Center prior 勝過實際 attribution</h2>
        <p>Pointing game 的最佳 baseline；這是 localization 證據，不是 causal faithfulness。</p>
      </article>
      <article>
        <strong>0.47–0.48</strong>
        <h2>IG 未通過 sanity check</h2>
        <p>Model-randomization 後仍保留偏高的 absolute Spearman similarity；健康結果應該更低。</p>
      </article>
      <article>
        <strong>{_percent(dashboard.spurious_patch_energy_max, 2)}</strong>
        <h2>Spurious-patch 實驗為負結果</h2>
        <p>最大 mean patch energy 仍低；不能解讀為 vision model 普遍抵抗 spurious cue。</p>
      </article>
    </section>
    """


def model_evidence_outputs(
    dashboard: EvidenceDashboard,
    key: ModelName,
) -> tuple[str, Path, Path, Path]:
    model = dashboard.model(key)
    summary = f"""
    <section class="vx-model-summary" aria-live="polite">
      <div><span>模型系列</span><strong>{html.escape(model.label)}</strong></div>
      <div><span>驗證集 accuracy</span><strong>{_percent(model.val_accuracy)}</strong></div>
      <div><span>驗證集 macro-F1</span><strong>{_percent(model.val_macro_f1)}</strong></div>
      <div>
        <span>最佳實際 attribution</span>
        <strong>{html.escape(model.best_method)} · {_percent(model.best_pointing)}</strong>
      </div>
      <div><span>IG randomization |ρ|</span><strong>{model.ig_randomization:.3f}</strong></div>
    </section>
    """
    return (
        summary,
        model.localization_figure,
        model.faithfulness_figure,
        model.spurious_figure,
    )


def _canary_html(dashboard: EvidenceDashboard) -> str:
    exact_states = " · ".join(html.escape(name) for name in dashboard.canary_exact_states)
    hash_status = "一致" if dashboard.canary_checkpoint_hash_equal else "不同"
    return f"""
    <section class="vx-canary">
      <div class="vx-canary-heading">
        <h2>CUDA resume canary</h2>
        <span class="vx-status-pass">PASS · scope 已界定</span>
      </div>
      <div class="vx-canary-grid">
        <div><span>硬體</span><strong>{html.escape(dashboard.canary_gpu)}</strong></div>
        <div>
          <span>軟體</span>
          <strong>PyTorch {html.escape(dashboard.canary_torch_version)}</strong>
        </div>
        <div><span>完全一致的 state</span><strong>{exact_states}</strong></div>
        <div><span>Scheduler</span><strong>{html.escape(dashboard.canary_scheduler_status)}</strong></div>
        <div>
          <span>完整 checkpoint SHA-256</span>
          <strong>{hash_status} · 僅供 diagnostic</strong>
        </div>
      </div>
      <p>
        這項證據使用 tiny synthetic data，在 epoch boundary 中斷後 resume；
        不是完整 L4 training resume 的證據。
      </p>
    </section>
    """


def _evidence_error_html(message: str) -> str:
    return f"""
    <section class="vx-evidence-error" role="alert">
      <h2>公開證據無法載入</h2>
      <p>{html.escape(message)}</p>
      <p>
        請執行 release verifier，確認 committed summary、CUDA canary 與
        aggregate figures 完整。
      </p>
    </section>
    """


def _readiness_html(available: list[InferenceModelName]) -> str:
    if not available:
        return """
        <section class="vx-readiness vx-readiness-empty">
          <h2>尚未偵測到本機 checkpoint</h2>
          <p>將相容 checkpoint 放入 config 指定位置後重新啟動即可；上方實驗證據不受影響。</p>
        </section>
        """
    labels = ["ConvNeXt-Tiny" if key == "cnn" else "ViT-B/16" for key in available]
    return f"""
    <section class="vx-readiness vx-readiness-ready">
      <h2>本機 inference 已就緒</h2>
      <p>可用模型：{html.escape("、".join(labels))}。模型只在第一次操作時 lazy load。</p>
    </section>
    """


def _methods_for_model(compatibility: dict[str, list[str]], model: InferenceModelName) -> list[str]:
    return [method for method, models in compatibility.items() if model in models]


def build_demo(cfg: AppConfig, *, evidence_root: Path | None = None) -> Any:
    """Build the two-layer zh-TW workbench without requiring weights or data."""
    import gradio as gr

    root = Path.cwd() if evidence_root is None else evidence_root
    service = InferenceService(cfg)
    compatibility = service.methods()
    available = cast(
        list[InferenceModelName],
        [model for model in service.available_models() if model in ("cnn", "vit")],
    )
    try:
        dashboard: EvidenceDashboard | None = load_evidence_dashboard(root)
        evidence_error: str | None = None
    except EvidenceError as exc:
        dashboard = None
        evidence_error = str(exc)

    with gr.Blocks(
        title="Vision XAI 證據工作台",
        fill_width=True,
        elem_classes=["vx-shell"],
    ) as demo:
        gr.HTML(_hero_html())
        with gr.Tabs(elem_classes=["vx-tabs"]):
            with gr.Tab("實驗證據"):
                if dashboard is None:
                    gr.HTML(_evidence_error_html(evidence_error or "unknown evidence error"))
                else:
                    gr.HTML(_findings_html(dashboard))
                    with gr.Group(elem_classes=["vx-evidence-panel"]):
                        model_selector = gr.Radio(
                            choices=[("ConvNeXt-Tiny", "cnn"), ("ViT-B/16", "vit")],
                            value="cnn",
                            label="模型系列",
                            elem_classes=["vx-model-toggle"],
                        )
                        initial = model_evidence_outputs(dashboard, "cnn")
                        model_summary = gr.HTML(initial[0])
                        with gr.Row(elem_classes=["vx-figure-grid"], equal_height=False):
                            localization = gr.Image(
                                value=initial[1],
                                label="Localization（不是 causal faithfulness）",
                                interactive=False,
                                buttons=[],
                            )
                            faithfulness = gr.Image(
                                value=initial[2],
                                label="Faithfulness（deletion / insertion）",
                                interactive=False,
                                buttons=[],
                            )
                            spurious = gr.Image(
                                value=initial[3],
                                label="Spurious patch（負結果）",
                                interactive=False,
                                buttons=[],
                            )

                        def select_model(key: str) -> tuple[str, Path, Path, Path]:
                            model_key: ModelName = "vit" if key == "vit" else "cnn"
                            return model_evidence_outputs(dashboard, model_key)

                        model_selector.change(
                            select_model,
                            inputs=model_selector,
                            outputs=[model_summary, localization, faithfulness, spurious],
                        )
                    gr.HTML(_canary_html(dashboard))
                    gr.HTML(
                        f'<p class="vx-scope-note">所有 attribution metrics 使用固定 '
                        f"<strong>{dashboard.attribution_samples} 筆 attribution subset</strong>，"
                        "不是完整 test split；localization 不是 causal faithfulness。</p>"
                    )

            with gr.Tab("本機模型"):
                gr.HTML(_readiness_html(available))
                if available:
                    initial_model = (
                        cfg.serve.default_model
                        if cfg.serve.default_model in available
                        else available[0]
                    )
                    initial_methods = _methods_for_model(compatibility, initial_model)
                    with gr.Row(elem_classes=["vx-local-grid"], equal_height=False):
                        with gr.Column(scale=1, min_width=300):
                            image_input = gr.Image(
                                type="pil",
                                label="上傳影像",
                                sources=["upload"],
                            )
                            model_dropdown = gr.Dropdown(
                                choices=[
                                    (
                                        "ConvNeXt-Tiny" if key == "cnn" else "ViT-B/16",
                                        key,
                                    )
                                    for key in available
                                ],
                                value=initial_model,
                                label="模型系列",
                            )
                            method_dropdown = gr.Dropdown(
                                choices=[
                                    (METHOD_LABELS[method], method) for method in initial_methods
                                ],
                                value=initial_methods[0],
                                label="Attribution 方法",
                            )
                            explain_button = gr.Button(
                                "產生本機 attribution",
                                variant="primary",
                                elem_classes=["vx-action"],
                            )
                        with gr.Column(scale=2, min_width=360):
                            live_heatmap = gr.Image(
                                label="Attribution（已正規化顯示）",
                                interactive=False,
                            )
                            live_info = gr.HTML(
                                '<p class="vx-live-placeholder">結果會顯示 prediction、'
                                "probability、方法 metadata 與限制。</p>"
                            )

                    def update_methods(model: str) -> Any:
                        model_name: InferenceModelName = "vit" if model == "vit" else "cnn"
                        methods = _methods_for_model(compatibility, model_name)
                        return gr.Dropdown(
                            choices=[(METHOD_LABELS[method], method) for method in methods],
                            value=methods[0],
                        )

                    def live_explain(
                        image: Image.Image | None,
                        model: str,
                        method: str,
                    ) -> tuple[Image.Image | None, str]:
                        if image is None:
                            return None, '<p class="vx-inline-error">請先上傳一張影像。</p>'
                        model_name: InferenceModelName = "vit" if model == "vit" else "cnn"
                        try:
                            png, info = service.explain_image(image, model_name, method)
                        except VisionXAIError:
                            logger.exception("local Gradio inference failed")
                            return (
                                None,
                                '<p class="vx-inline-error">本機模型執行失敗；'
                                "請檢查 server log 與 checkpoint/config 是否一致。</p>",
                            )
                        with Image.open(io.BytesIO(png)) as rendered:
                            heatmap = rendered.copy()
                        details = html.escape(json.dumps(info, indent=2, ensure_ascii=False))
                        return (
                            heatmap,
                            f'<pre class="vx-json">{details}</pre>'
                            '<p class="vx-warning">Heatmap 不是 causal reasoning 的證據。</p>',
                        )

                    model_dropdown.change(
                        update_methods,
                        inputs=model_dropdown,
                        outputs=method_dropdown,
                    )
                    explain_button.click(
                        live_explain,
                        inputs=[image_input, model_dropdown, method_dropdown],
                        outputs=[live_heatmap, live_info],
                    )
    return demo
