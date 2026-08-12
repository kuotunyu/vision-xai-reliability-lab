# 正體中文 Portfolio 與 Gradio 證據工作台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 GitHub 公開入口與 Gradio 改為正體中文優先，並以不需 weights 的 evidence dashboard 取代公開候選中的空白 explorer。

**Architecture:** README 與 report/release tooling 維持雙語且共享同一 generated result block。新增獨立的 `app.evidence` presentation layer，從 immutable full summary 與 CUDA canary 建立 fail-closed view model；`app.demo` 只負責兩頁籤 Gradio component wiring 與既有 inference service。Static showcase 沿用現有結構，只翻譯可見文案並保留 JSON 驗證。

**Tech Stack:** Python 3.11、Gradio 6、FastAPI、Pydantic-free typed dataclasses、HTML/CSS、vanilla JavaScript、pytest、Ruff、strict mypy、Playwright Chromium、Hatch/uv。

## Global Constraints

- 公開介面以正體中文（`zh-TW`）為主，專有名詞保留原文。
- `README.md` 是正體中文主版；完整英文副版是 `README_en.md`。
- Gradio 只有「實驗證據」與「本機模型」兩個頂層頁籤。
- 不新增 explainer、模型、dataset、metric 或訓練功能。
- 不發布 dataset、weights、checkpoints 或 per-sample attribution arrays。
- 所有 headline values 必須來自 committed full summary，不得 hard-code fallback。
- 500 samples 只能稱為固定 attribution subset；localization 不得稱為 causal faithfulness。
- CUDA canary 必須明示 tiny scope，不能暗示 full L4 training equivalence。
- Gradio analytics 預設停用；CI 不需要 GPU、dataset、weights 或 secrets。
- 不建立 remote、push、PR、tag、Release 或部署。
- 使用本機 `main`，不開 subagent；每個 commit 只含一個可審核主題。

## File structure

- `README.md`: 正體中文 GitHub／package 主版。
- `README_en.md`: 完整英文副版。
- `OWNER_ACTIONS.md`: 正體中文 About description 與 GitHub owner handoff。
- `src/vision_xai/report/build.py`: full report 可發布 README 路徑。
- `tools/verify_release.py`: 雙語 result synchronization 與公開邊界。
- `tools/audit_distribution.py`: sdist 必備雙語 README。
- `app/evidence.py`: canonical JSON → immutable dashboard view model；不含 Gradio code。
- `app/demo.py`: 兩頁籤 Gradio layout、callbacks、local inference wiring。
- `app/demo.css`: 大字、緊湊、responsive 的 Evidence Cartography skin。
- `app/api.py`: analytics default 與 graceful Gradio mount。
- `showcase/index.html`, `showcase/app.js`: 正體中文 static showcase copy。
- `Dockerfile`: 將安全 committed aggregates、canary 與 figures 放入 CPU image。
- `tests/test_evidence.py`: view model schema/value/fail-closed tests。
- `tests/test_demo.py`: Gradio structure、copy、analytics、availability tests。
- `tests/test_release_verifier.py`, `tests/test_report.py`, `tests/test_distribution_audit.py`: README rename regression。
- `tests/test_showcase.py`: `zh-TW` static UI 與 Docker safe-evidence regression。

---

### Task 1: 正體中文 README 主版與 GitHub About

**Files:**
- Rename: `README.md` → `README_en.md`
- Rename: `README_zh-TW.md` → `README.md`
- Modify: `OWNER_ACTIONS.md`
- Modify: `src/vision_xai/report/build.py:364-373`
- Modify: `tools/verify_release.py:205-211`
- Modify: `tools/audit_distribution.py:181-204`
- Modify: `tests/test_release_verifier.py`
- Modify: `tests/test_report.py`
- Modify: `tests/test_distribution_audit.py`

**Interfaces:**
- Consumes: `RESULTS_BEGIN`, `RESULTS_END`, canonical `results/derived/summary.md`。
- Produces: public README pair `("README.md", "README_en.md")`，供 report builder、verifier 與 sdist audit 共用。

- [ ] **Step 1: 先將 tests 改成新語言契約**

在 release tests 加入：

```python
def test_readme_language_contract() -> None:
    zh = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    en = (REPO_ROOT / "README_en.md").read_text(encoding="utf-8")
    assert "## 這次實驗真正發現了什麼" in zh
    assert "[English version](README_en.md)" in zh
    assert "## What this experiment found" in en
    assert "[正體中文](README.md)" in en
    assert not (REPO_ROOT / "README_zh-TW.md").exists()
```

將所有 `("README.md", "README_zh-TW.md")` expectation 改為
`("README.md", "README_en.md")`，並要求 sdist 同時包含兩者。

- [ ] **Step 2: 執行 focused tests，確認舊結構會失敗**

Run:

```powershell
uv run --no-sync pytest tests/test_release_verifier.py tests/test_report.py tests/test_distribution_audit.py -q
```

Expected: FAIL，原因包含 `README_en.md` 不存在或舊 `README_zh-TW.md` 仍存在。

- [ ] **Step 3: 執行純 rename 並更新雙向 links**

Run:

```powershell
git mv README.md README_en.md
git mv README_zh-TW.md README.md
```

以 patch 將中文主版入口設為 `[English version](README_en.md)`，英文副版設為
`[正體中文](README.md)`；保留兩份 generated result markers 與區塊內容逐 byte
一致。

- [ ] **Step 4: 更新 report、verifier、distribution 與 About handoff**

將 `build_report()` 預設 publish paths 與 `verify_readme_synchronization()` 改為：

```python
PUBLIC_READMES = (Path("README.md"), Path("README_en.md"))
```

`OWNER_ACTIONS.md` 的 About description 改為核准文字，Topics 保留英文；sdist
required public files 加入 `README_en.md`。

- [ ] **Step 5: 重跑 focused gates**

Run:

```powershell
uv run --no-sync pytest tests/test_release_verifier.py tests/test_report.py tests/test_distribution_audit.py -q
uv run --no-sync ruff format --check src tests tools
uv run --no-sync ruff check src tests tools
uv run --no-sync mypy src tests tools
uv run --no-sync python tools/verify_release.py --root . --git --allow-dirty
```

Expected: PASS；verifier 顯示 README synchronization 與 Markdown local links PASS。

- [ ] **Step 6: Commit**

```powershell
git add README.md README_en.md OWNER_ACTIONS.md src/vision_xai/report/build.py tools/verify_release.py tools/audit_distribution.py tests/test_release_verifier.py tests/test_report.py tests/test_distribution_audit.py
git commit -m "docs: make zh-TW the primary README"
```

### Task 2: Canonical evidence presentation model

**Files:**
- Create: `app/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `results/derived/summary.json`, `release/cuda-resume-canary.json`, `assets/figures/*.png`。
- Produces: `load_evidence_dashboard(root: Path) -> EvidenceDashboard`、`EvidenceDashboard.model(key: ModelName) -> ModelEvidence`。

- [ ] **Step 1: 寫 exact-value 與 fail-closed tests**

建立 `tests/test_evidence.py`，至少包含：

```python
import json
import shutil
from pathlib import Path

import pytest

from app.evidence import EvidenceError, load_evidence_dashboard


def test_full_evidence_extracts_canonical_headlines() -> None:
    dashboard = load_evidence_dashboard(REPO_ROOT)
    assert dashboard.center_pointing == 0.922
    assert dashboard.attribution_samples == 500
    assert dashboard.spurious_patch_energy_max == pytest.approx(0.012507)
    assert dashboard.model("cnn").best_pointing == 0.864
    assert dashboard.model("vit").best_pointing == 0.626
    assert dashboard.model("cnn").ig_randomization == pytest.approx(0.480816)


def test_non_full_summary_fails_closed(tmp_path: Path) -> None:
    summary_path = tmp_path / "results" / "derived" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary = json.loads(
        (REPO_ROOT / "results" / "derived" / "summary.json").read_text(encoding="utf-8")
    )
    summary["experiment"] = "smoke"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    canary_path = tmp_path / "release" / "cuda-resume-canary.json"
    canary_path.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "release" / "cuda-resume-canary.json", canary_path)
    shutil.copytree(REPO_ROOT / "assets" / "figures", tmp_path / "assets" / "figures")
    with pytest.raises(EvidenceError, match="canonical full"):
        load_evidence_dashboard(tmp_path)
```

Fixture 直接複製 committed JSON 與 figures，再只修改被測欄位；不可從 production
loader 產生 expected values。

- [ ] **Step 2: 執行 tests，確認 module 尚不存在**

```powershell
uv run --no-sync pytest tests/test_evidence.py -q
```

Expected: collection FAIL with `ModuleNotFoundError: app.evidence`。

- [ ] **Step 3: 實作 immutable dataclasses 與 schema guards**

在 `app/evidence.py` 定義：

```python
@dataclass(frozen=True)
class ModelEvidence:
    key: ModelName
    label: str
    val_accuracy: float
    val_macro_f1: float
    best_method: str
    best_pointing: float
    ig_randomization: float
    localization_figure: Path
    faithfulness_figure: Path
    spurious_figure: Path


@dataclass(frozen=True)
class EvidenceDashboard:
    center_pointing: float
    attribution_samples: int
    spurious_patch_energy_max: float
    models: Mapping[ModelName, ModelEvidence]
    canary_exact_states: tuple[str, ...]
    canary_scheduler_status: str
    canary_not_full_scale: bool

    def model(self, key: ModelName) -> ModelEvidence:
        return self.models[key]
```

Loader 必須檢查 `schema_version == 1`、`experiment == "full"`、兩個 model family、
500-sample localization `n`、canary `status == "PASS"` 與 `not_full_scale is True`。
任何 key/type/value 缺失轉成 `EvidenceError`，訊息只包含 public relative path。

- [ ] **Step 4: 執行 unit、format、lint、type gates**

```powershell
uv run --no-sync pytest tests/test_evidence.py -q
uv run --no-sync ruff format --check app/evidence.py tests/test_evidence.py
uv run --no-sync ruff check app/evidence.py tests/test_evidence.py
uv run --no-sync mypy app/evidence.py tests/test_evidence.py
```

Expected: all PASS。

- [ ] **Step 5: Commit**

```powershell
git add app/evidence.py tests/test_evidence.py
git commit -m "feat: add canonical evidence view model"
```

### Task 3: 雙層正體中文 Gradio UI

**Files:**
- Create: `app/demo.css`
- Create: `tests/test_demo.py`
- Modify: `app/demo.py`
- Modify: `app/api.py`
- Modify: `app/README.md`

**Interfaces:**
- Consumes: `load_evidence_dashboard(Path.cwd())`、`InferenceService.available_models()`、`InferenceService.explain_image()`。
- Produces: `build_demo(cfg: AppConfig) -> Any` with exactly two top-level tabs；`model_evidence_outputs(dashboard, key) -> tuple[str, Path, Path, Path]`。

- [ ] **Step 1: 寫 Gradio structure 與 analytics tests**

測試不依賴 dataset 或 checkpoint：

```python
import json
import os
from pathlib import Path

import pytest

from app.api import configure_gradio_environment
from app.demo import build_demo
from conftest import ConfigFactory


def test_demo_has_two_zh_tw_tabs(
    config_factory: ConfigFactory, synthetic_data_dir: Path
) -> None:
    cfg = config_factory(synthetic_data_dir)
    demo = build_demo(cfg, evidence_root=REPO_ROOT)
    config = demo.get_config_file()
    labels = [component.get("label") for component in config["components"]]
    assert labels.count("實驗證據") == 1
    assert labels.count("本機模型") == 1
    assert "Precomputed explorer" not in json.dumps(config)


def test_gradio_analytics_default_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    configure_gradio_environment()
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
```

另測 `model_evidence_outputs()` 切換 CNN／ViT 時分別回傳 `.864`／`.626` 與對應
figure paths；missing evidence 產生正體中文 error panel，不 raise 到 FastAPI mount。

- [ ] **Step 2: 執行 tests，確認舊 UI 不符合契約**

```powershell
uv run --no-sync pytest tests/test_demo.py -q
```

Expected: FAIL，原因為新 interfaces 不存在或舊 UI 仍有 `Precomputed explorer`。

- [ ] **Step 3: 建立 Evidence Cartography CSS**

`app/demo.css` 定義 `.vx-shell`、`.vx-hero`、`.vx-findings`、`.vx-metric-grid`、
`.vx-canary`、`.vx-readiness` 與 responsive breakpoints。使用 CSS variables：

```css
:root {
  --vx-bg: #07131d;
  --vx-panel: #0c212c;
  --vx-text: #f3efe7;
  --vx-muted: #9bb0ba;
  --vx-cyan: #42d2e1;
  --vx-coral: #ff785f;
  --vx-lime: #a7e33f;
}
```

Desktop body font 不低於 `18px`，mobile 不低於 `16px`；移除 Gradio footer；
`.gradio-container` 最大寬度 `1280px`，縮小 top/bottom padding。

- [ ] **Step 4: 重寫 `build_demo()` 為兩個 tabs**

函式 signature 改為：

```python
def build_demo(cfg: AppConfig, *, evidence_root: Path = Path(".")) -> Any:
```

Evidence load 用 `try/except EvidenceError` 轉為 error HTML。成功時先 render hero、三張
finding cards、model-family selector、metrics HTML、三個 `gr.Image` 與 canary card。
Selector change callback 同步更新數值與 figures。

本機模型 tab 先呼叫 `service.available_models()`；空 list 時只 render readiness
說明與 disabled controls。有 model 時 selector 只列 available models，method choices
依 `service.methods()` 更新。沿用現有 `live_explain()` 並將 user-facing errors 改為
正體中文，不改 service/API error types。

- [ ] **Step 5: 在 Gradio import 前停用 analytics**

於 `app/api.py` 新增：

```python
def configure_gradio_environment() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
```

`_try_mount_demo()` 在 `import gradio` 前呼叫此函式。使用 `setdefault`，保留 owner
明確 override 的能力。

- [ ] **Step 6: 更新 app usage 說明並執行 focused gates**

`app/README.md` 以正體中文說明 `/demo/`、兩頁籤、analytics default 與 checkpoint
需求。

```powershell
uv run --no-sync pytest tests/test_demo.py tests/test_api.py tests/test_evidence.py -q
uv run --no-sync ruff format --check app tests/test_demo.py tests/test_api.py tests/test_evidence.py
uv run --no-sync ruff check app tests/test_demo.py tests/test_api.py tests/test_evidence.py
uv run --no-sync mypy app tests/test_demo.py tests/test_api.py tests/test_evidence.py
```

Expected: all PASS。

- [ ] **Step 7: Commit**

```powershell
git add app/demo.py app/demo.css app/api.py app/README.md tests/test_demo.py
git commit -m "feat: redesign Gradio as zh-TW evidence workbench"
```

### Task 4: 正體中文 static showcase

**Files:**
- Modify: `showcase/index.html`
- Modify: `showcase/app.js`
- Modify: `tests/test_showcase.py`

**Interfaces:**
- Consumes: existing `data/summary.json` and `data/cuda-resume-canary.json` schema。
- Produces: same 18-file static bundle with `html[lang="zh-TW"]` and正體中文 user-visible copy。

- [ ] **Step 1: 寫 language 與 claim regression tests**

```python
def test_showcase_is_zh_tw_and_preserves_claim_boundaries() -> None:
    html = (REPO_ROOT / "showcase" / "index.html").read_text(encoding="utf-8")
    assert '<html lang="zh-TW">' in html
    assert "Heatmap 經得起證據檢驗嗎？" in html
    assert "固定 500-sample subset" in html
    assert "不是 causal faithfulness" in html
    assert "不是 full L4 training" in html
    assert "Precomputed explorer" not in html
```

`app.js` test 要求 success/error status 使用正體中文，metric keys 與 numeric extraction
保持不變。

- [ ] **Step 2: 執行 tests，確認英文頁會失敗**

```powershell
uv run --no-sync pytest tests/test_showcase.py -q
```

Expected: FAIL at `lang="en"` 或缺少核准的正體中文文案。

- [ ] **Step 3: 翻譯 visible HTML/JS copy**

只修改 headings、captions、buttons、status、table questions 與 boundary prose；保留
IDs、`data-*` attributes、JSON paths、model keys、figure paths 與 outbound URLs。
專有名詞使用原文，`meta description` 改為正體中文。

- [ ] **Step 4: 建置並 audit static bundle**

```powershell
uv run --no-sync pytest tests/test_showcase.py -q
uv run --no-sync python tools/build_showcase.py --root . --output .artifacts/showcase-zh-tw
uv run --no-sync python tools/build_showcase.py --audit .artifacts/showcase-zh-tw
node --check showcase/app.js
```

Expected: 18 files built/audited；JavaScript syntax PASS。

- [ ] **Step 5: Commit**

```powershell
git add showcase/index.html showcase/app.js tests/test_showcase.py
git commit -m "feat: localize results showcase for zh-TW"
```

### Task 5: Docker safe evidence 與 UI runtime boundary

**Files:**
- Modify: `Dockerfile`
- Modify: `tests/test_showcase.py`
- Modify: `OWNER_ACTIONS.md`

**Interfaces:**
- Consumes: `app/evidence.py` relative paths under `/app`。
- Produces: CPU image containing only `summary.json`、CUDA canary、six aggregate figures；no dataset/weights/checkpoints。

- [ ] **Step 1: 寫 Docker content boundary test**

```python
def test_docker_copies_only_safe_dashboard_evidence() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "results/derived/summary.json" in dockerfile
    assert "release/cuda-resume-canary.json" in dockerfile
    assert "assets/figures/" in dockerfile
    assert "checkpoints/" not in dockerfile
    assert "COPY data/" not in dockerfile
```

- [ ] **Step 2: 執行 test，確認目前 image 缺少 evidence**

```powershell
uv run --no-sync pytest tests/test_showcase.py::test_docker_copies_only_safe_dashboard_evidence -q
```

Expected: FAIL because current runtime stage only copies configs and app。

- [ ] **Step 3: 加入 explicit safe COPY**

在 runtime stage 使用三個明確 COPY，不 copy directories with runtime data：

```dockerfile
COPY results/derived/summary.json results/derived/summary.json
COPY release/cuda-resume-canary.json release/cuda-resume-canary.json
COPY assets/figures/ assets/figures/
```

更新 `OWNER_ACTIONS.md`，說明 About 是正體中文、Pages/Gradio 是 results explorer，
且 Docker image 仍不含 dataset、weights 或 checkpoints。

- [ ] **Step 4: 執行 focused tests 與 conditional Docker gate**

```powershell
uv run --no-sync pytest tests/test_showcase.py tests/test_api.py tests/test_demo.py -q
docker version --format "client={{.Client.Version}} server={{.Server.Version}}"
```

若 daemon 可用，接著執行：

```powershell
docker build -t vision-xai:zh-tw-rc .
docker run --rm vision-xai:zh-tw-rc python -m vision_xai.cli self-check --quiet
```

若 daemon 不可用，記錄 exact error；CI 的 Docker build/API/Gradio smoke 保持必要 gate。

- [ ] **Step 5: Commit**

```powershell
git add Dockerfile OWNER_ACTIONS.md tests/test_showcase.py
git commit -m "build: include safe dashboard evidence"
```

### Task 6: Browser UX verification 與 final release gates

**Files:**
- Modify only if a verified UI regression requires a focused fix and regression test。

**Interfaces:**
- Consumes: committed repository after Tasks 1–5。
- Produces: fresh desktop/mobile screenshots、clean-export test evidence、final clean Git state。

- [ ] **Step 1: 啟動 Gradio with production command**

```powershell
uv run --no-sync python -m vision_xai.cli serve --config configs/smoke.yaml --host 127.0.0.1 --port 8000
```

用 condition-based server helper 等待 `/health`，不要固定 sleep。

- [ ] **Step 2: 執行 Playwright desktop/mobile checks**

在 gitignored `.artifacts/` script 驗證：

- `/demo/` 200；console/page errors 為空；
- 正體中文 hero 與兩個 tabs 可見；
- 只有兩個頂層 tabs；
- ConvNeXt／ViT 切換更新 `.864`／`.626` 與 figures；
- 1440×900 與 390×844 無 horizontal overflow；
- keyboard focus 可辨識；正文 computed font size desktop ≥ 17 px、mobile ≥ 16 px；
- missing checkpoint 顯示正體中文 readiness，不顯示 absolute path；
- screenshot 存到 `.artifacts/gradio-screenshots/` 並人工查看。

- [ ] **Step 3: 執行完整 local gates**

```powershell
uv sync --frozen --no-editable
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy src tests app tools
uv run --no-sync pytest -m "not real_data" --cov=vision_xai --cov=app --cov-report=term
uv run --no-sync python tools/build_showcase.py --root . --output .artifacts/showcase-final
uv run --no-sync python tools/build_showcase.py --audit .artifacts/showcase-final
uv build --out-dir .artifacts/dist-final
```

接著對 wheel/sdist 執行 `tools/audit_distribution.py`，在新 venv 安裝 wheel，執行
isolated import 與 `vision_xai.cli self-check --quiet`。

- [ ] **Step 4: 在 workspace 外建立 clean committed export**

Clone final HEAD 到 `%TEMP%\vision-xai-zh-tw-<short-head>`，移除 clone 自動建立的
`origin`，確認 branch `main`、remote 空、tag 空、working tree clean。於該 clone
重跑 Ruff、mypy、full CPU tests、showcase build/audit、package build/audit、wheel
smoke、API/Gradio browser smoke 與：

```powershell
uv run --no-sync python tools/verify_release.py --root . --git
```

- [ ] **Step 5: 最終 Git/privacy audit**

確認 100% commits 的 author/committer 都是
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`；不存在 contributor
trailers、remote、tag、secret、private path、tracked runtime data 或 >1 MiB file。
再確認來源封存 `1_Vision_xAI` status 與 HEAD 未變。

- [ ] **Step 6: Final handoff**

報告 final HEAD、commits、Gradio URL/啟動指令、README/About language、browser
screenshots、tests、package/privacy audit、Docker daemon 狀態、limitations 與 owner
明天的 GitHub 操作。保留本機 `main`，不 push。
