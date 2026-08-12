# Vision XAI Reliability Lab

[![CI](https://github.com/kuotunyu/vision-xai-reliability-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/vision-xai-reliability-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)

這是一套以可靠性為核心的 Vision XAI benchmark，針對 ConvNeXt-Tiny 與 ViT-B/16，分開檢驗 Heatmap 的 Localization、Faithfulness、Model Randomization、Flip Consistency 與 Spurious Patch 行為。

重點不在產生更漂亮的 Heatmap，而在回答：它是否真的依賴模型學到的證據，以及結論能否被重算、稽核與反駁。

[成果展示](https://kuotunyu.github.io/vision-xai-reliability-lab/) · [快速開始](#快速開始) · [實驗證據](ARTIFACTS.md) · [Model Card](MODEL_CARD.md) · [English](README_en.md)

![Vision XAI reliability evidence](assets/portfolio/hero.png)

> **Heatmap 不是因果推理的證據。** 本專案以統計指標檢驗解釋在哪裡成立、在哪裡失效，並以 SHA-256 manifest 防止 smoke test 或 CI 改寫正式結果。

不含 weights 的[靜態展示原始檔](showcase/)只讀取已提交的 aggregate artifacts；不載入模型、dataset、後端服務、analytics 或外部 JavaScript。

---

## 三項核心發現

1. **Center Prior 在 Pointing Game 勝出。** 固定 Center Prior 在兩種模型都達到 **0.922**，高於所有受測 Attribution 方法。這反映 dataset 的中心構圖偏差；Localization 只代表幾何重疊，不等於 causal Faithfulness。
2. **Integrated Gradients 未通過 Model Randomization。** Head Randomization 後仍保留約 **0.47–0.48** 的絕對 Spearman 相似度，顯著高於其他受測方法。
3. **Spurious Patch 是負結果。** 在 frozen-backbone、head-only 設定下，模型沒有學到預期的 corner-patch shortcut；這不代表 vision model 普遍能抵抗 spurious cue。

以上數字來自真實的 full-scale L4 run：四個 classifier heads 使用完整 training split。Attribution metrics 使用 test split 中固定的 500 samples，**不是完整 test split**。Aggregate、provenance 與限制見 [ARTIFACTS.md](ARTIFACTS.md) 與 [FAILURES.md](FAILURES.md)。

---

## 證據流程與本機模型

```mermaid
flowchart LR
    Data["Oxford-IIIT Pet<br/>deterministic manifest"] --> Train["ConvNeXt-Tiny · ViT-B/16<br/>head-only training"]
    Train --> Attr["Grad-CAM · Integrated Gradients<br/>Occlusion · baselines"]
    Attr --> Eval["Localization · Faithfulness<br/>Randomization · Flip · Spurious Patch"]
    Eval --> Evidence["Aggregate evidence<br/>95% bootstrap CI · SHA-256"]
    Evidence --> Showcase["GitHub Pages<br/>weight-free showcase"]
    Train -. user-supplied checkpoints .-> Local["FastAPI + Gradio<br/>local model workspace"]
    Evidence --> Gate{"Release verifier"}
    Local --> Gate
```

---

## Release 狀態

- **Full-scale evidence：已驗證。** 四個 classifier heads 在完整 training split 上以 NVIDIA L4 執行；正式 aggregates 受 SHA-256 manifest 保護。
- **CUDA resume canary：PASS。** RTX 4090 上的 tiny synthetic canary 比對 uninterrupted 與 epoch-boundary resume；final head、optimizer、GradScaler 與 stable metrics 完全一致。它不是 full training 證據，且訓練 loop 沒有 scheduler。
- **Release gates：已納入。** Ruff、strict mypy、CPU tests、package、showcase allowlist 與 release verifier 都不依賴 GPU、dataset、weights 或 secrets。
- **公開邊界：明確。** Repository 與 GitHub Pages 只含 source、aggregates、figures 與文件；dataset、weights、checkpoints 與 runtime outputs 均排除。本機 Gradio 需由使用者另行提供 checkpoints。

---

## 實驗結果（自動產生）

以下標記之間的內容由 `results/derived/summary.json` 自動產生，保持機器可讀與可重現性：

<details>
<summary><strong>展開完整 machine-generated 結果表格</strong></summary>

<!-- RESULTS:BEGIN -->
**Experiment `full`** — generated 2026-07-25T14:53:10.938993+00:00

_experiment 'full': trained on the full dataset; explanations computed on the first 500 samples of the test split. Attribution and reliability metrics therefore describe that fixed subset, not the entire split._

**Classification (clean validation split)**

| variant | val acc | val macro-F1 | val ECE | device | time | peak VRAM |
|---|---|---|---|---|---|---|
| cnn | 0.962 | 0.962 | 0.070 | NVIDIA L4 | 66s | 934.2MB |
| cnn_patched | 0.962 | 0.962 | 0.070 | NVIDIA L4 | 66s | 934.2MB |
| vit | 0.950 | 0.950 | 0.053 | NVIDIA L4 | 69s | 899.3MB |
| vit_patched | 0.952 | 0.953 | 0.050 | NVIDIA L4 | 68s | 899.3MB |

**Localization (all predictions; mean [95% bootstrap CI]) — not causal faithfulness**

| variant | method | energy in mask | pointing game | topk iou 0.1 |
|---|---|---|---|---|
| cnn | center | 0.622 [0.604, 0.639] | 0.922 [0.898, 0.946] | 0.200 [0.192, 0.208] |
| cnn | gradcam | 0.753 [0.740, 0.766] | 0.864 [0.832, 0.894] | 0.206 [0.198, 0.216] |
| cnn | integrated_gradients | 0.505 [0.486, 0.525] | 0.506 [0.464, 0.552] | 0.111 [0.105, 0.117] |
| cnn | occlusion | 0.558 [0.541, 0.574] | 0.678 [0.636, 0.718] | 0.142 [0.137, 0.147] |
| cnn | random | 0.469 [0.451, 0.487] | 0.208 [0.174, 0.244] | 0.086 [0.085, 0.087] |
| cnn | uniform | 0.469 [0.451, 0.487] | 0.024 [0.010, 0.038] | 0.025 [0.022, 0.028] |
| vit | center | 0.622 [0.604, 0.639] | 0.922 [0.898, 0.946] | 0.200 [0.192, 0.208] |
| vit | integrated_gradients | 0.537 [0.520, 0.554] | 0.578 [0.530, 0.620] | 0.124 [0.119, 0.130] |
| vit | occlusion | 0.544 [0.526, 0.561] | 0.626 [0.582, 0.672] | 0.139 [0.132, 0.145] |
| vit | random | 0.469 [0.451, 0.487] | 0.208 [0.174, 0.244] | 0.086 [0.085, 0.087] |
| vit | uniform | 0.469 [0.451, 0.487] | 0.024 [0.010, 0.038] | 0.025 [0.022, 0.028] |

**Faithfulness (deletion lower / insertion higher = better)**

| variant | method | deletion AUC | insertion AUC |
|---|---|---|---|
| cnn | center | 0.456 [0.437, 0.474] | 0.559 [0.542, 0.578] |
| cnn | gradcam | 0.329 [0.313, 0.344] | 0.646 [0.629, 0.664] |
| cnn | integrated_gradients | 0.249 [0.236, 0.261] | 0.309 [0.296, 0.323] |
| cnn | occlusion | 0.501 [0.483, 0.518] | 0.577 [0.559, 0.597] |
| cnn | random | 0.257 [0.244, 0.270] | 0.258 [0.244, 0.270] |
| cnn | uniform | 0.498 [0.479, 0.515] | 0.483 [0.469, 0.497] |
| vit | center | 0.348 [0.330, 0.364] | 0.548 [0.530, 0.565] |
| vit | integrated_gradients | 0.199 [0.187, 0.210] | 0.381 [0.366, 0.397] |
| vit | occlusion | 0.322 [0.306, 0.339] | 0.571 [0.554, 0.588] |
| vit | random | 0.312 [0.297, 0.326] | 0.307 [0.292, 0.321] |
| vit | uniform | 0.446 [0.430, 0.462] | 0.470 [0.455, 0.486] |

**Sanity & stability (|Spearman|): randomization LOW, flip consistency HIGH**

| variant | method | randomization sim. | flip consistency |
|---|---|---|---|
| cnn | gradcam | 0.216 [0.174, 0.259] | 0.858 [0.834, 0.880] |
| cnn | integrated_gradients | 0.481 [0.458, 0.504] | 0.556 [0.538, 0.577] |
| cnn | occlusion | 0.225 [0.189, 0.263] | 0.565 [0.530, 0.598] |
| vit | integrated_gradients | 0.471 [0.451, 0.492] | 0.688 [0.673, 0.702] |
| vit | occlusion | 0.324 [0.268, 0.377] | 0.815 [0.773, 0.855] |

**Spurious corner patch (patched-trained models on three test variants)**

| variant | method | test variant | accuracy | patch energy (patched inputs) | pet-mask energy |
|---|---|---|---|---|---|
| cnn_patched | gradcam | correlated | 0.878 [0.848, 0.908] | 0.002 [0.001, 0.002] | 0.752 [0.740, 0.765] |
| cnn_patched | gradcam | counter_correlated | 0.866 [0.834, 0.896] | 0.001 [0.000, 0.003] | 0.753 [0.740, 0.766] |
| cnn_patched | gradcam | no_patch | 0.868 [0.838, 0.896] | n/a | 0.753 [0.741, 0.766] |
| cnn_patched | integrated_gradients | correlated | 0.878 [0.848, 0.908] | 0.005 [0.004, 0.005] | 0.509 [0.490, 0.528] |
| cnn_patched | integrated_gradients | counter_correlated | 0.866 [0.834, 0.896] | 0.005 [0.004, 0.005] | 0.507 [0.488, 0.527] |
| cnn_patched | integrated_gradients | no_patch | 0.868 [0.838, 0.896] | n/a | 0.506 [0.487, 0.526] |
| vit_patched | integrated_gradients | correlated | 0.836 [0.804, 0.868] | 0.012 [0.011, 0.013] | 0.547 [0.531, 0.564] |
| vit_patched | integrated_gradients | counter_correlated | 0.836 [0.802, 0.868] | 0.013 [0.010, 0.015] | 0.538 [0.521, 0.554] |
| vit_patched | integrated_gradients | no_patch | 0.836 [0.804, 0.868] | n/a | 0.537 [0.520, 0.554] |

_A heatmap is not proof of causal reasoning._
<!-- RESULTS:END -->

</details>

---

## 快速開始

需要 Python ≥ 3.11 與 [uv](https://docs.astral.sh/uv/)。除特別標註外，PowerShell 與 bash 指令通用：

```bash
# 1. 建立虛擬環境並安裝鎖定套件 (CPU-only Torch)
uv sync --frozen --no-editable

# 2. 執行自我檢驗與單元測試 (只用合成 fixtures，不需下載 dataset)
uv run --no-sync python -m vision_xai.cli self-check
uv run --no-sync pytest -q
```

### 準備 Dataset（下載約 800 MB，來源為牛津 VGG 官方伺服器）

```bash
uv run --no-sync python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

長任務每 32 筆存一次 Checkpoint，可隨時安全中斷與續跑：

```bash
uv run --no-sync python -m vision_xai.cli data prepare --config configs/full.yaml --max-items 200
uv run --no-sync python -m vision_xai.cli data prepare --config configs/full.yaml --resume
```

---

## Docker 容器化部署 (CPU)

```bash
# 建置映像檔並啟動 API 服務
docker build -t vision-xai:dev .
docker run --rm -p 8000:8000 vision-xai:dev

# 或使用 Docker Compose 啟動（唯讀掛載 checkpoints/results）
docker compose up --build
```

---

## 專案結構與文件導覽

```text
configs/            smoke.yaml（小子集）與 full.yaml
src/vision_xai/     核心架構：config、資料管線、模型適配、CLI
  data/             source、splits、manifest、fingerprint、trimap、patches、prepare
tests/              pytest 單元測試（合成 fixtures，絕不依賴外部資料集）
app/                FastAPI 後端與 Gradio 互動介面 (Stage 6)
results/            不可變 full 聚合結果與安全的資料準備摘要
schemas/            機器可讀之產物契約定義 (JSON Schema)
tools/              發布前審計驗證與 CUDA Canary 驗證工具
```

- [DATA_CARD.md](DATA_CARD.md)：資料來源、切分、授權與預處理規範。
- [MODEL_CARD.md](MODEL_CARD.md)：模型架構、訓練超參數與評測範圍。
- [ARTIFACTS.md](ARTIFACTS.md)：不可變公開產物與 SHA-256 驗證清單。
- [FAILURES.md](FAILURES.md)：負結果分析與失敗歸因。
- [OWNER_ACTIONS.md](OWNER_ACTIONS.md)：維護者操作指引與驗證流程。

---

## 授權與聲明

本專案程式碼採 [MIT License](LICENSE) 授權。Oxford-IIIT Pet 資料集請遵循其原始學術研究授權條款（見 [DATA_CARD.md](DATA_CARD.md)）。
