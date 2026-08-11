# vision-xai-reliability-lab（繁體中文）

比較 CNN（**ConvNeXt-Tiny**）與 Vision Transformer（**ViT-B/16**）在
**Oxford-IIIT Pet** 上的 attribution methods，並且不只產生漂亮的 heatmap，而是
量化檢驗 explanation 是否**可靠**：以 segmentation mask 驗證 localization、
deletion / insertion faithfulness、parameter randomization 檢查、augmentation
consistency，以及 synthetic spurious-cue 實驗。

> Heatmap 不等於因果推理的證據。本 repo 量測 attribution 方法在哪些情況成立、
> 在哪些情況失效。

[English README → README.md](README.md)

## 專案進度

| 階段 | 範圍 | 狀態 |
|---|---|---|
| 0 | Repo 骨架：uv 打包、lint / type / test gates、Docker CPU smoke、CI | ✅ 完成 |
| 1 | 資料管線：固定 split、manifest 與 fingerprint、trimap 對齊 mask、spurious patch 指派、`--resume` | ✅ 完成 |
| 2 | 訓練（classifier head、CUDA 上 AMP、每 epoch checkpoint、`--resume`） | ✅ 完成 |
| 3 | Explainers：Grad-CAM、Integrated Gradients、Occlusion ＋ random/uniform/center baselines（統一 `explain()` 介面） | ✅ 完成 |
| 4 | 可靠性評測：energy-in-mask、pointing game、top-k IoU、deletion / insertion AUC、randomization、flip consistency、patch energy | ✅ 完成 |
| 5 | 報告產生（`results/derived/summary.json` → README 的所有數字） | ✅ 完成 |
| 6 | 部署：FastAPI（`/health` `/predict` `/explain` `/methods`）＋ Gradio `/demo` | ✅ 完成 |

下方數字來自 **full 規模實跑**：四個變體都在 Google Colab NVIDIA L4 上以完整資料集訓練。
不可變的聚合證據與來源紀錄見 [ARTIFACTS.md](ARTIFACTS.md)。
Attribution 與可靠性指標是在 test split 的固定子集上計算（attribution 方法計算成本高），
確切規模直接寫在下方自動產生的區塊裡。另有 `smoke` config 可在 CPU 上幾分鐘跑完同一條
流程，用於驗證機制。

## 實驗結果（自動產生）

以下標記之間的內容全部由 `vision_xai report` 從
`results/derived/summary.json` 產生，不手填。

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

## 快速開始

需要 Python ≥ 3.11 與 [uv](https://docs.astral.sh/uv/)。除特別標註外，
PowerShell 與 bash 指令相同。

```sh
uv sync --frozen                                # CPU-only torch，由 uv.lock 決定版本
uv run python -m vision_xai.cli self-check      # CPU 煙霧測試
uv run pytest -q                                # 只用合成 fixtures，不需下載 dataset
```

準備 dataset（一次性下載約 800 MB，來源為牛津 VGG 官方伺服器）：

```sh
uv run python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

長任務每 32 筆存一次 checkpoint，可隨時中斷：

```sh
uv run python -m vision_xai.cli data prepare --config configs/full.yaml --max-items 200
uv run python -m vision_xai.cli data prepare --config configs/full.yaml --resume
```

不改 config 直接覆寫 dataset 位置：

```powershell
# PowerShell
$env:VISION_XAI_DATA_DIR = "<dataset-path>"
uv run python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

```sh
# bash
VISION_XAI_DATA_DIR=/mnt/d/datasets/pets uv run python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

## Docker（CPU）

```sh
docker build -t vision-xai:dev .
docker run --rm -p 8000:8000 vision-xai:dev     # 在 :8000 提供 API 服務
docker compose up --build                        # + 唯讀掛載 checkpoints/results
```

Image 依 `uv.lock` 從 PyTorch CPU index 安裝 torch，天生 CPU-only；不含
dataset、model weights 或 secrets。CI 不需要 GPU。

## GPU 工作

Commit 中的 full-scale 證據是在 **Google Colab NVIDIA L4** 產生。CUDA 並非一般開發需求：
本機開發、CI、API health path 與除了明確標示之 CUDA resume canary 以外的 release gates
皆可在 CPU 上執行。

## 目錄結構

```
configs/            smoke.yaml（小子集）與 full.yaml
src/vision_xai/     所有邏輯：config、資料管線、CLI
  data/             source、splits、manifest、fingerprint、trimap、patches、transforms、datasets、prepare
tests/              pytest（PIL 合成 fixtures，絕不使用真實 dataset）
app/                FastAPI 與 Gradio（Stage 6）
results/            不可變 full 聚合結果與安全的資料準備摘要
schemas/            machine-readable artifact contracts
tools/              本機 release 與 CUDA canary 驗證工具
```

重要文件：[DATA_CARD.md](DATA_CARD.md) · [MODEL_CARD.md](MODEL_CARD.md) ·
[ARTIFACTS.md](ARTIFACTS.md) · [FAILURES.md](FAILURES.md)

## 設計重點（Stage 1）

- **全面決定性。** train/val split 依 class 分層、seed 固定；per-sample 決策
  （spurious patch 指派）由 `sha256(seed, namespace, sample_id)` 導出，與迭代
  順序、子集選擇、平台無關。
- **Manifest 與 fingerprint。** 每個樣本的 image 與 trimap 都以 sha256 寫入
  JSONL manifest；dataset fingerprint 與處理順序無關，因此 resume 後的結果與
  一次跑完完全一致。
- **Trimap 語義驗證而非假設。** 官方語義（1 = pet、2 = background、3 =
  boundary）在 prepare 時以邊界像素 heuristic 實測驗證。
- **Spurious patch：指派落盤、像素即時套用。** 誰有 patch 在 prepare 時決定並
  存檔可稽核；像素在 resize/crop 之後套用，patch 邊界框在 model-input 座標中
  精確已知 — Stage 4 的 patch-attribution-energy 指標需要這一點。

## 開發

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## 授權

[MIT](LICENSE)。Oxford-IIIT Pet dataset 另有其授權 — 見
[DATA_CARD.md](DATA_CARD.md)。
