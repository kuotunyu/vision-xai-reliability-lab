# 正體中文 Portfolio 與 Gradio 證據工作台設計

日期：2026-08-12  
狀態：已確認設計方向，待 implementation plan

## 目標

讓 `vision-xai-reliability-lab` 的 GitHub 首頁與互動介面以臺灣使用者為
主要讀者，同時保留技術精確度與英文搜尋入口。公開介面以正體中文（`zh-TW`）
為主；模型、方法、metric、framework 與工程術語保留原文，不做生硬翻譯。

本次同時解決現有 Gradio 的三個問題：公開候選沒有 per-sample arrays 時首頁
空白、字級偏小，以及表單層級與無效留白過多。

## 已核准的設計決策

1. GitHub 預設 `README.md` 改為正體中文主版。
2. 英文副版保留為 `README_en.md`，不再使用 `README_zh-TW.md`。
3. GitHub About description 使用正體中文；Topics 保持英文標準關鍵字以維持
   discoverability。
4. Gradio 採深色、低裝飾的「證據工作台」視覺，與既有 portfolio assets
   同一語彙，但不加入浮誇動畫或多餘 ornament。
5. Gradio 只保留兩個頂層頁籤：「實驗證據」與「本機模型」。
6. 字級提高、控制項與內容改用緊湊 grid；避免空白 output panel、空 dropdown
   與重複標題。

## 語言與內容架構

### README 與 About

- `README.md` 使用現有正體中文內容作為完整主版，首頁第一屏依序為定位、hero、
  badges、核心主張、行動連結與三個真實結果。
- `README_en.md` 保存完整英文說明，主版頂部提供「English version」入口；英文版
  頂部則返回正體中文主版。
- 兩個 README 的 generated results block 必須持續由同一份
  `results/derived/summary.json` 產生並由 verifier 比對，不能手動分叉數字。
- Package metadata 繼續使用 `README.md`，因此 PyPI／sdist 的預設長描述也會是
  正體中文主版。
- About description 使用：
  `以可靠性為核心的 XAI benchmark：比較 ConvNeXt 與 ViT 的 localization、faithfulness、sanity checks 與可重現 CUDA resume 證據。`
- Topics 維持 `computer-vision`、`explainable-ai`、`trustworthy-ai`、`pytorch`、
  `machine-learning`、`model-evaluation`、`reproducibility`、`fastapi`、`gradio`。

### 公開介面語氣

- 中文採臺灣常用表達，不使用簡體中文或中國用語。
- `Heatmap`、`localization`、`causal faithfulness`、`pointing game`、
  `Integrated Gradients`、`model randomization`、`checkpoint`、`resume`、
  `GradScaler` 等保留原文。
- 所有 claim 延續既有 evidence boundary：500 samples 明確稱為固定 attribution
  subset；localization 不稱為 causal faithfulness；CUDA canary 不稱為 full training。

## Gradio 資訊架構

### 頂層框架

頁首只保留產品名稱、`FULL L4` evidence 標示與版本。下方只有兩個頁籤：

1. `實驗證據`：預設開啟，完全不需要 dataset、weights、checkpoint 或 GPU。
2. `本機模型`：次要入口，使用既有 lazy-loading inference service；只有使用者
   自行提供 checkpoint 時才能執行。

現有 `Precomputed explorer` 不再作為公開頂層頁籤。它依賴刻意未發布的
per-sample attribution arrays，在乾淨 release candidate 中只會產生空 dropdown
與空 output，違反「減少無效層級」的要求。底層 inference API 不受影響。

### 實驗證據頁籤

由上至下只保留五個區塊：

1. **研究定位與 scope**：大標題「Heatmap 經得起證據檢驗嗎？」；同列標示
   Google Colab NVIDIA L4、完整資料集訓練與固定 500-sample attribution subset。
2. **三個核心結果**：`0.922 center prior`、約 `0.481 IG randomization`、
   `≤0.012 spurious patch energy`。每張 card 只有一個數值、一個結論與一句限制。
3. **Model family 比較**：ConvNeXt／ViT-B/16 單一 segmented control，切換後同步
   更新 accuracy、Macro-F1、best pointing、IG randomized similarity 與三張正式圖。
4. **Metric boundary**：用一個緊湊提示框說明 localization、faithfulness 與 sanity
   分別回答不同問題，避免錯誤合併。
5. **CUDA resume canary**：顯示 head、optimizer、GradScaler、stable metrics exact；
   scheduler 為 not applicable，並明示 tiny canary 不是 full L4 rerun。

所有數字直接讀取 committed `results/derived/summary.json` 與
`release/cuda-resume-canary.json`。圖表只使用 committed `assets/figures/`，不重新
計算或寫入任何公開 evidence。

### 本機模型頁籤

頁籤頂部先呈現 readiness status，列出目前偵測到的 model checkpoints。沒有
checkpoint 時顯示單一清楚說明，不產生 error stack、失效按鈕或巨大空白 image
panel。

有可用 checkpoint 時才顯示：

- image upload；
- model selector；
- 與 model 相容的 attribution method selector；
- `產生 explanation` primary action；
- attribution image 與必要 metadata。

Model 仍 lazy load；切換頁籤不觸發 model load。錯誤訊息使用正體中文包裝，且不
洩漏 local absolute path。API 的 `/health`、`/methods`、`/predict`、`/explain`
契約維持不變。

## 視覺規格

- 背景使用深 navy；內容使用 warm off-white；cyan 為主要 accent，coral 與 lime
  只標示 sanity failure 與 negative result。
- Desktop 主標約 40–48 px；section heading 26–32 px；正文至少 17–18 px；表單
  label 與 metadata 不低於 14 px。
- Mobile 主標約 34–38 px；正文至少 16 px。
- 最大內容寬度約 1200–1280 px，不使用目前 Gradio 預設的大面積上下留白。
- Result cards 在 desktop 為三欄、mobile 為單欄；chart 與 metrics 在 desktop
  並排、mobile 依閱讀順序堆疊。
- 使用細邊線與簡單 grid 建立層級；不使用 gradient、glassmorphism、連續動畫、
  emoji navigation 或裝飾性 hero illustration。
- Focus、hover、selected、disabled 與 error states 必須清楚，不只依賴顏色辨識。

## Data flow 與失敗處理

`build_demo()` 先將 committed JSON 解析成小型 presentation view model，再建立
Gradio components。UI callback 只接收 model-family 選擇並回傳既有圖表與格式化
數字，不寫檔。

若 committed JSON 缺失、schema 不符或 metric 不存在，證據頁顯示明確的
「evidence 無法驗證」狀態並停止展示該數字；不得以 hard-coded fallback 假裝
載入成功。這個錯誤不應使 `/health` 或其餘 API 無法啟動。

Gradio analytics 預設停用，避免本機 demo 在啟動時主動送出 telemetry。若 Gradio
套件未安裝或 mount 失敗，既有 FastAPI graceful fallback 保持不變。

## Pages showcase 一致性

本次將既有 static results showcase 的使用者可見文案同步改為正體中文，保留
原文專有名詞與目前已核准的 Evidence Cartography layout。這不增加頁面、功能或
新的 claims，只避免從中文 README 點入後突然切換成全英文介面。HTML `lang`
設為 `zh-TW`，machine-readable JSON 與檔名維持不變。

## 測試與 release gates

實作採 TDD，至少覆蓋：

1. `README.md` 為正體中文主版、`README_en.md` 存在、舊
   `README_zh-TW.md` 不再追蹤。
2. Report generator 與 release verifier 同步檢查兩個新 README 路徑。
3. About handoff 包含核准的正體中文 description 與英文 Topics。
4. Evidence view model 只從 canonical full summary 取值，錯誤 schema fail closed。
5. Gradio config 包含兩個頂層頁籤、正體中文 labels、無資料時不建立空白 explorer。
6. Model/method compatibility、missing checkpoint 與 path-redaction 行為維持既有測試。
7. Gradio analytics 預設停用。
8. Desktop 與 mobile Chromium smoke：無 console error、無 horizontal overflow、
   主要文字可見、model switch 更新數字／圖片、keyboard focus 可辨識。
9. Pages showcase 的正體中文文案、`lang=zh-TW`、local links 與 18-file allowlist。
10. 最終重新執行 Ruff、strict mypy、full CPU tests、package build、distribution
    audit、isolated wheel smoke、API/Gradio smoke、release verifier 與 clean export。

## 不在本次範圍

- 不加入新 explainer、模型、dataset、metric 或訓練功能。
- 不發布 weights、checkpoints 或 per-sample arrays。
- 不把 Gradio 改成遠端 live service。
- 不建立 remote、push、PR、tag、Release 或實際 Pages deployment。
- 不修改或重跑正式 full-scale L4 experiment。
