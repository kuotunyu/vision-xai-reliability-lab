# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

主要讀者是臺灣的 Computer Vision、Machine Learning 與 Trustworthy AI 招募主管、技術面試官與工程同儕。他們會在 GitHub 上快速判斷：這份作品是否有可信的實驗設計、可追溯證據、可重現工程能力，以及對負結果的誠實解讀。

## Product Purpose

`vision-xai-reliability-lab` 是一份 reliability-first XAI portfolio。它以 ConvNeXt 與 ViT 的 attribution 實驗為核心，讓讀者先看到可驗證結論，再依需要進入方法、artifact 與本機 inference 細節。成功標準不是宣稱某個 explainer 最好，而是讓每項公開 claim 都能回到 versioned evidence 與明確限制。

## Positioning

本專案把 baseline、sanity check、negative result 與 resume reproducibility 當成第一級成果，而不是只展示視覺上吸睛的 heatmap。它明確區分 localization、causal faithfulness 與小型 CUDA resume canary，並保留不利於方法的真實 full-scale 結果。

## Operating Context

讀者通常先瀏覽 GitHub README 或靜態 showcase，再選擇於本機啟動 FastAPI／Gradio。公開候選不攜帶 dataset、weights 或 checkpoints，因此「實驗證據」必須在乾淨 clone 後即可閱讀；「本機模型」只在使用者自行準備相容 checkpoint 時啟用。

## Capabilities and Constraints

- 公開展示 committed full-scale aggregate results、六張 aggregate figures 與 CUDA resume canary summary。
- 支援本機 API 與 Gradio inference，但不把 model weights 納入公開 artifact。
- CI 與 release verification 不依賴 GPU、dataset、weights 或 secrets。
- tiny CUDA canary 只驗證 resume mechanism，不代表 full training equivalence。
- 不新增 explainer、model 或產品功能；目前工作只強化呈現、證據邊界與 release reliability。
- 不執行 remote、push、PR、tag、Release、HF upload 或 deployment。

## Brand Commitments

- 公開入口與介面以正體中文（`zh-TW`）為主，Computer Vision、Machine Learning、Trustworthy AI、XAI、localization、faithfulness、sanity check 等專有名詞保留原文。
- 保留完整英文副版 README。
- 語氣克制、技術導向、證據優先；不可把 500-sample attribution subset 說成完整 test split，也不可把 localization 包裝成 causal faithfulness。

## Evidence on Hand

- Full-scale machine-readable result：`results/derived/summary.json`
- CUDA resume canary summary：`release/cuda-resume-canary.json`
- Aggregate figures：`assets/figures/`
- Artifact hashes：`release/manifest.sha256`
- Model 與 data 使用邊界：`MODEL_CARD.md`、`DATA_CARD.md`
- 已確認的核心結論：center prior 在 pointing game 勝過實際 attribution method；IG 未通過 model-randomization sanity check；spurious-patch experiment 是負結果。
- 公開候選沒有 per-sample attribution arrays、dataset、weights 或 checkpoints；不得為了展示而捏造。

## Product Principles

1. Evidence before interpretation：先給數值、來源與限制，再提供結論。
2. Negative results are results：如實突出 baseline 勝出與 sanity check 失敗。
3. Public by construction：介面與 package 在缺少私人／大型 artifact 時仍可安全運作。
4. Reproducibility with scope：每項可重現性證據都標明它能證明與不能證明的範圍。
5. Dense, legible communication：讓技術讀者能快速掃描，也能逐層追查證據。

## Accessibility & Inclusion

桌面與 mobile web 都必須可閱讀；使用較大字級、清楚對比、鍵盤 focus 與非僅靠顏色傳達狀態。介面主要語言標示為 `zh-TW`。
