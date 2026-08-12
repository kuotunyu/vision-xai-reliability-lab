# app/ — 本機 serving layer

- `service.py`：共用 inference service，負責 lazy model loading、predict／explain 與
  visualization-only heatmap rendering。
- `api.py`：FastAPI 的 `GET /health`、`GET /methods`、`POST /predict`、
  `POST /explain`。缺少 checkpoint 時 inference endpoint 回傳 503，`/health`
  仍維持 200。
- `evidence.py`：只讀取 committed full-scale aggregate、六張 figures 與 CUDA resume
  canary；缺失或不是 canonical `full` 時 fail closed。
- `demo.py`：掛載於 `/demo` 的正體中文 Gradio 證據工作台，只有「實驗證據」與
  「本機模型」兩層。

從 repository root 啟動：

```powershell
$env:GRADIO_ANALYTICS_ENABLED = "False"
uv run python -m vision_xai.cli serve --config configs/smoke.yaml
```

開啟 `http://127.0.0.1:8000/demo/`。公開候選不含 weights，因此「實驗證據」可直接
使用，「本機模型」會顯示 checkpoint readiness；只有自行放入相容 checkpoint 後才會
啟用 image upload 與 attribution controls。

`app/` 刻意不放入 installed wheel；Docker 會明確複製 serving code 與 allowlisted
公開 evidence，並使用同一個 CLI command 啟動。
