# app/ — serving layer

- `service.py` — shared inference service (lazy model loading, predict/explain,
  visualization-only heatmap rendering).
- `api.py` — FastAPI: `GET /health`, `GET /methods`, `POST /predict`,
  `POST /explain`. Missing checkpoints yield 503 with a clear message; `/health`
  stays 200.
- `demo.py` — Gradio UI mounted at `/demo`: a precomputed-results explorer
  (works without weights) plus a live tab (needs a trained checkpoint).

Run locally from the repository root:

```sh
uv run python -m vision_xai.cli serve --config configs/smoke.yaml
```

This package is intentionally not part of the installed wheel; Docker copies it
into the image and runs the same CLI command.
