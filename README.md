# vision-xai-reliability-lab

Compare attribution methods between a CNN (**ConvNeXt-Tiny**) and a Vision
Transformer (**ViT-B/16**) on **Oxford-IIIT Pet**, and — instead of stopping at
pretty heatmaps — quantitatively test whether the explanations are *reliable*:
localization against segmentation masks, deletion/insertion faithfulness,
parameter-randomization sanity checks, augmentation consistency, and a
synthetic spurious-cue experiment.

> A heatmap is not proof of causal reasoning. This repo measures where
> attribution methods hold up and where they break.

[繁體中文說明 → README_zh-TW.md](README_zh-TW.md)

## Project status

| Stage | Scope | Status |
|---|---|---|
| 0 | Repo scaffold: packaging (uv), lint/type/test gates, Docker CPU smoke path, CI | ✅ done |
| 1 | Data pipeline: deterministic splits, manifest + fingerprint, trimap-aligned masks, spurious-patch assignments, `--resume` | ✅ done |
| 2 | Training (classifier heads, AMP on CUDA, per-epoch checkpoints, `--resume`) | ✅ done |
| 3 | Explainers: Grad-CAM, Integrated Gradients, Occlusion + random/uniform/center baselines behind one `explain()` interface | ✅ done |
| 4 | Reliability evaluation: energy-in-mask, pointing game, top-k IoU, deletion/insertion AUC, randomization, flip consistency, patch energy | ✅ done |
| 5 | Report generation (`results/derived/summary.json` → all README numbers) | ✅ done |
| 6 | Serving: FastAPI (`/health` `/predict` `/explain` `/methods`) + Gradio at `/demo` | ✅ done |

The numbers below are the **full-scale run**: all four variants trained on the
complete dataset on a Google Colab NVIDIA L4. The immutable aggregate evidence
and its provenance are documented in [ARTIFACTS.md](ARTIFACTS.md).
Attribution and reliability metrics are computed on a fixed subset of the test
split (attribution methods are expensive) — the exact scale is stated in the
generated block itself. The repo also ships a `smoke` config that runs the same
chain on CPU in minutes, for verifying the mechanics.

## What this experiment found

1. A fixed center prior reached **0.922 pointing-game accuracy** for both model
   families, beating every evaluated attribution method. That exposes dataset
   composition bias and is localization evidence, not causal faithfulness.
2. Integrated Gradients **did not pass the model-randomization sanity
   expectation**: its maps retained about 0.47–0.48 absolute Spearman
   similarity after head randomization, substantially more than the other
   evaluated methods. The expectation was qualitative (low is healthy); no
   post-hoc numeric pass threshold is claimed.
3. The spurious-patch experiment was a **negative result**. Under this
   frozen-backbone, head-only regime, the models did not learn the intended
   shortcut. This is not evidence that vision models generally resist
   spurious cues.

## Results (generated)

Everything between the markers below is generated from
`results/derived/summary.json`; nothing is typed in by hand. Report generation
does not modify public evidence by default. Only the canonical full experiment
may opt in with `--update-public-artifacts` after review.

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

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). All commands are
identical in PowerShell and bash unless noted.

```sh
uv sync --frozen --no-editable                  # CPU-only torch, deterministic from uv.lock
uv run --no-sync python -m vision_xai.cli self-check
uv run --no-sync pytest -q                      # synthetic fixtures only, no dataset needed
```

The full pipeline (one-time ~800 MB dataset download from the official Oxford
VGG server on the first step):

```sh
uv run --no-sync python -m vision_xai.cli data prepare --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli train --model cnn --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli train --model vit --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli train --model cnn --patched --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli explain --model cnn --method gradcam --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli explain --model cnn --method integrated_gradients --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli explain --model vit --method integrated_gradients --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli evaluate --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli report --config configs/smoke.yaml
uv run --no-sync python -m vision_xai.cli serve --config configs/smoke.yaml   # API + /demo UI
```

The smoke report stays under `.artifacts/smoke/`. To regenerate the committed
full block and figures from complete local raw outputs, use the explicit guard:

```sh
uv run --no-sync python -m vision_xai.cli report --config configs/full.yaml --update-public-artifacts
```

Long runs checkpoint every 32 samples and can be interrupted at any point:

```sh
uv run --no-sync python -m vision_xai.cli data prepare --config configs/full.yaml --max-items 200
uv run --no-sync python -m vision_xai.cli data prepare --config configs/full.yaml --resume
```

Override the dataset location without editing configs:

```powershell
# PowerShell
$env:VISION_XAI_DATA_DIR = "<dataset-path>"
uv run --no-sync python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

```sh
# bash
VISION_XAI_DATA_DIR=/mnt/d/datasets/pets uv run --no-sync python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

## Docker (CPU)

```sh
docker build -t vision-xai:dev .
docker run --rm -p 8000:8000 vision-xai:dev     # serves the API on :8000
docker compose up --build                        # + mounts checkpoints/results read-only
```

The image is CPU-only by construction (torch resolves from the PyTorch CPU
index via `uv.lock`) and contains no dataset, weights, or secrets. CI never
requires a GPU.

## GPU work

The committed full-scale evidence was produced on a **Google Colab NVIDIA L4**.
CUDA is optional: local development, CI, the API health path, and all release
gates except the explicitly labelled CUDA resume canary are CPU-safe.

## Repository layout

```
configs/            smoke.yaml (tiny subset) and full.yaml
src/vision_xai/     all logic: config, data pipeline, CLI
  data/             source, splits, manifest, fingerprint, trimap, patches, transforms, datasets, prepare
tests/              pytest suite on PIL-generated synthetic fixtures (never the real dataset)
app/                FastAPI + Gradio (Stage 6)
results/            immutable full aggregates and safe data-preparation summaries
schemas/            machine-readable artifact contracts
tools/              local release and CUDA-canary verification
```

Key documents: [DATA_CARD.md](DATA_CARD.md) · [MODEL_CARD.md](MODEL_CARD.md) ·
[ARTIFACTS.md](ARTIFACTS.md) · [FAILURES.md](FAILURES.md) ·
[OWNER_ACTIONS.md](OWNER_ACTIONS.md)

## Design notes (Stage 1)

- **Deterministic everywhere.** Train/val split is stratified per class with a
  fixed seed; per-sample decisions (spurious-patch assignment) derive from
  `sha256(seed, namespace, sample_id)` so they are independent of iteration
  order, subsets, and platform.
- **Manifest + fingerprint.** Every sample's image and trimap are sha256-hashed
  into a JSONL manifest; the dataset fingerprint is order-independent, so a
  resumed run is bit-identical to an uninterrupted one.
- **Trimap semantics are verified, not assumed.** Official semantics
  (1 = pet, 2 = background, 3 = boundary) are cross-checked empirically at
  prepare time via a border-pixel heuristic.
- **Spurious patch: assignments materialized, pixels on-the-fly.** Who gets the
  patch is decided once and persisted for audit; the pixels are applied after
  resize/crop so the patch bounding box is exact in model-input coordinates —
  which the Stage-4 patch-attribution-energy metric requires.

## Development

```sh
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy src tests app tools
uv run --no-sync pytest -q
```

## License

[MIT](LICENSE). The Oxford-IIIT Pet dataset has its own license — see
[DATA_CARD.md](DATA_CARD.md).
