# Vision XAI Reliability Lab

[![CI](https://github.com/kuotunyu/vision-xai-reliability-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/vision-xai-reliability-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)

This reliability-first Vision XAI benchmark evaluates ConvNeXt-Tiny and ViT-B/16 across Localization, Faithfulness, Model Randomization, Flip Consistency, and a falsifiable Spurious Patch experiment.

Its purpose is not to produce prettier heatmaps, but to test whether they depend on learned model evidence and whether the resulting claims can be recomputed, audited, and challenged.

[Results](https://kuotunyu.github.io/vision-xai-reliability-lab/) · [Quickstart](#quickstart) · [Evidence](ARTIFACTS.md) · [Model Card](MODEL_CARD.md) · [正體中文](README.md)

![Vision XAI reliability evidence](assets/portfolio/showcase-demo-2026-08-12.png)

> **A heatmap is not proof of causal reasoning.** Statistical checks identify where explanations hold or fail, while a SHA-256 manifest prevents smoke tests or CI from rewriting formal results.

The weight-free [static showcase source](showcase/) reads only committed aggregate artifacts. It loads no model, dataset, backend, analytics, or external JavaScript.

---

## Three core findings

1. **Center Prior wins Pointing Game.** A fixed Center Prior reached **0.922** for both model families, above every evaluated Attribution method. This is evidence of dataset composition bias and Localization—not causal Faithfulness.
2. **Integrated Gradients fails Model Randomization.** Its maps retain about **0.47–0.48** absolute Spearman similarity after Head Randomization, substantially above the other evaluated methods.
3. **Spurious Patch is a negative result.** Under this frozen-backbone, head-only regime, neither model learned the intended corner-patch shortcut. This does not show that vision models generally resist spurious cues.

These are real full-scale results: all four classifier heads trained on the complete training split on an NVIDIA L4. Attribution metrics use a fixed 500-sample subset of the test split, **not the complete test split**. Aggregates, provenance, and limitations are documented in [ARTIFACTS.md](ARTIFACTS.md) and [FAILURES.md](FAILURES.md).

---

## Experiment evidence flow

This flow separates the data contract, models, Attribution, and reliability evaluation before producing public, verifiable aggregate artifacts.

```mermaid
flowchart TB
    subgraph Produce["①–③ Produce evidence"]
        direction LR
        Data["Oxford-IIIT Pet"] --> Contract["Fixed split<br/>manifest · fingerprint"]
        Contract --> Conditions["Clean<br/>Spurious Patch"]
        Conditions --> Models["ConvNeXt-Tiny<br/>ViT-B/16"]
        Models --> Attribution["Grad-CAM · Integrated Gradients<br/>Occlusion · Center/Random/Uniform"]
    end

    Attribution --> Evaluate{"④ Reliability evaluation"}
    Evaluate --> Localization["Localization"]
    Evaluate --> Dependence["Faithfulness<br/>Model Randomization"]
    Evaluate --> Robustness["Flip Consistency<br/>Spurious Patch"]
    Localization --> Aggregate["Aggregate evidence<br/>95% bootstrap CI"]
    Dependence --> Aggregate
    Robustness --> Aggregate
    Aggregate --> Integrity["JSON Schema<br/>SHA-256 manifest"]
    Integrity --> Showcase["GitHub Pages<br/>static showcase"]

    classDef data fill:#E8F4F8,stroke:#0B7285,stroke-width:2px,color:#102A43
    classDef model fill:#EDE9FE,stroke:#6D28D9,stroke-width:2px,color:#2E1065
    classDef evaluation fill:#FFF4E6,stroke:#C2410C,stroke-width:2px,color:#431407
    classDef evidence fill:#ECFDF3,stroke:#15803D,stroke-width:2px,color:#052E16
    class Data,Contract,Conditions data
    class Models,Attribution model
    class Evaluate,Localization,Dependence,Robustness evaluation
    class Aggregate,Integrity,Showcase evidence
```

## Public and local architecture

GitHub retains auditable source and aggregate evidence. Datasets, weights, and checkpoints stay in the user's local environment. Gradio can show committed evidence alongside user-supplied model output without moving local assets into the public repository.

```mermaid
flowchart LR
    subgraph Public["Public layer | GitHub (weight-free)"]
        direction TB
        Source["Source · tests · schemas"]
        Evidence["Aggregates · figures<br/>CUDA canary"]
        Gate["✓ CPU CI<br/>release verifier"]
        Pages["GitHub Pages<br/>static showcase"]
        Boundary["⊘ Not tracked<br/>dataset · weights · checkpoints"]
        Source --> Gate
        Evidence --> Gate
        Gate -->|PASS| Pages
        Gate -. "check boundary" .-> Boundary
    end

    subgraph Local["Local model layer | User environment"]
        direction TB
        Dataset["Local dataset"] --> Train["train / resume"]
        Train --> Checkpoint["User-supplied<br/>checkpoint"]
        Checkpoint --> API["FastAPI"]
        API --> Demo["Gradio evidence workbench"]
    end

    Source -. "clean install" .-> API
    Evidence -. "read-only evidence" .-> Demo

    classDef public fill:#E8F4F8,stroke:#0B7285,stroke-width:2px,color:#102A43
    classDef verified fill:#ECFDF3,stroke:#15803D,stroke-width:2px,color:#052E16
    classDef local fill:#F3E8FF,stroke:#7E22CE,stroke-width:2px,color:#3B0764
    classDef boundary fill:#FFF1F2,stroke:#BE123C,stroke-width:2px,color:#4C0519
    class Source,Evidence public
    class Gate,Pages verified
    class Dataset,Train,Checkpoint,API,Demo local
    class Boundary boundary
```

---

## Release status

- **Full-scale evidence: verified.** Four classifier heads ran on the complete training split on an NVIDIA L4; the formal aggregates are protected by a SHA-256 manifest.
- **CUDA resume canary: PASS.** A tiny synthetic RTX 4090 canary compared uninterrupted and epoch-boundary resume runs. Final head, optimizer, GradScaler, and stable metrics were exactly equal. It is not evidence for full training, and the training loop has no scheduler.
- **Release gates: included.** Ruff, strict mypy, CPU tests, package checks, the showcase allowlist, and the release verifier require no GPU, dataset, weights, or secrets.
- **Publication boundary: explicit.** The repository and GitHub Pages contain only source, aggregates, figures, and documentation. Datasets, weights, checkpoints, and runtime outputs are excluded; local Gradio use requires user-supplied checkpoints.

---

## Results (Generated)

Everything between the markers below is generated from `results/derived/summary.json`; nothing is typed in by hand:

<details>
<summary><strong>Full machine-generated result tables</strong></summary>

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

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
# 1. Sync locked environment (CPU-only Torch)
uv sync --frozen --no-editable

# 2. Self-check and unit tests (uses synthetic fixtures only)
uv run --no-sync python -m vision_xai.cli self-check
uv run --no-sync pytest -q
```

### Dataset Preparation (~800 MB from official Oxford VGG server)

```bash
uv run --no-sync python -m vision_xai.cli data prepare --config configs/smoke.yaml
```

Long runs checkpoint every 32 samples and can be resumed safely:

```bash
uv run --no-sync python -m vision_xai.cli data prepare --config configs/full.yaml --max-items 200
uv run --no-sync python -m vision_xai.cli data prepare --config configs/full.yaml --resume
```

---

## Docker (CPU)

```bash
# Build image and serve API on :8000
docker build -t vision-xai:dev .
docker run --rm -p 8000:8000 vision-xai:dev

# Or use Docker Compose with read-only mounts
docker compose up --build
```

---

## Project Structure & Documentation

```text
configs/            smoke.yaml and full.yaml
src/vision_xai/     Core implementation: config, data pipeline, models, CLI
  data/             source, splits, manifest, fingerprint, trimap, patches, prepare
tests/              pytest unit tests (synthetic fixtures, zero remote dataset dependencies)
app/                FastAPI backend and Gradio interface (Stage 6)
results/            Immutable full aggregate results and safe dataset summary
schemas/            Machine-readable artifact contracts (JSON Schema)
tools/              Release verification and CUDA resume canary auditing tools
```

- [DATA_CARD.md](DATA_CARD.md): Dataset provenance, split manifests, and license boundaries.
- [MODEL_CARD.md](MODEL_CARD.md): Architecture specifications and evaluation scopes.
- [ARTIFACTS.md](ARTIFACTS.md): Immutable release artifacts and SHA-256 checksums.
- [FAILURES.md](FAILURES.md): Negative results and failure analysis.

---

## License

Source code licensed under the [MIT License](LICENSE). The Oxford-IIIT Pet dataset is licensed separately by its authors (see [DATA_CARD.md](DATA_CARD.md)).
