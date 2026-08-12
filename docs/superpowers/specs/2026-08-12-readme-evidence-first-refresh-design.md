# README Evidence-First Refresh Design

**Date:** 2026-08-12

**Status:** Approved direction; implementation pending

**Scope:** `README.md`, `README_en.md`, the README hero screenshot, and their release checks

## Objective

Make the GitHub landing experience concise, visually current, and technically
credible. Traditional Chinese remains the primary language, while established
Computer Vision and XAI terms stay in their original form instead of appearing
as repetitive Chinese-English pairs.

This refresh must preserve the canonical full-scale L4 result block and its
claim boundaries. It does not add features, experiments, models, explainers, or
remote publication actions.

## Chosen approach

Use an evidence-first README rather than a long project report or an extremely
minimal landing page. The first screen should establish what the project tests,
show the current Pages interface, and link directly to verification material.
Engineering depth remains available below the fold.

## Information architecture

The primary README will use this order:

1. `Vision XAI Reliability Lab` title.
2. Four badges: CI, Python, PyTorch, and MIT License.
3. A two-sentence positioning statement.
4. One compact navigation row:
   `成果展示 · 快速開始 · 實驗證據 · Model Card · English`.
5. A current screenshot of the Pages evidence workbench.
6. A short scientific caution explaining that a plausible Heatmap is not by
   itself evidence of model Faithfulness, plus the weight-free public boundary.
7. Three concise findings with the committed numbers and limitations.
8. One consolidated Mermaid pipeline.
9. Four release-status bullets in place of the seven-row all-complete progress
   table.
10. The existing machine-generated result block inside `<details>`.
11. Quickstart, Docker, repository structure, and document links.

The English secondary README will preserve the same navigation and evidence
hierarchy, without requiring sentence-by-sentence mirroring.

## Language policy

Traditional Chinese supplies the grammar and explanation. These established
terms remain in original form where they are clearer than a translated label:

- Vision XAI, Attribution, Heatmap
- Localization, Faithfulness, Model Randomization, Flip Consistency
- Spurious Patch, Center Prior, Pointing Game
- Integrated Gradients, Head Randomization
- ConvNeXt-Tiny, ViT-B/16, CUDA AMP
- SHA-256 manifest, full-scale L4 run, test split

Avoid a translated phrase immediately followed by the same English term in
parentheses. Translate ordinary prose and actions, not recognized method,
metric, architecture, or artifact names.

Claims use restrained language:

- Localization does not establish causal Faithfulness.
- IG did not pass the recorded Model Randomization sanity check.
- The Spurious Patch experiment is a negative result limited to its setup.
- A SHA-256 manifest verifies file identity; it is not described as a digital
  signature or proof of immutability.

## Links and badges

The link row uses short labels with centered-dot separators. It removes arrow
prefixes and filenames from visible labels. Every link must resolve in GitHub's
README renderer, and local links remain covered by the release verifier.

FastAPI and Gradio badges are removed because they describe supporting
interfaces rather than the portfolio's core research contribution. Their
documentation remains in the body.

## Hero screenshot

Replace `assets/portfolio/hero.png` with a current desktop capture of the
weight-free Pages showcase:

- approximately 1440 by 900 pixels;
- no browser chrome;
- captured from the local allowlisted showcase build;
- includes the title, run scope, and all three core finding cards;
- excludes the long evidence figures below the fold;
- contains no dataset samples, weights, checkpoints, paths, or secrets.

The screenshot is a derived presentation artifact, not experimental evidence.
Its SHA-256 entry in `release/artifact-manifest.json` must be refreshed.
`social-preview.png` stays unchanged unless its existing validation fails.

## Consolidated architecture diagram

Replace the two dense Mermaid diagrams with one left-to-right flow:

`Oxford-IIIT Pet → deterministic data pipeline → ConvNeXt / ViT → Attribution → reliability evaluation → verified artifacts → Pages / local API`

Labels stay short and use original technical terms. The diagram must remain
legible in GitHub's default light and dark themes and on narrow layouts.

## Release-status summary

Replace the all-complete stage table with four factual bullets:

- committed full-scale NVIDIA L4 aggregate;
- fixed 500-sample Attribution subset, explicitly not the whole test split;
- CUDA resume canary limited to synthetic epoch-boundary recovery;
- CPU-only CI/package/Docker gates and weight-free Pages showcase.

## Validation

Implementation will follow TDD. Regression tests will first require:

- the compact link labels and reduced badge set;
- the approved original technical terms without redundant translations;
- one Mermaid diagram instead of two;
- absence of the obsolete project-progress table;
- the new hero asset dimensions and refreshed manifest hash;
- preservation of both README language links and the generated result markers.

Final gates include Ruff format/check, strict mypy, the full CPU suite, release
verifier, showcase build/audit, package build/audit, isolated wheel smoke, and a
rendered GitHub/Pages visual check. No push, tag, release, or deployment occurs
as part of this implementation.

## Recommended GitHub topics

Use these seven topics:

1. `computer-vision`
2. `explainable-ai`
3. `trustworthy-ai`
4. `model-interpretability`
5. `model-evaluation`
6. `reproducibility`
7. `pytorch`
