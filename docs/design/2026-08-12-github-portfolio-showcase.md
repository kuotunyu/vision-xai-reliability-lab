# GitHub portfolio showcase design

## Decision

Turn the existing release candidate into a portfolio-first presentation without
changing the experiment, adding models, or weakening its evidence boundary. The
approved direction combines a stronger README first screen with a static,
weight-free results showcase that can later be published through GitHub Pages.
Repository metadata and the social-preview upload remain owner actions because
they require a GitHub repository and remote writes.

## Goals and success criteria

A reviewer should understand the project in roughly 20 seconds:

1. This is a reliability-first XAI benchmark, not a heatmap gallery.
2. The center-prior, IG randomization, and spurious-patch conclusions are the
   three headline findings.
3. The conclusions come from a real full-scale L4 run, while attribution metrics
   cover a disclosed fixed 500-sample subset.
4. The engineering is reproducible and guarded by tests, artifact contracts,
   packaging audits, CPU CI, and a narrowly scoped CUDA resume canary.
5. Every public call to action works without a dataset, checkpoint, GPU, secret,
   or running backend.

The final local repository must remain clean, have no remote or tag, and pass
the existing release verifier plus all new showcase tests.

## Approaches considered

### A. Metadata-only launch

Add repository description, topics, badges, a social-preview image, and pin the
repository. This is fast but leaves the current text-heavy README and does not
turn the existing evidence into a compelling visual story.

### B. README plus static evidence showcase — selected

Create an evidence-led hero, restructure the README first screen, and add a
responsive static results experience built exclusively from committed safe
aggregates. This has the best portfolio impact without introducing model
hosting, operating cost, or new scientific claims.

### C. Hosted live inference demo

Host the FastAPI/Gradio application with checkpoints. This would be visually
appealing but conflicts with the public weight boundary, requires operational
infrastructure, and would turn a reliable local release into a service that can
fail for reasons unrelated to the research. It is explicitly out of scope.

## README information architecture

The first screen will contain:

1. A concise title and one-line reliability-first value proposition.
2. A deterministic portfolio hero derived from the committed full summary.
3. A restrained badge row for CI, Python, license, and CPU reproducibility.
4. Three evidence cards in prose: center prior, IG randomization, and the
   spurious-patch negative result.
5. Clear links to the static showcase, quickstart, artifact boundary, model
   card, and Traditional Chinese README.

The stage table and engineering detail will move below this first-screen story.
The exact generated result block will remain byte-for-byte synchronized with
`results/derived/summary.md`, but may be placed inside a collapsed
`<details>` section to keep the landing page scannable. No generated number is
manually rewritten.

The Traditional Chinese README will receive the same hierarchy and visual
links, while continuing to embed the identical generated English result block.

## Visual assets

Two committed assets will be produced from the canonical summary using a
deterministic local generator:

- `assets/portfolio/hero.png`: a wide README visual combining the project
  thesis, the three headline results, and an explicit evidence-boundary footer.
- `assets/portfolio/social-preview.png`: a 1280×640, solid-background version
  suitable for GitHub's Social Preview setting and below 1 MiB.

The generator will parse `results/derived/summary.json`; it will not accept
manually supplied metric values. Regression tests will verify that the visible
metrics and claims are sourced from the canonical aggregate. The artifact
manifest and release verifier will cover both images.

The current source archive contains only smoke-scale per-sample attribution
files; the full L4 per-sample outputs are not locally available. A qualitative
pet/heatmap montage would therefore either substitute smoke evidence or imply
full-run provenance that cannot be proven. This iteration will not create such
a montage. The aggregate evidence visualization is the honest replacement.

## Static showcase

The showcase will be a small, dependency-free site under `showcase/`:

- `index.html`: semantic document structure and complete content fallback.
- `styles.css`: responsive editorial layout, strong contrast, keyboard-visible
  focus states, and reduced-motion support.
- `app.js`: progressive enhancement for model/method filtering and evidence
  details; core findings remain readable if JavaScript is disabled.

It will display:

- the three headline findings with their interpretation boundaries;
- CNN and ViT classification context;
- localization, faithfulness, and spurious-patch figures;
- a method-by-evaluation matrix;
- CUDA canary scope and exact comparison status;
- links to the machine-readable summary, schemas, artifact manifest, data card,
  model card, and source repository README.

The public page will call itself a **results showcase**, never a live model
demo. It will not expose `/predict` or imply that checkpoints are hosted.

## Build and deployment boundary

`tools/build_showcase.py` will create a gitignored `.artifacts/showcase/`
directory. It will copy only the static site, the six committed aggregate
figures, the two portfolio images, and the approved JSON evidence needed by the
page. It must refuse unknown raw results, weights, checkpoints, symlinks,
private paths, or files above the release size limit.

A GitHub Pages workflow may build and upload this safe directory after the
owner publishes the repository and enables Pages. Pull requests will build and
audit the site without deploying it. Deployment permissions will be limited to
the Pages job, and no dataset, GPU, model weight, or secret will be required.
No Pages deployment is performed during local implementation.

## Repository metadata handoff

`OWNER_ACTIONS.md` will provide the exact post-push values:

- Description: `Reliability-first XAI benchmark for CNNs and Vision
  Transformers, with localization, faithfulness, sanity checks, and
  reproducible CUDA resume evidence.`
- Topics: `computer-vision`, `explainable-ai`, `trustworthy-ai`, `pytorch`,
  `vision-transformer`, `grad-cam`, `integrated-gradients`,
  `model-evaluation`, `fastapi`, and `gradio`.
- Homepage: the final GitHub Pages URL.
- Social Preview: `assets/portfolio/social-preview.png`.
- Profile: pin the repository after CI and Pages are green.

The CI badge will target the expected
`kuotunyu/vision-xai-reliability-lab` repository. It is a link to the real
workflow, not a manually asserted passing status.

## Verification

Implementation will use test-first changes for each new release boundary:

1. Unit tests for metric extraction and deterministic visual generation.
2. Tests for the safe showcase export allowlist and rejection paths.
3. Static checks for required headings, links, image alternative text, and
   absence of live-demo claims.
4. Browser verification at desktop and mobile widths, including no console
   errors and usable keyboard navigation.
5. README result-block synchronization and local-link verification.
6. Ruff format/check, strict mypy, full CPU tests, package build, distribution
   audit, isolated wheel smoke, and clean-export rerun.
7. Final author, committer, trailer, secret, private-path, large-file, remote,
   tag, and clean-worktree audits.

The Docker daemon gate remains conditional on local daemon availability; the
CI Docker build and health/API/Gradio checks remain unchanged.

## Non-goals

- No new explainer, backbone, dataset, metric, training run, or result claim.
- No regeneration or replacement of the full L4 aggregate.
- No publication of raw images, per-sample attributions, checkpoints, or model
  weights.
- No fake coverage badge, fake live demo, analytics, telemetry, CDN dependency,
  remote creation, push, tag, release, or deployment.
- No rewriting, squashing, or amending existing Git history.
