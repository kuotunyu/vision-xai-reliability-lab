# Artifact and evidence boundary

## Committed full-scale evidence

`results/derived/summary.json` is the canonical machine-readable aggregate for
experiment `full`, generated on 2026-07-25 from a Google Colab NVIDIA L4 run.
All four classifier heads were trained on the complete Oxford-IIIT Pet training
split. Attribution-derived metrics use the first 500 test samples by stable
sample id; they do not describe the complete test split.

`results/derived/summary.md` is the exact generated block embedded in both
READMEs. The six PNGs under `assets/figures/` visualize the same aggregates.
The three JSON files under `results/raw/data_prepare/full/` contain only dataset
fingerprint and aggregate split/patch counts; no image, per-sample path, weight,
checkpoint, or attribution is committed.

The release manifest and JSON schemas added by the hardening work are the
integrity authority for these files. A local verifier checks their digests,
schemas, README synchronization, claim invariants, and repository boundary.

## What the evidence supports

- The center prior scores 0.922 on pointing game for both model families,
  higher than every evaluated attribution method. This is a localization result,
  not causal-faithfulness evidence.
- Integrated Gradients retains approximately 0.47–0.48 absolute Spearman
  similarity after head randomization. It does not satisfy the preregistered
  qualitative expectation that healthy randomization similarity should be low.
  No post-hoc numeric pass threshold is invented.
- The spurious-patch experiment is negative: accuracy and attribution behavior
  remain nearly unchanged across correlated, no-patch, and counter-correlated
  inputs under this frozen-backbone training regime.

## Reproducibility boundary

The aggregate result can be recomputed from raw local outputs using
`vision-xai report`, but those per-sample outputs and model checkpoints are
deliberately excluded from the public candidate. The Oxford-IIIT Pet dataset
and official pretrained weights must be downloaded from their original
providers to reproduce training.

The CUDA resume canary is separate evidence. It exercises interruption and
resume on a tiny synthetic configuration and never writes into `results/` or
`assets/`; it cannot overwrite or stand in for the full-scale experiment.
