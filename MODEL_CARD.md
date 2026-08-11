# Model Card — vision-xai-reliability-lab

> **Status:** the full pipeline has been run end-to-end at **full scale** on a
> Colab NVIDIA L4 (2026-07-25): all four variants trained on the complete
> dataset, then explained, evaluated, and reported. A `smoke` config runs the
> same chain on CPU in minutes for verifying mechanics. All numbers live in the
> generated results block of [README.md](README.md) and in
> `results/derived/summary.json` — never entered by hand.

## Models

| Variant | Backbone | Weights | Training regime |
|---|---|---|---|
| `cnn` | ConvNeXt-Tiny | `torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | classifier head only (backbone frozen) |
| `vit` | ViT-B/16 | `torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1` | classifier head only (backbone frozen) |
| `cnn_patched` / `vit_patched` | as above | as above | head-only, trained with the spurious corner patch (train_correlated assignment) |

- Task: 37-class breed classification on Oxford-IIIT Pet
  ([DATA_CARD.md](DATA_CARD.md)).
- Checkpoints store only the head (+ optimizer state + config hash); the
  backbone is reproducible from the official pretrained weights. Checkpoints
  are gitignored and never committed.
- Training facts (device name, elapsed time, peak VRAM) are recorded as
  measured; on CPU the VRAM field is `null` — never fabricated.
- Partial fine-tuning is intentionally not enabled yet (spec: head-only first).
- Classification metrics recorded per variant: accuracy, macro-F1, and ECE
  (binned expected calibration error, 10 equal-width bins over the predicted
  class's softmax confidence, Guo et al. 2017) on the clean validation split.

## Explanation methods

| Method | cnn | vit | Notes |
|---|---|---|---|
| Grad-CAM | ✅ | — | captum `LayerGradCam` on the last ConvNeXt feature stage |
| Integrated Gradients | ✅ | ✅ | zero baseline = dataset mean in normalized space |
| Occlusion | ✅ | ✅ | window/stride configurable per config |
| random / uniform / center prior | ✅ | ✅ | reference baselines, deterministic per sample |

**Attention Rollout is deliberately absent**: per the project stop-loss rule it
is only added if its tensor semantics are validated; it has not been attempted
yet. Attention is never presented as explanation.

## Reliability evaluation

Implemented in `src/vision_xai/eval/`: energy-in-mask, pointing game, top-k
IoU (pre-registered fractions, never tuned on test), deletion/insertion AUC,
head-randomization sanity check (|Spearman|, low = healthy), horizontal-flip
consistency, and the spurious-patch experiment (accuracy + patch energy vs
pet-mask energy across correlated / no-patch / counter-correlated test sets).

## What the full-scale run actually found

Three results worth reading before trusting any heatmap here. The numbers are in
the generated block of [README.md](README.md); the interpretation is below.

1. **A trivial centre prior beats every real method on the pointing game.**
   Oxford-IIIT Pet photos are centred on the animal, so a fixed centre blob
   scores higher than Grad-CAM without looking at the image at all. Pointing
   game — and localization generally — is therefore weak evidence about a
   method's quality on this dataset.
2. **Integrated Gradients largely fails the model-randomization sanity check.**
   After re-initializing the classification head, IG's maps stay far more
   similar to the originals (|Spearman| ≈ 0.47–0.48) than Grad-CAM's (≈ 0.22) or
   Occlusion's (≈ 0.23). A method whose output barely moves when the model's
   learned parameters are destroyed is not, on this evidence, explaining the
   model. This reproduces the failure mode reported by Adebayo et al. (2018).
3. **The spurious-cue experiment is a negative result, reported as such.** The
   patched-trained models score almost identically on the correlated,
   no-patch, and counter-correlated test sets, and attribution energy on the
   patch stays near zero while pet-mask energy stays high. The models did not
   learn the shortcut. A plausible reading: with a frozen ImageNet backbone and
   only a linear head trainable, there is little capacity or pressure to exploit
   a small synthetic corner patch when the pet features already separate the
   classes. This does not show that models in general resist such cues — only
   that this training regime did.

## Intended use & limitations

- Research/portfolio project benchmarking attribution reliability — not a
  production pet classifier.
- Localization metrics measure *where* attributions land, **not** causal
  faithfulness; the two are reported separately by design.
- Correctly-predicted and all-prediction subsets are reported separately.
- Attribution and reliability metrics are computed over a fixed 500-sample
  subset of the test split (attribution methods are expensive), so they describe
  that subset rather than the whole split. Classification metrics use the full
  validation split. The generated `scale_note` states this on every report.
- One seed, one training regime (head-only), one dataset. Differences between
  methods here are not general claims about those methods.
- **A heatmap is not proof of causal reasoning.**
