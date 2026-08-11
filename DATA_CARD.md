# Data Card — Oxford-IIIT Pet

## Source

- **Dataset:** [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
  (Parkhi, Vedaldi, Zisserman, Jawahar — *Cats and Dogs*, CVPR 2012).
- **Access:** downloaded via `torchvision.datasets.OxfordIIITPet` from the
  official VGG server (`images.tar.gz` ≈ 790 MB, `annotations.tar.gz` ≈ 19 MB);
  torchvision verifies both archives' md5 checksums.
- **License:** the dataset page states availability under a
  [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
  license. Images are **never** committed to this repository.
- If the VGG server is slow or unreachable, download the two archives manually
  from the dataset page and place them under `<data_dir>/oxford-iiit-pet/`;
  torchvision will detect and extract pre-placed archives.

## Contents

- 7,349 images of cats and dogs: **37 breed classes** (12 cat, 25 dog),
  roughly 200 images per class. Official splits: **trainval 3,680 / test 3,669**.
- Per image: breed label (class id 1–37), species (cat/dog), and a
  **trimap** segmentation annotation.
- **Trimap semantics** (official annotations README): `1` = foreground (pet),
  `2` = background, `3` = not classified (border/ambiguous). Because
  third-party copies sometimes remap these values, `data prepare` empirically
  verifies the semantics (image-border pixels must be predominantly value 2)
  and aborts with `DatasetIntegrityError` if they look swapped.

## How this project uses it

- **Classification:** breed label (37 classes) only.
- **Segmentation trimaps:** used **only** to evaluate explanations
  (localization metrics in Stage 4). Never used for classifier training.
- **Mask policy:** by default the pet mask is `trimap == 1` (boundary pixels
  excluded); configurable via `data.mask_policy: foreground_and_boundary`.

## Splits

- Official **test** split is kept as test and is never used for tuning.
- Official **trainval** is split into train/val, stratified per class,
  `val_fraction: 0.2`, `seed: 42` (see `configs/*.yaml`). The split is a pure
  function of the sorted sample ids and the seed — reproducible on any machine.
- All thresholds/choices in later stages are decided on **val** only.

## Manifest & fingerprint

`data prepare` writes to `<data_dir>/manifests/<experiment>/` (gitignored):

- `manifest.jsonl` — one record per sample: id, official split, our split,
  class id/name, species, POSIX-relative image/trimap paths, sha256 of both
  files, image size, observed trimap values, border-background fraction.
- `patch_assignments_{train_correlated,test_correlated,test_no_patch,test_counter_correlated}.jsonl`
  — the spurious-patch experiment assignments (see below).
- `prepare_state.json` — resume checkpoint (deleted on success). Interrupted
  runs continue with `--resume`; a resumed run produces a byte-identical
  manifest.

Small, commit-eligible summaries go to `results/raw/data_prepare/<experiment>/`:
`fingerprint.json` (order-independent sha256 over
`sample_id:image_sha256:trimap_sha256` lines), `split_summary.json`,
`patch_summary.json`.

## Synthetic spurious patch

To test whether models latch onto shortcuts (and whether attribution methods
expose that), Stage 1 assigns a synthetic corner patch (default: checkerboard
magenta/black, 15% of the image side, bottom-right):

- **train_correlated:** patch present with p=0.9 for the target group
  (default: the 12 cat breeds) and p=0.1 otherwise.
- **test_correlated / test_no_patch / test_counter_correlated:** three test
  variants; *counter* swaps the probabilities.
- Assignment is deterministic per `(seed, variant, sample_id)`; pixels are
  applied on the fly **after** resize/crop, so no modified images are stored
  and the patch bounding box is exact in model-input coordinates.

## Caveats

- The **smoke config still downloads the full archives** (torchvision fetches
  whole tarballs); `limit_per_class: 3` only bounds how many samples are
  processed afterwards.
- A handful of dataset images are grayscale or CMYK; the pipeline converts all
  images to RGB at load time.
- Dataset numbers above (7,349 / 3,680 / 3,669 / 37) come from the official
  dataset page and are re-checked at prepare time; a mismatch is logged as a
  warning in the run log and visible in `fingerprint.json`.
