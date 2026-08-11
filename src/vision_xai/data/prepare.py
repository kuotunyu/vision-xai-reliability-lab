"""Orchestration of ``vision_xai data prepare``: download/verify, hash every sample
with incremental checkpoints (resume-safe), assign splits and spurious-patch
variants, and emit the manifest, fingerprint, and summary artifacts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vision_xai.config import AppConfig, config_hash
from vision_xai.data.fingerprint import build_fingerprint, sha256_file
from vision_xai.data.manifest import (
    MANIFEST_FILENAME,
    PARTIAL_MANIFEST_FILENAME,
    ManifestRecord,
    PrepareStage,
    PrepareState,
    append_records,
    clear_state,
    load_manifest,
    load_state,
    save_state,
    write_manifest,
)
from vision_xai.data.patches import PatchVariant, assign_patches, in_target_group
from vision_xai.data.source import (
    EXPECTED_COUNTS,
    DatasetSource,
    OxfordPetSource,
    SampleRef,
)
from vision_xai.data.splits import stratified_train_val_split
from vision_xai.data.trimap import verify_trimap
from vision_xai.errors import (
    DataPreparationError,
    DatasetIntegrityError,
    ResumeStateError,
)
from vision_xai.paths import manifests_dir, raw_results_dir, resolve_data_dir
from vision_xai.utils.io import atomic_write_json, write_jsonl_atomic
from vision_xai.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# If, on average, fewer than half of the image-border pixels carry the documented
# background value, the trimap semantics are almost certainly swapped.
MIN_MEAN_BORDER_BACKGROUND = 0.5


@dataclass(frozen=True)
class PrepareResult:
    completed: bool
    num_processed_this_run: int
    num_total: int
    elapsed_seconds: float
    manifest_path: Path | None = None
    fingerprint_path: Path | None = None
    counts_by_split: dict[str, int] | None = None
    content_sha256: str | None = None


@dataclass(frozen=True)
class _HashProgress:
    count: int
    stopped_early: bool


def prepare_data(
    cfg: AppConfig,
    *,
    resume: bool = False,
    max_items: int | None = None,
    source: DatasetSource | None = None,
) -> PrepareResult:
    """Run the full preparation pipeline; see the module docstring for the stages.

    ``max_items`` bounds how many *new* samples are hashed this run (used by tests
    and resume drills); a bounded run saves its checkpoint and returns
    ``completed=False``.
    """
    start = time.monotonic()
    seed_everything(cfg.seed)
    cfg_hash = config_hash(cfg)
    data_dir = resolve_data_dir(cfg)
    man_dir = manifests_dir(cfg)
    man_dir.mkdir(parents=True, exist_ok=True)
    partial_path = man_dir / PARTIAL_MANIFEST_FILENAME

    src: DatasetSource = (
        source if source is not None else OxfordPetSource(data_dir, download=cfg.data.download)
    )
    src.ensure_available()
    samples = _limit_per_class(src.list_samples(), cfg.data.limit_per_class)
    total = len(samples)
    logger.info(
        "preparing %d samples (experiment=%s, config_hash=%s)",
        total,
        cfg.experiment_name,
        cfg_hash,
    )

    completed_ids = _validate_resume_state(man_dir, partial_path, cfg_hash, resume)
    progress = _hash_samples(
        samples, completed_ids, data_dir, cfg, man_dir, partial_path, cfg_hash, max_items
    )
    if progress.stopped_early:
        logger.info(
            "stopped after %d newly hashed samples (max_items=%s); continue with --resume",
            progress.count,
            max_items,
        )
        return PrepareResult(
            completed=False,
            num_processed_this_run=progress.count,
            num_total=total,
            elapsed_seconds=time.monotonic() - start,
        )

    records = load_manifest(partial_path, require_split=False)
    records.sort(key=lambda r: r.sample_id)
    mean_border = _check_integrity(records, cfg)

    _save_stage(man_dir, "splitting", len(records), total, cfg_hash)
    final_records = _assign_splits(records, cfg)
    manifest_path = man_dir / MANIFEST_FILENAME
    write_manifest(manifest_path, final_records)

    _save_stage(man_dir, "patching", len(records), total, cfg_hash)
    patch_summary = _write_patch_assignments(final_records, cfg, man_dir)

    out_dir = raw_results_dir(cfg)
    fingerprint = build_fingerprint(final_records, _fingerprint_meta(cfg, cfg_hash, mean_border))
    fingerprint_path = out_dir / "fingerprint.json"
    atomic_write_json(fingerprint_path, fingerprint)
    atomic_write_json(out_dir / "split_summary.json", _split_summary(final_records))
    atomic_write_json(out_dir / "patch_summary.json", patch_summary)

    partial_path.unlink(missing_ok=True)
    clear_state(man_dir)
    elapsed = time.monotonic() - start
    counts: dict[str, int] = fingerprint["num_samples"]
    logger.info("data prepare complete: %s in %.1fs", counts, elapsed)
    return PrepareResult(
        completed=True,
        num_processed_this_run=progress.count,
        num_total=total,
        elapsed_seconds=elapsed,
        manifest_path=manifest_path,
        fingerprint_path=fingerprint_path,
        counts_by_split=counts,
        content_sha256=fingerprint["content_sha256"],
    )


def _limit_per_class(samples: list[SampleRef], limit: int | None) -> list[SampleRef]:
    """Keep the first ``limit`` samples (by sorted id) per (official_split, class)."""
    if limit is None:
        return samples
    counts: dict[tuple[str, int], int] = {}
    kept: list[SampleRef] = []
    for sample in samples:  # list_samples() returns them sorted by sample_id
        key = (sample.official_split, sample.class_id)
        if counts.get(key, 0) < limit:
            counts[key] = counts.get(key, 0) + 1
            kept.append(sample)
    return kept


def _validate_resume_state(
    man_dir: Path, partial_path: Path, cfg_hash: str, resume: bool
) -> set[str]:
    state = load_state(man_dir)
    if resume:
        if state is None:
            raise ResumeStateError(f"--resume requested but no checkpoint exists in {man_dir}")
        if state.config_hash != cfg_hash:
            raise ResumeStateError(
                f"config changed since the checkpoint was written (checkpoint "
                f"{state.config_hash}, current {cfg_hash}); clear {man_dir} to start over"
            )
        if partial_path.exists():
            done = {r.sample_id for r in load_manifest(partial_path, require_split=False)}
            logger.info("resuming: %d samples already hashed", len(done))
            return done
        return set()
    if state is not None or partial_path.exists():
        raise ResumeStateError(
            f"an incomplete run exists in {man_dir}; pass --resume to continue it "
            "or delete that directory to start over"
        )
    return set()


def _hash_samples(
    samples: list[SampleRef],
    completed_ids: set[str],
    data_dir: Path,
    cfg: AppConfig,
    man_dir: Path,
    partial_path: Path,
    cfg_hash: str,
    max_items: int | None,
) -> _HashProgress:
    pending = [s for s in samples if s.sample_id not in completed_ids]
    total = len(samples)
    done = len(completed_ids)
    buffer: list[ManifestRecord] = []
    processed = 0

    def flush() -> None:
        nonlocal done
        if buffer:
            append_records(partial_path, buffer)
            done += len(buffer)
            buffer.clear()
        _save_stage(man_dir, "hashing", done, total, cfg_hash)

    for position, sample in enumerate(pending):
        buffer.append(_build_record(sample, data_dir))
        processed += 1
        if len(buffer) >= cfg.data.incremental_save_every:
            flush()
            logger.info("hashed %d/%d samples", done, total)
        if max_items is not None and processed >= max_items and position + 1 < len(pending):
            flush()
            return _HashProgress(count=processed, stopped_early=True)
    flush()
    return _HashProgress(count=processed, stopped_early=False)


def _build_record(sample: SampleRef, data_dir: Path) -> ManifestRecord:
    try:
        image_sha256 = sha256_file(sample.image_path)
        trimap_sha256 = sha256_file(sample.trimap_path)
        with Image.open(sample.image_path) as image:
            width, height = image.size
        with Image.open(sample.trimap_path) as trimap:
            trimap_array = np.asarray(trimap, dtype=np.uint8)
    except OSError as exc:
        raise DataPreparationError(
            f"failed reading files for sample {sample.sample_id!r}: {exc}"
        ) from exc
    check = verify_trimap(trimap_array)
    if not check.values_ok:
        raise DatasetIntegrityError(
            f"trimap for {sample.sample_id!r} contains unexpected values {check.unique_values}"
        )
    return ManifestRecord(
        sample_id=sample.sample_id,
        official_split=sample.official_split,
        split=None,
        class_id=sample.class_id,
        class_name=sample.class_name,
        species=sample.species,
        image_relpath=sample.image_path.relative_to(data_dir).as_posix(),
        trimap_relpath=sample.trimap_path.relative_to(data_dir).as_posix(),
        image_sha256=image_sha256,
        trimap_sha256=trimap_sha256,
        width=width,
        height=height,
        trimap_values=list(check.unique_values),
        border_background_fraction=check.border_background_fraction,
    )


def _check_integrity(records: list[ManifestRecord], cfg: AppConfig) -> float:
    if not records:
        raise DataPreparationError("no samples were processed; the dataset appears empty")
    mean_border = float(np.mean([r.border_background_fraction for r in records]))
    if mean_border < MIN_MEAN_BORDER_BACKGROUND:
        raise DatasetIntegrityError(
            f"mean border-background fraction {mean_border:.2f} is below "
            f"{MIN_MEAN_BORDER_BACKGROUND}: trimap semantics look swapped relative to the "
            "official README (1=pet, 2=background). Investigate and record in FAILURES.md."
        )
    if cfg.data.limit_per_class is None:
        for split, expected in EXPECTED_COUNTS.items():
            actual = sum(1 for r in records if r.official_split == split)
            if actual != expected:
                logger.warning(
                    "official split %r has %d samples, expected %d", split, actual, expected
                )
    return mean_border


def _assign_splits(records: list[ManifestRecord], cfg: AppConfig) -> list[ManifestRecord]:
    trainval_labels = {r.sample_id: r.class_id for r in records if r.official_split == "trainval"}
    assignment = stratified_train_val_split(
        trainval_labels, cfg.data.split.val_fraction, cfg.data.split.seed
    )
    return [
        r.model_copy(
            update={"split": assignment[r.sample_id] if r.official_split == "trainval" else "test"}
        )
        for r in records
    ]


def _write_patch_assignments(
    records: list[ManifestRecord], cfg: AppConfig, man_dir: Path
) -> dict[str, Any]:
    patch_cfg = cfg.data.spurious_patch
    train_records = [r for r in records if r.split == "train"]
    test_records = [r for r in records if r.split == "test"]
    jobs: list[tuple[str, list[ManifestRecord], PatchVariant]] = [
        ("train_correlated", train_records, PatchVariant.CORRELATED)
    ]
    jobs.extend(
        (f"test_{name}", test_records, PatchVariant(name)) for name in patch_cfg.test_variants
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "enabled": patch_cfg.enabled,
        "target_group": patch_cfg.target_group,
        "p_patch_in_group": patch_cfg.p_patch_in_group,
        "p_patch_out_group": patch_cfg.p_patch_out_group,
        "assignments": {},
    }
    for name, subset, variant in jobs:
        assignment = assign_patches(subset, patch_cfg, cfg.seed, variant)
        rows = [
            {
                "sample_id": r.sample_id,
                "variant": variant.value,
                "in_target_group": in_target_group(r, patch_cfg),
                "patched": assignment[r.sample_id],
            }
            for r in subset
        ]
        write_jsonl_atomic(man_dir / f"patch_assignments_{name}.jsonl", rows)
        summary["assignments"][name] = {
            "num_samples": len(rows),
            "num_in_group": sum(1 for row in rows if row["in_target_group"]),
            "patched_in_group": sum(1 for row in rows if row["in_target_group"] and row["patched"]),
            "patched_out_group": sum(
                1 for row in rows if not row["in_target_group"] and row["patched"]
            ),
        }
        logger.info("patch assignment %r: %s", name, summary["assignments"][name])
    return summary


def _split_summary(records: list[ManifestRecord]) -> dict[str, Any]:
    per_class: dict[str, dict[str, int]] = {}
    totals = {"train": 0, "val": 0, "test": 0}
    for record in records:
        if record.split is None:  # defensive; _assign_splits fills every record
            raise DataPreparationError(f"record {record.sample_id!r} missing split")
        key = f"{record.class_id:02d}_{record.class_name}"
        entry = per_class.setdefault(key, {"train": 0, "val": 0, "test": 0})
        entry[record.split] += 1
        totals[record.split] += 1
    return {
        "schema_version": 1,
        "num_classes": len(per_class),
        "totals": totals,
        "per_class": dict(sorted(per_class.items())),
    }


def _fingerprint_meta(cfg: AppConfig, cfg_hash: str, mean_border: float) -> dict[str, Any]:
    import torchvision

    return {
        "dataset": cfg.data.dataset,
        "experiment_name": cfg.experiment_name,
        "limit_per_class": cfg.data.limit_per_class,
        "split_seed": cfg.data.split.seed,
        "val_fraction": cfg.data.split.val_fraction,
        "config_hash": cfg_hash,
        "mean_border_background_fraction": round(mean_border, 4),
        "torchvision_version": getattr(torchvision, "__version__", "unknown"),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _save_stage(
    man_dir: Path, stage: PrepareStage, completed: int, total: int, cfg_hash: str
) -> None:
    save_state(
        man_dir,
        PrepareState(
            stage=stage,
            completed=completed,
            total=total,
            config_hash=cfg_hash,
            updated_at=datetime.now(UTC).isoformat(),
        ),
    )
