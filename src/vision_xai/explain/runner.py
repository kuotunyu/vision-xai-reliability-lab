"""Explanation runner: iterate a manifest split, compute attributions, and write
per-sample ``.npz`` maps plus a meta JSONL — incrementally and resumably."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision_xai.config import AppConfig, SpuriousPatchConfig, config_hash
from vision_xai.data.datasets import PetClassificationDataset
from vision_xai.data.manifest import MANIFEST_FILENAME, ManifestRecord, load_manifest
from vision_xai.data.transforms import build_eval_transform, make_normalizer
from vision_xai.errors import ExplainError
from vision_xai.explain.base import check_compatibility, explain
from vision_xai.models.factory import ModelName
from vision_xai.models.loading import load_trained_model, variant_name
from vision_xai.paths import manifests_dir, raw_explanations_dir, resolve_data_dir
from vision_xai.utils.io import append_jsonl, atomic_write_json, read_jsonl
from vision_xai.utils.seed import seed_everything

logger = logging.getLogger(__name__)

TestVariant = Literal["clean", "correlated", "no_patch", "counter_correlated"]
TEST_VARIANTS: tuple[TestVariant, ...] = ("clean", "correlated", "no_patch", "counter_correlated")


@dataclass(frozen=True)
class ExplainRunResult:
    variant: str
    method: str
    test_variant: TestVariant
    num_processed_this_run: int
    num_total: int
    out_dir: Path
    elapsed_seconds: float


def method_dir_name(method: str, test_variant: TestVariant) -> str:
    return method if test_variant == "clean" else f"{method}__{test_variant}"


def _load_records(cfg: AppConfig) -> list[ManifestRecord]:
    manifest_path = manifests_dir(cfg) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ExplainError(f"manifest not found at {manifest_path} — run `data prepare` first")
    records = [r for r in load_manifest(manifest_path) if r.split == cfg.explain.split]
    records.sort(key=lambda r: r.sample_id)
    if cfg.explain.num_samples is not None:
        records = records[: cfg.explain.num_samples]
    if not records:
        raise ExplainError(f"no records in split {cfg.explain.split!r}")
    return records


def _patch_setup(
    cfg: AppConfig, test_variant: TestVariant
) -> tuple[SpuriousPatchConfig | None, dict[str, bool] | None]:
    if test_variant == "clean":
        return None, None
    if cfg.explain.split != "test":
        raise ExplainError("patched test variants require explain.split == 'test'")
    path = manifests_dir(cfg) / f"patch_assignments_test_{test_variant}.jsonl"
    if not path.is_file():
        raise ExplainError(f"patch assignments not found at {path} — rerun `data prepare`")
    assignment = {str(row["sample_id"]): bool(row["patched"]) for row in read_jsonl(path)}
    return cfg.data.spurious_patch, assignment


def run_explanations(
    cfg: AppConfig,
    model_name: ModelName,
    method: str,
    *,
    patched: bool = False,
    test_variant: TestVariant = "clean",
    resume: bool = False,
    device: str | None = None,
) -> ExplainRunResult:
    start = time.monotonic()
    seed_everything(cfg.seed)
    check_compatibility(method, model_name)
    records = _load_records(cfg)
    variant = variant_name(model_name, patched)
    out_dir = raw_explanations_dir(cfg, variant, method_dir_name(method, test_variant))
    attr_dir = out_dir / "attr"
    meta_path = out_dir / "meta.jsonl"

    done: set[str] = set()
    if meta_path.exists():
        if not resume:
            raise ExplainError(
                f"explanations already exist in {out_dir}; pass --resume to continue "
                "or delete the directory to recompute"
            )
        done = {str(row["sample_id"]) for row in read_jsonl(meta_path)}
    pending = [r for r in records if r.sample_id not in done]
    logger.info(
        "explaining %s/%s (%s): %d pending of %d total",
        variant,
        method,
        test_variant,
        len(pending),
        len(records),
    )

    bundle, resolved_device = load_trained_model(cfg, model_name, patched=patched, device=device)
    patch_cfg, assignment = _patch_setup(cfg, test_variant)
    dataset = PetClassificationDataset(
        pending,
        resolve_data_dir(cfg),
        transform=build_eval_transform(cfg.data),
        patch_cfg=patch_cfg,
        patch_assignment=assignment,
        post_patch_transform=make_normalizer(bundle.normalize_mean, bundle.normalize_std),
    )
    # shuffle is deliberately left at its default (False/SequentialSampler) so
    # each yielded batch lines up with the matching `pending[...]` slice below;
    # do not add shuffle=True here without also reworking that indexing.
    loader: DataLoader[Any] = DataLoader(dataset, batch_size=cfg.explain.batch_size)
    attr_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for batch_index, (batch_images, batch_targets) in enumerate(loader):
        batch_records = pending[
            batch_index * cfg.explain.batch_size : (batch_index + 1) * cfg.explain.batch_size
        ]
        batch_ids = [r.sample_id for r in batch_records]
        images = batch_images.to(resolved_device)
        targets = batch_targets.to(resolved_device)
        with torch.no_grad():
            probabilities = torch.softmax(bundle.model(images), dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_start = time.monotonic()
        batch = explain(
            bundle.model,
            images,
            targets,
            method,
            cfg.explain,
            model_name,
            seed=cfg.seed,
            sample_ids=batch_ids,
        )
        per_sample_latency = (time.monotonic() - batch_start) / len(batch_ids)

        rows: list[dict[str, Any]] = []
        for position, record in enumerate(batch_records):
            attribution = batch.attributions[position].numpy().astype(np.float16)
            np.savez_compressed(attr_dir / f"{record.sample_id}.npz", attribution=attribution)
            prediction = int(predictions[position])
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "target": record.class_id,
                    "prediction": prediction,
                    "probability": round(float(probabilities[position, prediction]), 6),
                    "correct": prediction == record.class_id,
                    "latency_s": round(per_sample_latency, 4),
                }
            )
        append_jsonl(meta_path, rows)
        processed += len(rows)
        logger.info("explained %d/%d", processed + len(done), len(records))

    device_name = (
        torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else "cpu"
    )
    elapsed = time.monotonic() - start
    atomic_write_json(
        out_dir / "run.json",
        {
            "schema_version": 1,
            "variant": variant,
            "method": method,
            "test_variant": test_variant,
            "split": cfg.explain.split,
            "num_samples": len(records),
            "method_metadata": {},
            "device": device_name,
            "torch_version": torch.__version__,
            "elapsed_seconds": round(elapsed, 2),
            "config_hash": config_hash(cfg),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return ExplainRunResult(
        variant=variant,
        method=method,
        test_variant=test_variant,
        num_processed_this_run=processed,
        num_total=len(records),
        out_dir=out_dir,
        elapsed_seconds=elapsed,
    )
