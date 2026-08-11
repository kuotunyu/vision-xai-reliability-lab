"""Evaluation runner: scans completed explanation runs and computes, per family,
per-sample JSONL rows (incremental — already-evaluated samples are skipped).

Families
--------
- localization (clean runs, all methods): energy-in-mask, pointing game, top-k IoU
- faithfulness (clean runs, all methods): deletion / insertion AUC
- randomization (clean runs, real methods): |Spearman| vs head-randomized model
- consistency (clean runs, real methods): |Spearman| under horizontal flip
- spurious (patched test variants): patch energy vs pet-mask energy + predictions

Correct-vs-all subsets are separated downstream via the per-row ``correct`` flag.
Top-k fractions are pre-registered in the config and reported in full — never
tuned on the test set.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image as PILImage

from vision_xai.config import AppConfig, config_hash
from vision_xai.data.manifest import MANIFEST_FILENAME, ManifestRecord, load_manifest
from vision_xai.data.patches import patch_mask
from vision_xai.data.transforms import build_eval_transform, build_mask_transform, make_normalizer
from vision_xai.errors import EvaluationError
from vision_xai.eval.faithfulness import perturbation_curves
from vision_xai.eval.localization import energy_in_mask, pointing_game_hit, topk_iou
from vision_xai.eval.randomization import randomize_head
from vision_xai.eval.similarity import abs_spearman
from vision_xai.explain.base import REAL_METHODS, explain
from vision_xai.models.factory import IMAGENET_MEAN, IMAGENET_STD, ModelName
from vision_xai.models.loading import load_trained_model
from vision_xai.paths import manifests_dir, raw_eval_dir, resolve_data_dir, resolve_results_dir
from vision_xai.utils.io import append_jsonl, atomic_write_json, read_jsonl
from vision_xai.utils.seed import seed_everything

logger = logging.getLogger(__name__)

RANDOMIZATION_SEED_OFFSET = 1000


@dataclass(frozen=True)
class RunInfo:
    variant: str
    method: str
    test_variant: str
    directory: Path


@dataclass(frozen=True)
class EvalRunResult:
    rows_written: dict[str, int]
    out_dir: Path
    elapsed_seconds: float


def _variant_parts(variant: str) -> tuple[ModelName, bool]:
    patched = variant.endswith("_patched")
    base = variant.removesuffix("_patched")
    if base not in ("cnn", "vit"):
        raise EvaluationError(f"cannot parse model name from variant {variant!r}")
    return ("cnn" if base == "cnn" else "vit"), patched


class _Context:
    """Caches shared across families: records, masks, models, and transforms."""

    def __init__(self, cfg: AppConfig, device: str | None) -> None:
        self.cfg = cfg
        self.device_arg = device
        manifest_path = manifests_dir(cfg) / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise EvaluationError(f"manifest not found at {manifest_path}")
        self.records: dict[str, ManifestRecord] = {
            record.sample_id: record for record in load_manifest(manifest_path)
        }
        self.data_root = resolve_data_dir(cfg)
        self._mask_transform = build_mask_transform(cfg.data)
        self._image_transform = build_eval_transform(cfg.data)
        self._normalize = make_normalizer(IMAGENET_MEAN, IMAGENET_STD)
        self._masks: dict[str, npt.NDArray[np.bool_]] = {}
        self._models: dict[str, tuple[Any, torch.device]] = {}

    def record(self, sample_id: str) -> ManifestRecord:
        if sample_id not in self.records:
            raise EvaluationError(f"sample {sample_id!r} not present in the manifest")
        return self.records[sample_id]

    def mask_for(self, sample_id: str) -> npt.NDArray[np.bool_]:
        if sample_id not in self._masks:
            record = self.record(sample_id)
            with PILImage.open(self.data_root / record.trimap_relpath) as trimap:
                self._masks[sample_id] = self._mask_transform(trimap)
        return self._masks[sample_id]

    def clean_image_for(self, sample_id: str) -> torch.Tensor:
        record = self.record(sample_id)
        with PILImage.open(self.data_root / record.image_relpath) as raw:
            return self._normalize(self._image_transform(raw.convert("RGB")))

    def model_for(self, variant: str) -> tuple[Any, torch.device]:
        if variant not in self._models:
            model_name, patched = _variant_parts(variant)
            self._models[variant] = load_trained_model(
                self.cfg, model_name, patched=patched, device=self.device_arg
            )
        return self._models[variant]


def _discover_runs(cfg: AppConfig) -> list[RunInfo]:
    root = resolve_results_dir(cfg) / "raw" / "explanations" / cfg.experiment_name
    runs: list[RunInfo] = []
    for run_json in sorted(root.glob("*/*/run.json")):
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        runs.append(
            RunInfo(
                variant=str(payload["variant"]),
                method=str(payload["method"]),
                test_variant=str(payload["test_variant"]),
                directory=run_json.parent,
            )
        )
    return runs


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["sample_id"]) for row in read_jsonl(path)}


def _load_attribution(run: RunInfo, sample_id: str) -> npt.NDArray[np.float32]:
    path = run.directory / "attr" / f"{sample_id}.npz"
    if not path.is_file():
        raise EvaluationError(f"missing attribution file {path}")
    loaded: npt.NDArray[np.float32] = np.load(path)["attribution"].astype(np.float32)
    return loaded


def _family_out(cfg: AppConfig, run: RunInfo, family: str) -> Path:
    return raw_eval_dir(cfg) / run.variant / run.directory.name / f"{family}.jsonl"


def _run_localization(ctx: _Context, runs: list[RunInfo]) -> int:
    written = 0
    fractions = ctx.cfg.evaluate.topk_fractions
    for run in (r for r in runs if r.test_variant == "clean"):
        out_path = _family_out(ctx.cfg, run, "localization")
        done = _done_ids(out_path)
        rows: list[dict[str, Any]] = []
        for meta in read_jsonl(run.directory / "meta.jsonl"):
            sample_id = str(meta["sample_id"])
            if sample_id in done:
                continue
            attribution = _load_attribution(run, sample_id)
            mask = ctx.mask_for(sample_id)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "correct": bool(meta["correct"]),
                "energy_in_mask": round(energy_in_mask(attribution, mask), 6),
                "pointing_hit": pointing_game_hit(attribution, mask),
            }
            for fraction in fractions:
                row[f"topk_iou_{fraction}"] = round(topk_iou(attribution, mask, fraction), 6)
            rows.append(row)
        append_jsonl(out_path, rows)
        written += len(rows)
    return written


def _run_faithfulness(ctx: _Context, runs: list[RunInfo]) -> int:
    written = 0
    for run in (r for r in runs if r.test_variant == "clean"):
        out_path = _family_out(ctx.cfg, run, "faithfulness")
        done = _done_ids(out_path)
        bundle, device = ctx.model_for(run.variant)
        for meta in read_jsonl(run.directory / "meta.jsonl"):
            sample_id = str(meta["sample_id"])
            if sample_id in done:
                continue
            record = ctx.record(sample_id)
            result = perturbation_curves(
                bundle.model,
                ctx.clean_image_for(sample_id),
                record.class_id,
                _load_attribution(run, sample_id),
                steps=ctx.cfg.evaluate.deletion_steps,
                batch_size=ctx.cfg.evaluate.batch_size,
                device=device,
            )
            append_jsonl(
                out_path,
                [
                    {
                        "sample_id": sample_id,
                        "correct": bool(meta["correct"]),
                        "deletion_auc": result.deletion_auc,
                        "insertion_auc": result.insertion_auc,
                        "fractions": result.fractions,
                        "deletion_curve": result.deletion_curve,
                        "insertion_curve": result.insertion_curve,
                    }
                ],
            )
            written += 1
    return written


def _similarity_rows(
    ctx: _Context,
    run: RunInfo,
    sample_ids: list[str],
    other_attributions: torch.Tensor,
    flip: bool,
    correct_by_id: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, sample_id in enumerate(sample_ids):
        other = other_attributions[position].numpy()
        if flip:
            other = np.flip(other, axis=-1).copy()
        similarity = abs_spearman(_load_attribution(run, sample_id), other)
        rows.append(
            {
                "sample_id": sample_id,
                "correct": correct_by_id[sample_id],
                "abs_spearman": round(similarity, 6),
            }
        )
    return rows


def _explain_batch(
    ctx: _Context,
    run: RunInfo,
    model: Any,
    device: torch.device,
    sample_ids: list[str],
    flip: bool,
) -> torch.Tensor:
    images = torch.stack([ctx.clean_image_for(sample_id) for sample_id in sample_ids])
    if flip:
        images = torch.flip(images, dims=[-1])
    targets = torch.tensor([ctx.record(sample_id).class_id for sample_id in sample_ids])
    model_name, _ = _variant_parts(run.variant)
    batch = explain(
        model,
        images.to(device),
        targets.to(device),
        run.method,
        ctx.cfg.explain,
        model_name,
        seed=ctx.cfg.seed,
        sample_ids=sample_ids,
    )
    return batch.attributions


def _run_randomization(ctx: _Context, runs: list[RunInfo]) -> int:
    written = 0
    for run in (r for r in runs if r.test_variant == "clean" and r.method in REAL_METHODS):
        out_path = _family_out(ctx.cfg, run, "randomization")
        done = _done_ids(out_path)
        meta = read_jsonl(run.directory / "meta.jsonl")
        correct_by_id = {str(row["sample_id"]): bool(row["correct"]) for row in meta}
        sample_ids = [
            str(row["sample_id"])
            for row in meta[: ctx.cfg.evaluate.randomization_samples]
            if str(row["sample_id"]) not in done
        ]
        if not sample_ids:
            continue
        model_name, patched = _variant_parts(run.variant)
        # A fresh model instance: the cached one must never be mutated.
        bundle, device = load_trained_model(
            ctx.cfg, model_name, patched=patched, device=ctx.device_arg
        )
        randomize_head(bundle.model, model_name, ctx.cfg.seed + RANDOMIZATION_SEED_OFFSET)
        attributions = _explain_batch(ctx, run, bundle.model, device, sample_ids, flip=False)
        rows = _similarity_rows(
            ctx, run, sample_ids, attributions, flip=False, correct_by_id=correct_by_id
        )
        append_jsonl(out_path, rows)
        written += len(rows)
    return written


def _run_consistency(ctx: _Context, runs: list[RunInfo]) -> int:
    written = 0
    for run in (r for r in runs if r.test_variant == "clean" and r.method in REAL_METHODS):
        out_path = _family_out(ctx.cfg, run, "consistency")
        done = _done_ids(out_path)
        meta = read_jsonl(run.directory / "meta.jsonl")
        correct_by_id = {str(row["sample_id"]): bool(row["correct"]) for row in meta}
        sample_ids = [
            str(row["sample_id"])
            for row in meta[: ctx.cfg.evaluate.consistency_samples]
            if str(row["sample_id"]) not in done
        ]
        if not sample_ids:
            continue
        bundle, device = ctx.model_for(run.variant)
        attributions = _explain_batch(ctx, run, bundle.model, device, sample_ids, flip=True)
        rows = _similarity_rows(
            ctx, run, sample_ids, attributions, flip=True, correct_by_id=correct_by_id
        )
        append_jsonl(out_path, rows)
        written += len(rows)
    return written


def _run_spurious(ctx: _Context, runs: list[RunInfo]) -> int:
    written = 0
    region = patch_mask(ctx.cfg.data.image_size, ctx.cfg.data.spurious_patch)
    for run in (r for r in runs if r.test_variant != "clean"):
        out_path = _family_out(ctx.cfg, run, "spurious")
        done = _done_ids(out_path)
        assignment_path = (
            manifests_dir(ctx.cfg) / f"patch_assignments_test_{run.test_variant}.jsonl"
        )
        if not assignment_path.is_file():
            raise EvaluationError(f"patch assignments not found at {assignment_path}")
        assignment = {
            str(row["sample_id"]): bool(row["patched"]) for row in read_jsonl(assignment_path)
        }
        rows: list[dict[str, Any]] = []
        for meta in read_jsonl(run.directory / "meta.jsonl"):
            sample_id = str(meta["sample_id"])
            if sample_id in done:
                continue
            attribution = _load_attribution(run, sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "test_variant": run.test_variant,
                    "correct": bool(meta["correct"]),
                    "prediction": int(meta["prediction"]),
                    "target": int(meta["target"]),
                    "patched_input": assignment.get(sample_id, False),
                    "patch_energy": round(energy_in_mask(attribution, region), 6),
                    "pet_mask_energy": round(
                        energy_in_mask(attribution, ctx.mask_for(sample_id)), 6
                    ),
                }
            )
        append_jsonl(out_path, rows)
        written += len(rows)
    return written


def evaluate_all(cfg: AppConfig, *, device: str | None = None) -> EvalRunResult:
    """Run every applicable evaluation family; incremental and safe to rerun."""
    start = time.monotonic()
    seed_everything(cfg.seed)
    runs = _discover_runs(cfg)
    if not runs:
        raise EvaluationError("no explanation runs found — run `explain` first")
    ctx = _Context(cfg, device)
    out_dir = raw_eval_dir(cfg)

    rows_written = {
        "localization": _run_localization(ctx, runs),
        "faithfulness": _run_faithfulness(ctx, runs),
        "randomization": _run_randomization(ctx, runs),
        "consistency": _run_consistency(ctx, runs),
        "spurious": _run_spurious(ctx, runs),
    }
    atomic_write_json(
        out_dir / "thresholds.json",
        {
            "schema_version": 1,
            "topk_fractions": cfg.evaluate.topk_fractions,
            "policy": (
                "top-k fractions are pre-registered in the config and reported in "
                "full; no threshold is tuned on the test set"
            ),
        },
    )
    elapsed = time.monotonic() - start
    atomic_write_json(
        out_dir / "run.json",
        {
            "schema_version": 1,
            "rows_written_this_run": rows_written,
            "num_explanation_runs": len(runs),
            "elapsed_seconds": round(elapsed, 2),
            "config_hash": config_hash(cfg),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    logger.info("evaluation complete: %s in %.1fs", rows_written, elapsed)
    return EvalRunResult(rows_written=rows_written, out_dir=out_dir, elapsed_seconds=elapsed)
