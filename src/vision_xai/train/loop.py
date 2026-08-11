"""Head-only training loop with per-epoch checkpoints and ``--resume``.

Checkpoints store only the classification head (small, reproducible on top of
the official pretrained backbone) plus optimizer state and the config hash.
GPU facts (device name, peak VRAM) are recorded as measured; on CPU the VRAM
field is ``null`` — never fabricated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vision_xai.config import AppConfig, config_hash, train_config_hash
from vision_xai.data.datasets import PetClassificationDataset
from vision_xai.data.manifest import MANIFEST_FILENAME, ManifestRecord, load_manifest
from vision_xai.data.transforms import build_eval_transform, build_train_transform, make_normalizer
from vision_xai.errors import ResumeStateError, TrainingError
from vision_xai.models.factory import ModelBundle, ModelName, build_model, head_module
from vision_xai.paths import manifests_dir, raw_train_dir, resolve_checkpoints_dir, resolve_data_dir
from vision_xai.train.metrics import accuracy, expected_calibration_error, macro_f1
from vision_xai.utils.io import append_jsonl, atomic_write_json, read_jsonl
from vision_xai.utils.seed import seed_everything

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "last.pt"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainResult:
    variant: str
    epochs_run_this_call: int
    total_epochs_completed: int
    final_val_accuracy: float
    final_val_macro_f1: float
    final_val_ece: float
    checkpoint_path: Path
    device: str
    elapsed_seconds: float
    peak_vram_mb: float | None


def variant_name(model_name: ModelName, patched: bool) -> str:
    return f"{model_name}_patched" if patched else model_name


def load_head_checkpoint(
    path: Path,
    expected_config_hash: str | None = None,
    expected_train_config_hash: str | None = None,
) -> dict[str, Any]:
    """Load and validate a training checkpoint written by :func:`train_model`.

    ``expected_config_hash`` covers experiment/seed/data (checked by both
    inference loading and training resume). ``expected_train_config_hash``
    additionally guards that no train hyperparameter (batch size, lr, weight
    decay, AMP, pretrained) changed — only meaningful for --resume, so
    inference callers (:mod:`vision_xai.models.loading`) leave it unset.
    """
    if not path.is_file():
        raise TrainingError(f"checkpoint not found: {path}")
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError(f"unsupported checkpoint schema in {path}")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ResumeStateError(
            f"config changed since checkpoint {path} was written "
            f"(checkpoint {payload['config_hash']}, current {expected_config_hash})"
        )
    if (
        expected_train_config_hash is not None
        and payload.get("train_config_hash") != expected_train_config_hash
    ):
        raise ResumeStateError(
            f"train hyperparameters (batch_size/lr/weight_decay/amp/pretrained) changed "
            f"since checkpoint {path} was written, or it predates hyperparameter "
            "tracking; resuming with different hyperparameters would not reproduce "
            "an uninterrupted run — delete the checkpoint to start over"
        )
    return payload


def _load_split_records(cfg: AppConfig) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    manifest_path = manifests_dir(cfg) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise TrainingError(
            f"manifest not found at {manifest_path} — run `data prepare` for this config first"
        )
    records = load_manifest(manifest_path)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    if not train_records or not val_records:
        raise TrainingError("manifest has an empty train or val split")
    return train_records, val_records


def _train_patch_assignment(cfg: AppConfig) -> dict[str, bool]:
    path = manifests_dir(cfg) / "patch_assignments_train_correlated.jsonl"
    if not path.is_file():
        raise TrainingError(f"patch assignments not found at {path} — rerun `data prepare`")
    return {str(row["sample_id"]): bool(row["patched"]) for row in read_jsonl(path)}


def _evaluate(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, num_classes: int
) -> tuple[float, float, float]:
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_confidences: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            confidence, prediction = probabilities.max(dim=1)
            all_predictions.append(prediction.cpu())
            all_confidences.append(confidence.cpu())
            all_targets.append(targets)
    predictions = torch.cat(all_predictions)
    confidences = torch.cat(all_confidences)
    targets = torch.cat(all_targets)
    ece = expected_calibration_error(confidences, predictions == targets)
    return accuracy(predictions, targets), macro_f1(predictions, targets, num_classes), ece


def _build_datasets(
    cfg: AppConfig,
    bundle: ModelBundle,
    train_records: list[ManifestRecord],
    val_records: list[ManifestRecord],
    patched: bool,
) -> tuple[PetClassificationDataset, PetClassificationDataset]:
    data_root = resolve_data_dir(cfg)
    normalizer = make_normalizer(bundle.normalize_mean, bundle.normalize_std)
    train_dataset = PetClassificationDataset(
        train_records,
        data_root,
        transform=build_train_transform(cfg.data),
        patch_cfg=cfg.data.spurious_patch if patched else None,
        patch_assignment=_train_patch_assignment(cfg) if patched else None,
        post_patch_transform=normalizer,
        train_augment_seed=cfg.seed,
    )
    # Validation is clean (no patch) for every variant: val drives threshold and
    # model selection, and those decisions must not depend on the injected cue.
    val_dataset = PetClassificationDataset(
        val_records,
        data_root,
        transform=build_eval_transform(cfg.data),
        post_patch_transform=normalizer,
    )
    return train_dataset, val_dataset


def _epoch_train_loader(
    cfg: AppConfig, train_dataset: PetClassificationDataset, epoch: int
) -> DataLoader[Any]:
    """A fresh shuffled DataLoader for one epoch, seeded by ``(cfg.seed, epoch)``.

    A single generator reused across the whole training loop would make epoch
    N's shuffle order depend on how many epochs ran earlier *in this process* —
    fine for one uninterrupted call, but wrong across --resume: a resumed call
    is a brand-new process whose first epoch would replay epoch 0's shuffle
    instead of epoch N's. Deriving the seed from the epoch index makes every
    epoch's minibatch order a pure function of (seed, epoch), so a resumed run
    sees exactly the same order an uninterrupted run would have.
    """
    generator = torch.Generator().manual_seed(cfg.seed * 1_000_003 + epoch)
    return DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        generator=generator,
    )


def _resolve_start_epoch(
    resume: bool,
    ckpt_path: Path,
    cfg_hash: str,
    train_hash: str,
    model_name: ModelName,
    variant: str,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    total_epochs: int,
) -> int:
    start_epoch = 0
    if resume:
        payload = load_head_checkpoint(
            ckpt_path, expected_config_hash=cfg_hash, expected_train_config_hash=train_hash
        )
        if payload["model_name"] != model_name or payload["variant"] != variant:
            raise ResumeStateError(
                f"checkpoint {ckpt_path} belongs to variant {payload['variant']!r}, not {variant!r}"
            )
        head.load_state_dict(payload["head_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if use_amp:
            # GradScaler's loss scale evolves step-by-step (growing on
            # successful steps, backing off on inf/nan gradients); restarting
            # it from its default initial state would make a resumed run's
            # optimizer steps diverge from an uninterrupted run's the moment
            # the scale trajectories differ. A checkpoint written before this
            # field existed (or under amp=False) has no usable scaler state —
            # resuming it under amp=True can't reproduce a continuous run, so
            # fail loudly instead of silently restarting the scale schedule.
            scaler_state = payload.get("scaler_state")
            if not scaler_state:
                raise ResumeStateError(
                    f"checkpoint {ckpt_path} has no GradScaler state (written before "
                    "this was tracked, or under amp=False); resuming it with amp=True "
                    "would not reproduce a continuous run — delete the checkpoint to "
                    "start over"
                )
            scaler.load_state_dict(scaler_state)
        start_epoch = int(payload["epochs_completed"])
        logger.info("resuming %s from epoch %d", variant, start_epoch)
    elif ckpt_path.is_file():
        raise ResumeStateError(
            f"a checkpoint already exists at {ckpt_path}; pass --resume to continue "
            "training or delete the directory to start over"
        )
    if start_epoch >= total_epochs:
        raise TrainingError(
            f"checkpoint already covers {start_epoch} epochs (config asks for "
            f"{total_epochs}); raise train.epochs to continue"
        )
    return start_epoch


def _record_epoch(
    *,
    ckpt_path: Path,
    history_path: Path,
    checkpoint_payload: dict[str, Any],
    history_row: dict[str, Any],
    variant: str,
    epochs_completed: int,
    total_epochs: int,
    train_loss: float,
    val_acc: float,
    val_f1: float,
    val_ece: float,
) -> None:
    # Write via a temp file + atomic replace: a crash mid-torch.save (OOM kill,
    # a disconnected Colab runtime, Ctrl+C) must not truncate the *existing*
    # last.pt — this file is overwritten every epoch, so an in-place write
    # failing partway would corrupt the previous epoch's already-good
    # checkpoint too, not just fail to add a new one.
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(checkpoint_payload, tmp_path)
    tmp_path.replace(ckpt_path)
    append_jsonl(history_path, [history_row])
    logger.info(
        "%s epoch %d/%d: loss=%.4f val_acc=%.4f val_f1=%.4f val_ece=%.4f",
        variant,
        epochs_completed,
        total_epochs,
        train_loss,
        val_acc,
        val_f1,
        val_ece,
    )


def train_model(
    cfg: AppConfig,
    model_name: ModelName,
    *,
    patched: bool = False,
    resume: bool = False,
    device: str | None = None,
) -> TrainResult:
    start = time.monotonic()
    seed_everything(cfg.seed)
    cfg_hash = config_hash(cfg)
    train_hash = train_config_hash(cfg)
    variant = variant_name(model_name, patched)
    train_cfg = cfg.train

    train_records, val_records = _load_split_records(cfg)
    num_classes = len({r.class_id for r in train_records} | {r.class_id for r in val_records})

    bundle: ModelBundle = build_model(
        model_name, num_classes=num_classes, pretrained=train_cfg.pretrained, head_only=True
    )
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = bundle.model.to(resolved_device)
    use_amp = train_cfg.amp and resolved_device.type == "cuda"
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    train_dataset, val_dataset = _build_datasets(cfg, bundle, train_records, val_records, patched)
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
    )
    head = head_module(model, model_name)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(resolved_device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = resolve_checkpoints_dir(cfg) / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CHECKPOINT_FILENAME
    out_dir = raw_train_dir(cfg, variant)
    history_path = out_dir / "history.jsonl"

    start_epoch = _resolve_start_epoch(
        resume,
        ckpt_path,
        cfg_hash,
        train_hash,
        model_name,
        variant,
        head,
        optimizer,
        scaler,
        use_amp,
        train_cfg.epochs,
    )

    val_acc = 0.0
    val_f1 = 0.0
    val_ece = 0.0
    for epoch in range(start_epoch, train_cfg.epochs):
        epoch_start = time.monotonic()
        train_loader = _epoch_train_loader(cfg, train_dataset, epoch)
        # head_only training freezes 100% of the backbone; the only trainable
        # module is a bare nn.Linear head, which has no train/eval-sensitive
        # behavior. Keeping the whole model in eval() disables the frozen
        # backbone's internal stochasticity (ConvNeXt's StochasticDepth, ViT's
        # attention/MLP Dropout) instead of injecting pointless noise into
        # features the head can't even learn from — and makes the forward
        # pass a pure function of the input, which --resume equivalence needs.
        model.eval()
        running_loss = 0.0
        batches = 0
        for batch_images, batch_targets in train_loader:
            images = batch_images.to(resolved_device)
            targets = batch_targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(resolved_device.type, enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            batches += 1
        train_loss = running_loss / max(batches, 1)
        val_acc, val_f1, val_ece = _evaluate(model, val_loader, resolved_device, num_classes)
        epochs_completed = epoch + 1
        _record_epoch(
            ckpt_path=ckpt_path,
            history_path=history_path,
            checkpoint_payload={
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_name": model_name,
                "variant": variant,
                "arch": bundle.arch,
                "num_classes": num_classes,
                "epochs_completed": epochs_completed,
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config_hash": cfg_hash,
                "train_config_hash": train_hash,
                "scaler_state": scaler.state_dict(),
            },
            history_row={
                "epoch": epochs_completed,
                "train_loss": round(train_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "val_macro_f1": round(val_f1, 6),
                "val_ece": round(val_ece, 6),
                "lr": train_cfg.lr,
                "epoch_seconds": round(time.monotonic() - epoch_start, 2),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            variant=variant,
            epochs_completed=epochs_completed,
            total_epochs=train_cfg.epochs,
            train_loss=train_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            val_ece=val_ece,
        )

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(resolved_device) / 2**20, 1)
        if resolved_device.type == "cuda"
        else None
    )
    device_name = (
        torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else "cpu"
    )
    elapsed = time.monotonic() - start
    atomic_write_json(
        out_dir / "final.json",
        {
            "schema_version": 1,
            "variant": variant,
            "model_name": model_name,
            "arch": bundle.arch,
            "patched": patched,
            "epochs": train_cfg.epochs,
            "pretrained": train_cfg.pretrained,
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "val_ece": round(val_ece, 6),
            "device": device_name,
            "amp": use_amp,
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_mb": peak_vram_mb,
            "torch_version": torch.__version__,
            "config_hash": cfg_hash,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return TrainResult(
        variant=variant,
        epochs_run_this_call=train_cfg.epochs - start_epoch,
        total_epochs_completed=train_cfg.epochs,
        final_val_accuracy=val_acc,
        final_val_macro_f1=val_f1,
        final_val_ece=val_ece,
        checkpoint_path=ckpt_path,
        device=device_name,
        elapsed_seconds=elapsed,
        peak_vram_mb=peak_vram_mb,
    )
