"""Run a tiny, isolated CUDA AMP checkpoint/resume equivalence canary.

The canary uses synthetic images and random (not pretrained) ConvNeXt weights.
It never reads or writes the committed full-scale evidence, datasets, or model
weights. It is evidence for the CUDA resume path only, not a benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path(".artifacts/cuda-resume-canary")
DEFAULT_LEASE_PATH = Path(tempfile.gettempdir()) / "codex-rtx4090-compute.lease"
CANARY_SCHEMA_VERSION = 1
GPU_METADATA_FIELD_COUNT = 2
IMAGE_SIZE = 64
TRAINVAL_PER_CLASS = 6
TEST_PER_CLASS = 3
SYNTHETIC_CLASSES = (
    ("Fakecat", 1, 1),
    ("Mockcat", 2, 1),
    ("fakedog", 3, 2),
    ("mockdog", 4, 2),
)
METRIC_FIELDS = ("epoch", "train_loss", "val_accuracy", "val_macro_f1", "val_ece", "lr")


class CanaryError(RuntimeError):
    """The canary cannot run safely or an equivalence requirement failed."""


def _run_command(*args: str) -> str:
    try:
        result = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise CanaryError(f"could not execute {args[0]!r}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CanaryError(f"{' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _compute_processes() -> list[str]:
    output = _run_command(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    if not output or "No running processes found" in output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _gpu_metadata() -> dict[str, str]:
    output = _run_command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",", 1)]
    if len(parts) != GPU_METADATA_FIELD_COUNT:
        raise CanaryError("could not parse nvidia-smi GPU metadata")
    return {"name": parts[0], "driver_version": parts[1]}


@contextmanager
def _gpu_lease(path: Path) -> Generator[str, None, None]:
    token = uuid.uuid4().hex
    payload = json.dumps(
        {"token": token, "pid": os.getpid(), "created_at": datetime.now(UTC).isoformat()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CanaryError(f"GPU lease already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        yield token
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                path.unlink()
        except (OSError, json.JSONDecodeError):
            pass


def _safe_run_directory(repo_root: Path, output_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise CanaryError("run id may contain only letters, digits, dot, underscore, and hyphen")
    allowed_root = (repo_root / DEFAULT_OUTPUT_ROOT).resolve()
    candidate_root = (
        (repo_root / output_root).resolve()
        if not output_root.is_absolute()
        else output_root.resolve()
    )
    try:
        candidate_root.relative_to(allowed_root)
    except ValueError as exc:
        raise CanaryError(f"output root must stay under {DEFAULT_OUTPUT_ROOT.as_posix()}") from exc
    run_dir = candidate_root / run_id
    if run_dir.exists():
        raise CanaryError(f"refusing to overwrite existing canary output: {run_dir}")
    return run_dir


def _write_synthetic_tree(data_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    from vision_xai.utils.seed import per_sample_rng

    base = data_dir / "oxford-iiit-pet"
    images = base / "images"
    annotations = base / "annotations"
    trimaps = annotations / "trimaps"
    images.mkdir(parents=True)
    trimaps.mkdir(parents=True)
    lines: dict[str, list[str]] = {"trainval": [], "test": []}
    for class_name, class_id, species_code in SYNTHETIC_CLASSES:
        for split, count, offset in (
            ("trainval", TRAINVAL_PER_CLASS, 100),
            ("test", TEST_PER_CLASS, 200),
        ):
            for index in range(count):
                sample_id = f"{class_name}_{offset + index}"
                rng = per_sample_rng(0, sample_id, "cuda-canary-image")
                base_color = rng.integers(0, 216, size=3).astype(np.int16)
                noise = rng.integers(0, 40, size=(IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.int16)
                image = np.clip(base_color[None, None, :] + noise, 0, 255).astype(np.uint8)
                Image.fromarray(image, mode="RGB").save(
                    images / f"{sample_id}.jpg", format="JPEG", quality=90
                )
                trimap = np.full((IMAGE_SIZE, IMAGE_SIZE), 2, dtype=np.uint8)
                trimap[15:49, 15:49] = 3
                trimap[16:48, 16:48] = 1
                Image.fromarray(trimap, mode="L").save(trimaps / f"{sample_id}.png")
                lines[split].append(f"{sample_id} {class_id} {species_code} {class_id}")
    for split, rows in lines.items():
        (annotations / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _config(data_dir: Path, results_dir: Path, *, epochs: int) -> Any:
    from vision_xai.config import AppConfig

    return AppConfig.model_validate(
        {
            "experiment_name": "cuda-resume-canary",
            "seed": 42,
            "paths": {"data_dir": str(data_dir), "results_dir": str(results_dir)},
            "train": {
                "epochs": epochs,
                "batch_size": 8,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "num_workers": 0,
                "amp": True,
                "pretrained": False,
            },
            "data": {
                "download": False,
                "image_size": 32,
                "resize_size": 40,
                "incremental_save_every": 5,
                "split": {"val_fraction": 0.2, "seed": 42},
            },
        }
    )


def _validate_lease(path: Path, token: str) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("CUDA worker cannot read the parent GPU lease") from exc
    if current.get("token") != token:
        raise CanaryError("CUDA worker does not own the active GPU lease")


@contextmanager
def _checkpoint_root(path: Path) -> Generator[None, None, None]:
    key = "VISION_XAI_CHECKPOINTS_DIR"
    previous = os.environ.get(key)
    os.environ[key] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _worker_train(spec_path: Path, lease_path: Path, lease_token: str) -> None:
    _validate_lease(lease_path, lease_token)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cfg = _config(
        Path(spec["data_dir"]),
        Path(spec["results_dir"]),
        epochs=int(spec["epochs"]),
    )
    from vision_xai.train.loop import train_model

    _configure_cuda_determinism()
    with _checkpoint_root(Path(spec["checkpoints_dir"])):
        result = train_model(cfg, "cnn", resume=bool(spec["resume"]), device="cuda")
    _atomic_json(
        Path(spec["result_path"]),
        {
            "checkpoint_path": str(result.checkpoint_path),
            "epochs_run_this_call": result.epochs_run_this_call,
            "total_epochs_completed": result.total_epochs_completed,
        },
    )


def _run_train_worker(
    run_dir: Path,
    phase: str,
    *,
    data_dir: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    epochs: int,
    resume: bool,
    lease_path: Path,
    lease_token: str,
) -> dict[str, Any]:
    spec_path = run_dir / f"{phase}-worker-spec.json"
    result_path = run_dir / f"{phase}-worker-result.json"
    _atomic_json(
        spec_path,
        {
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "epochs": epochs,
            "resume": resume,
            "result_path": str(result_path),
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-spec",
            str(spec_path),
            "--worker-lease-token",
            lease_token,
            "--lease-path",
            str(lease_path),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise CanaryError(f"CUDA worker {phase!r} failed: {detail}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CanaryError(f"CUDA worker {phase!r} returned a non-object result")
    return {str(key): value for key, value in payload.items()}


def _state_differences(reference: Any, resumed: Any, path: str = "state") -> list[str]:
    import torch

    differences: list[str]
    if isinstance(reference, torch.Tensor) and isinstance(resumed, torch.Tensor):
        if reference.dtype != resumed.dtype or reference.shape != resumed.shape:
            differences = [f"{path}: tensor metadata differs"]
        else:
            differences = (
                [] if torch.equal(reference, resumed) else [f"{path}: tensor values differ"]
            )
    elif isinstance(reference, dict) and isinstance(resumed, dict):
        if reference.keys() != resumed.keys():
            differences = [f"{path}: keys differ"]
        else:
            differences = []
            for key in reference:
                differences.extend(
                    _state_differences(reference[key], resumed[key], f"{path}.{key}")
                )
    elif isinstance(reference, (list, tuple)) and isinstance(resumed, type(reference)):
        if len(reference) != len(resumed):
            differences = [f"{path}: length differs"]
        else:
            differences = []
            for index, (left, right) in enumerate(zip(reference, resumed, strict=True)):
                differences.extend(_state_differences(left, right, f"{path}[{index}]"))
    else:
        differences = [] if reference == resumed else [f"{path}: values differ"]
    return differences


def _stable_history(path: Path) -> list[dict[str, Any]]:
    from vision_xai.utils.io import read_jsonl

    return [{key: row[key] for key in METRIC_FIELDS} for row in read_jsonl(path)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configure_cuda_determinism() -> None:
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_canary(run_dir: Path, lease_path: Path, lease_token: str) -> dict[str, Any]:
    import torch

    from vision_xai.data.prepare import prepare_data
    from vision_xai.paths import raw_train_dir
    from vision_xai.train.loop import load_head_checkpoint

    if not torch.cuda.is_available():
        raise CanaryError(f"CUDA is unavailable in torch {torch.__version__}")
    _configure_cuda_determinism()
    gpu = _gpu_metadata()
    device_name = torch.cuda.get_device_name(0)
    cudnn_backend: Any = torch.backends.cudnn

    reference_root = run_dir / "reference"
    resumed_root = run_dir / "resumed"
    for root in (reference_root, resumed_root):
        _write_synthetic_tree(root / "data")

    reference_cfg = _config(reference_root / "data", reference_root / "results", epochs=2)
    resumed_one_cfg = _config(resumed_root / "data", resumed_root / "results", epochs=1)
    resumed_two_cfg = _config(resumed_root / "data", resumed_root / "results", epochs=2)
    if not prepare_data(reference_cfg).completed or not prepare_data(resumed_one_cfg).completed:
        raise CanaryError("synthetic data preparation did not complete")

    reference_result = _run_train_worker(
        run_dir,
        "reference",
        data_dir=reference_root / "data",
        results_dir=reference_root / "results",
        checkpoints_dir=reference_root / "checkpoints",
        epochs=2,
        resume=False,
        lease_path=lease_path,
        lease_token=lease_token,
    )
    first_epoch = _run_train_worker(
        run_dir,
        "interrupted",
        data_dir=resumed_root / "data",
        results_dir=resumed_root / "results",
        checkpoints_dir=resumed_root / "checkpoints",
        epochs=1,
        resume=False,
        lease_path=lease_path,
        lease_token=lease_token,
    )
    resumed_result = _run_train_worker(
        run_dir,
        "resumed",
        data_dir=resumed_root / "data",
        results_dir=resumed_root / "results",
        checkpoints_dir=resumed_root / "checkpoints",
        epochs=2,
        resume=True,
        lease_path=lease_path,
        lease_token=lease_token,
    )

    reference_checkpoint_path = Path(reference_result["checkpoint_path"])
    resumed_checkpoint_path = Path(resumed_result["checkpoint_path"])
    reference_checkpoint = load_head_checkpoint(reference_checkpoint_path)
    resumed_checkpoint = load_head_checkpoint(resumed_checkpoint_path)
    head_differences = _state_differences(
        reference_checkpoint["head_state"], resumed_checkpoint["head_state"], "head_state"
    )
    optimizer_differences = _state_differences(
        reference_checkpoint["optimizer_state"],
        resumed_checkpoint["optimizer_state"],
        "optimizer_state",
    )
    scaler_differences = _state_differences(
        reference_checkpoint["scaler_state"],
        resumed_checkpoint["scaler_state"],
        "scaler_state",
    )
    reference_history = _stable_history(raw_train_dir(reference_cfg, "cnn") / "history.jsonl")
    resumed_history = _stable_history(raw_train_dir(resumed_two_cfg, "cnn") / "history.jsonl")
    metrics_exact = reference_history == resumed_history
    requirements_pass = not (
        head_differences or optimizer_differences or scaler_differences or not metrics_exact
    )

    evidence: dict[str, Any] = {
        "schema_version": CANARY_SCHEMA_VERSION,
        "kind": "cuda_amp_epoch_boundary_resume_canary",
        "status": "PASS" if requirements_pass else "FAIL",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "dataset": "deterministic synthetic Oxford-IIIT Pet layout",
            "model": "convnext_tiny random backbone, classifier head only",
            "epochs": 2,
            "interruption": (
                "supported epoch-boundary stop after epoch 1, then "
                "new-process-equivalent call with --resume"
            ),
            "not_full_scale": True,
        },
        "hardware": {"gpu": device_name, "driver_version": gpu["driver_version"]},
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": str(cudnn_backend.version()),
        },
        "configuration": {
            "seed": 42,
            "batch_size": 8,
            "amp": True,
            "pretrained": False,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "execution": {
            "external_compute_processes_at_start": 0,
            "process_isolation": "three sequential fresh Python worker processes",
            "reference_epochs_this_call": reference_result["epochs_run_this_call"],
            "interrupted_epochs_first_call": first_epoch["epochs_run_this_call"],
            "resumed_epochs_second_call": resumed_result["epochs_run_this_call"],
        },
        "comparisons": {
            "head_state_exact": not head_differences,
            "optimizer_state_exact": not optimizer_differences,
            "grad_scaler_state_exact": not scaler_differences,
            "stable_metrics_exact": metrics_exact,
            "checkpoint_file_sha256_equal": _sha256(reference_checkpoint_path)
            == _sha256(resumed_checkpoint_path),
            "scheduler_state": {
                "status": "not_applicable",
                "reason": "the training loop does not instantiate a learning-rate scheduler",
            },
            "differences": head_differences
            + optimizer_differences
            + scaler_differences
            + ([] if metrics_exact else ["stable per-epoch metrics differ"]),
        },
        "metrics": {"reference": reference_history, "resumed": resumed_history},
        "equivalence_policy": {
            "required": (
                "exact tensor/scalar equality for final head, optimizer, "
                "GradScaler, and stable metrics"
            ),
            "diagnostic_only": (
                "whole-checkpoint SHA-256; serialization bytes are not the semantic state contract"
            ),
        },
        "limitations": [
            "This tiny synthetic canary is not evidence that the full L4 experiment was resumed.",
            "Checkpointing occurs at epoch boundaries; no unsafe mid-batch crash is simulated.",
            "The training loop has no scheduler, so scheduler restoration is not applicable.",
            "Results apply to the recorded hardware and software stack, not every CUDA stack.",
        ],
    }
    _atomic_json(run_dir / "cuda-resume-canary.json", evidence)
    if not requirements_pass:
        raise CanaryError("CUDA resume equivalence requirements failed")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="gitignored root (must stay under .artifacts/cuda-resume-canary)",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="new output directory name",
    )
    parser.add_argument(
        "--lease-path",
        type=Path,
        default=DEFAULT_LEASE_PATH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-lease-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec is not None:
        if not args.worker_lease_token:
            sys.stderr.write("FAIL CUDA worker requires its parent lease token\n")
            return 1
        try:
            _worker_train(args.worker_spec, args.lease_path, args.worker_lease_token)
        except (CanaryError, OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"FAIL {exc}\n")
            return 1
        return 0
    repo_root = Path(__file__).resolve().parents[1]
    try:
        run_dir = _safe_run_directory(repo_root, args.output_root, args.run_id)
        if _compute_processes():
            raise CanaryError("another NVIDIA compute workload is active; refusing CUDA")
        with _gpu_lease(args.lease_path) as lease_token:
            if _compute_processes():
                raise CanaryError("an NVIDIA compute workload appeared after lease acquisition")
            run_dir.mkdir(parents=True)
            evidence = run_canary(run_dir, args.lease_path, lease_token)
    except CanaryError as exc:
        sys.stderr.write(f"FAIL {exc}\n")
        return 1
    sys.stdout.write(
        f"PASS CUDA resume canary: {evidence['comparisons']}\n"
        f"evidence: {(run_dir / 'cuda-resume-canary.json')}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
