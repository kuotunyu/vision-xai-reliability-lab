"""Directory resolution. Relative config paths resolve against the working directory;
``VISION_XAI_DATA_DIR`` overrides the data directory (rule: no hardcoded absolute paths)."""

from __future__ import annotations

import os
from pathlib import Path

from vision_xai.config import AppConfig

DATA_DIR_ENV = "VISION_XAI_DATA_DIR"
CHECKPOINTS_DIR_ENV = "VISION_XAI_CHECKPOINTS_DIR"


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def resolve_data_dir(cfg: AppConfig) -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    return _absolute(Path(override) if override else cfg.paths.data_dir)


def resolve_results_dir(cfg: AppConfig) -> Path:
    return _absolute(cfg.paths.results_dir)


def manifests_dir(cfg: AppConfig) -> Path:
    return resolve_data_dir(cfg) / "manifests" / cfg.experiment_name


def raw_results_dir(cfg: AppConfig) -> Path:
    return resolve_results_dir(cfg) / "raw" / "data_prepare" / cfg.experiment_name


def resolve_checkpoints_dir(cfg: AppConfig) -> Path:
    override = os.environ.get(CHECKPOINTS_DIR_ENV)
    base = _absolute(Path(override) if override else Path("checkpoints"))
    return base / cfg.experiment_name


def raw_train_dir(cfg: AppConfig, variant: str) -> Path:
    return resolve_results_dir(cfg) / "raw" / "train" / cfg.experiment_name / variant


def raw_explanations_dir(cfg: AppConfig, variant: str, method: str) -> Path:
    return (
        resolve_results_dir(cfg) / "raw" / "explanations" / cfg.experiment_name / variant / method
    )


def raw_eval_dir(cfg: AppConfig) -> Path:
    return resolve_results_dir(cfg) / "raw" / "eval" / cfg.experiment_name


def derived_dir(cfg: AppConfig) -> Path:
    return resolve_results_dir(cfg) / "derived"
