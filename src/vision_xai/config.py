"""Config models (pydantic v2) — the single source of truth for the YAML schema."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from vision_xai.errors import ConfigError

logger = logging.getLogger(__name__)

PatchCorner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
PatchVariantName = Literal["correlated", "no_patch", "counter_correlated"]
MaskPolicy = Literal["foreground", "foreground_and_boundary"]


def _default_test_variants() -> list[PatchVariantName]:
    return ["correlated", "no_patch", "counter_correlated"]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42


class SpuriousPatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    corner: PatchCorner = "bottom_right"
    size_fraction: float = Field(default=0.15, gt=0.0, lt=0.5)
    margin_fraction: float = Field(default=0.02, ge=0.0, lt=0.25)
    pattern: Literal["solid", "checker"] = "checker"
    color: tuple[int, int, int] = (255, 0, 255)
    color2: tuple[int, int, int] = (0, 0, 0)
    target_group: Literal["cats", "dogs"] | list[int] = "cats"
    p_patch_in_group: float = Field(default=0.9, ge=0.0, le=1.0)
    p_patch_out_group: float = Field(default=0.1, ge=0.0, le=1.0)
    test_variants: list[PatchVariantName] = Field(default_factory=_default_test_variants)

    @field_validator("color", "color2")
    @classmethod
    def _rgb_in_range(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        max_channel = 255
        if any(not 0 <= channel <= max_channel for channel in value):
            raise ValueError("RGB components must be in [0, 255]")
        return value

    @field_validator("target_group")
    @classmethod
    def _group_not_empty(
        cls, value: Literal["cats", "dogs"] | list[int]
    ) -> Literal["cats", "dogs"] | list[int]:
        if isinstance(value, list):
            if not value:
                raise ValueError("target_group class-id list must not be empty")
            if any(class_id < 0 for class_id in value):
                raise ValueError("target_group class ids must be >= 0")
        return value


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Literal["oxford-iiit-pet"] = "oxford-iiit-pet"
    download: bool = True
    limit_per_class: int | None = Field(default=None, ge=1)
    image_size: int = Field(default=224, ge=32)
    resize_size: int = Field(default=256, ge=32)
    mask_policy: MaskPolicy = "foreground"
    incremental_save_every: int = Field(default=32, ge=1)
    split: SplitConfig = Field(default_factory=SplitConfig)
    spurious_patch: SpuriousPatchConfig = Field(default_factory=SpuriousPatchConfig)

    @model_validator(mode="after")
    def _resize_covers_crop(self) -> DataConfig:
        if self.resize_size < self.image_size:
            raise ValueError(
                f"resize_size ({self.resize_size}) must be >= image_size ({self.image_size})"
            )
        return self


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    amp: bool = True  # effective only on CUDA
    pretrained: bool = True  # tests set False so CI never downloads weights


class ExplainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: Literal["val", "test"] = "test"
    num_samples: int | None = Field(default=None, ge=1)  # None = the whole split
    batch_size: int = Field(default=8, ge=1)
    ig_steps: int = Field(default=16, ge=2)
    occlusion_window: int = Field(default=32, ge=4)
    occlusion_stride: int = Field(default=16, ge=1)
    incremental_save_every: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _stride_covers_window(self) -> ExplainConfig:
        if self.occlusion_stride > self.occlusion_window:
            raise ValueError(
                f"occlusion_stride ({self.occlusion_stride}) must be <= "
                f"occlusion_window ({self.occlusion_window}) to cover the image"
            )
        return self


def _default_topk_fractions() -> list[float]:
    return [0.05, 0.1, 0.2]


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deletion_steps: int = Field(default=20, ge=2)
    # Pre-registered fractions reported in full — never tuned on the test set.
    topk_fractions: list[float] = Field(default_factory=_default_topk_fractions)
    randomization_samples: int = Field(default=8, ge=1)
    consistency_samples: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)

    @field_validator("topk_fractions")
    @classmethod
    def _fractions_in_range(cls, value: list[float]) -> list[float]:
        if not value or any(not 0.0 < fraction <= 1.0 for fraction in value):
            raise ValueError("topk_fractions must be non-empty values in (0, 1]")
        return value


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"  # Docker overrides with --host 0.0.0.0
    port: int = Field(default=8000, ge=1, le=65535)
    default_model: Literal["cnn", "vit"] = "cnn"
    top_k: int = Field(default=5, ge=1, le=37)


class AppConfig(BaseModel):
    # Unknown top-level sections are tolerated (sections for future stages);
    # load_config logs a warning for each. Known sections stay strict.
    model_config = ConfigDict(extra="allow")

    experiment_name: str = Field(min_length=1)
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)


def load_config(path: Path) -> AppConfig:
    """Load and validate a YAML config; raises :class:`ConfigError` with field paths."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}: {path}")
    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"invalid config {path}: {details}") from exc
    unknown = set(raw) - set(AppConfig.model_fields)
    for key in sorted(unknown):
        logger.warning("config %s: unknown top-level section %r ignored in this stage", path, key)
    return cfg


def config_hash(cfg: AppConfig) -> str:
    """Stable 12-hex-char hash of the pipeline-relevant config.

    ``paths`` is deliberately excluded: relocating data/results directories must not
    invalidate resume checkpoints or fingerprints. ``train``/``explain``/``evaluate``/
    ``serve`` are also excluded here since this hash backs dataset resume/fingerprint
    checks; see :func:`train_config_hash` for the training-specific counterpart.
    """
    payload = cfg.model_dump(mode="json", include={"experiment_name", "seed", "data"})
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def train_config_hash(cfg: AppConfig) -> str:
    """Stable 12-hex-char hash guarding ``train --resume``.

    Combines :func:`config_hash` (experiment/seed/data) with every ``train``
    hyperparameter that must not silently change across a resume (batch size,
    learning rate, weight decay, AMP, pretrained) — resuming with a different
    learning rate, for instance, would not reproduce an uninterrupted run.
    ``epochs`` is deliberately excluded: raising it to continue training for
    longer is an intended, documented resume workflow.
    """
    train_payload = cfg.train.model_dump(mode="json", exclude={"epochs"})
    combined = f"{config_hash(cfg)}:{json.dumps(train_payload, sort_keys=True)}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]
