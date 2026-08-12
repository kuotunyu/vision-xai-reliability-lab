"""Fail-closed presentation model for committed public evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

ModelName = Literal["cnn", "vit"]

SUMMARY_PATH = Path("results/derived/summary.json")
CANARY_PATH = Path("release/cuda-resume-canary.json")
ATTRIBUTION_METHODS = {
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated Gradients",
    "occlusion": "Occlusion",
}
MODEL_LABELS: Mapping[ModelName, str] = {
    "cnn": "ConvNeXt-Tiny",
    "vit": "ViT-B/16",
}
MODEL_KEYS: tuple[ModelName, ...] = ("cnn", "vit")
EXPECTED_ATTRIBUTION_SAMPLES = 500


class EvidenceError(RuntimeError):
    """Raised when committed public evidence is absent or violates its contract."""


class _ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ModelEvidence:
    key: ModelName
    label: str
    val_accuracy: float
    val_macro_f1: float
    best_method: str
    best_pointing: float
    ig_randomization: float
    localization_figure: Path
    faithfulness_figure: Path
    spurious_figure: Path


@dataclass(frozen=True)
class EvidenceDashboard:
    center_pointing: float
    attribution_samples: int
    spurious_patch_energy_max: float
    models: Mapping[ModelName, ModelEvidence]
    canary_exact_states: tuple[str, ...]
    canary_scheduler_status: str
    canary_not_full_scale: bool

    def model(self, key: ModelName) -> ModelEvidence:
        return self.models[key]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _ContractError(field)
    return cast(dict[str, object], value)


def _at(payload: Mapping[str, object], *path: str) -> object:
    current: object = payload
    walked: list[str] = []
    for key in path:
        walked.append(key)
        mapping = _mapping(current, ".".join(walked[:-1]) or "root")
        if key not in mapping:
            raise _ContractError(".".join(walked))
        current = mapping[key]
    return current


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ContractError(field)
    return float(value)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ContractError(field)
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _ContractError(field)
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise _ContractError(field)
    return value


def _load_json(root: Path, relative: Path) -> Mapping[str, object]:
    try:
        payload: object = json.loads((root / relative).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvidenceError(f"missing public evidence: {relative.as_posix()}") from None
    except (OSError, json.JSONDecodeError):
        raise EvidenceError(f"invalid public evidence JSON: {relative.as_posix()}") from None
    try:
        return _mapping(payload, "root")
    except _ContractError:
        raise EvidenceError(
            f"public evidence must be a JSON object: {relative.as_posix()}"
        ) from None


def _require_figure(root: Path, filename: str) -> Path:
    relative = Path("assets/figures") / filename
    path = root / relative
    if not path.is_file():
        raise EvidenceError(f"missing public evidence: {relative.as_posix()}")
    return path


def _model_evidence(
    root: Path,
    summary: Mapping[str, object],
    key: ModelName,
    attribution_samples: int,
) -> ModelEvidence:
    localization = _mapping(_at(summary, "localization", key), f"localization.{key}")
    candidates: list[tuple[float, str]] = []
    for method_key, method_label in ATTRIBUTION_METHODS.items():
        if method_key not in localization:
            continue
        score = _number(
            _at(localization, method_key, "all", "pointing_rate", "mean"),
            f"localization.{key}.{method_key}.all.pointing_rate.mean",
        )
        sample_count = _integer(
            _at(localization, method_key, "all", "pointing_rate", "n"),
            f"localization.{key}.{method_key}.all.pointing_rate.n",
        )
        if sample_count != attribution_samples:
            raise _ContractError(f"localization.{key}.{method_key}.all.pointing_rate.n")
        candidates.append((score, method_label))
    if not candidates:
        raise _ContractError(f"localization.{key}.attribution_methods")
    best_pointing, best_method = max(candidates)

    train = _mapping(_at(summary, "train", key), f"train.{key}")
    ig_randomization = _number(
        _at(
            summary,
            "randomization",
            key,
            "integrated_gradients",
            "all",
            "abs_spearman",
            "mean",
        ),
        f"randomization.{key}.integrated_gradients.all.abs_spearman.mean",
    )
    return ModelEvidence(
        key=key,
        label=MODEL_LABELS[key],
        val_accuracy=_number(train.get("val_accuracy"), f"train.{key}.val_accuracy"),
        val_macro_f1=_number(train.get("val_macro_f1"), f"train.{key}.val_macro_f1"),
        best_method=best_method,
        best_pointing=best_pointing,
        ig_randomization=ig_randomization,
        localization_figure=_require_figure(root, f"localization_{key}.png"),
        faithfulness_figure=_require_figure(root, f"faithfulness_{key}.png"),
        spurious_figure=_require_figure(root, f"spurious_{key}_patched.png"),
    )


def _spurious_patch_energy_max(summary: Mapping[str, object]) -> float:
    spurious = _mapping(_at(summary, "spurious"), "spurious")
    means: list[float] = []
    for variant_payload in spurious.values():
        methods = _mapping(variant_payload, "spurious.variant")
        for method_payload in methods.values():
            test_variants = _mapping(method_payload, "spurious.variant.method")
            for test_payload in test_variants.values():
                value = _at(
                    _mapping(test_payload, "spurious.variant.method.test_variant"),
                    "patch_energy_patched_inputs",
                    "all",
                    "patch_energy",
                    "mean",
                )
                if value is not None:
                    means.append(_number(value, "spurious.patch_energy.mean"))
    if not means:
        raise _ContractError("spurious.patch_energy.mean")
    return max(means)


def _canary_contract(canary: Mapping[str, object]) -> tuple[tuple[str, ...], str, bool]:
    if _integer(canary.get("schema_version"), "schema_version") != 1:
        raise _ContractError("schema_version")
    if _string(canary.get("status"), "status") != "PASS":
        raise _ContractError("status")
    comparisons = _mapping(_at(canary, "comparisons"), "comparisons")
    exact_fields = (
        ("head_state_exact", "final head"),
        ("optimizer_state_exact", "optimizer"),
        ("grad_scaler_state_exact", "GradScaler"),
        ("stable_metrics_exact", "stable metrics"),
    )
    exact_states = tuple(
        label
        for field, label in exact_fields
        if _boolean(comparisons.get(field), f"comparisons.{field}")
    )
    if len(exact_states) != len(exact_fields):
        raise _ContractError("comparisons.exact_states")
    scheduler_status = _string(
        _at(comparisons, "scheduler_state", "status"),
        "comparisons.scheduler_state.status",
    )
    not_full_scale = _boolean(
        _at(canary, "scope", "not_full_scale"),
        "scope.not_full_scale",
    )
    if not not_full_scale:
        raise _ContractError("scope.not_full_scale")
    return exact_states, scheduler_status, not_full_scale


def load_evidence_dashboard(root: Path) -> EvidenceDashboard:
    """Load only the committed aggregate evidence required by the public UI."""
    summary = _load_json(root, SUMMARY_PATH)
    canary = _load_json(root, CANARY_PATH)
    try:
        if _integer(summary.get("schema_version"), "schema_version") != 1:
            raise _ContractError("schema_version")
        if _string(summary.get("experiment"), "experiment") != "full":
            raise EvidenceError("results/derived/summary.json is not canonical full evidence")
        attribution_samples = _integer(
            _at(summary, "localization", "cnn", "center", "all", "pointing_rate", "n"),
            "localization.cnn.center.all.pointing_rate.n",
        )
        if attribution_samples != EXPECTED_ATTRIBUTION_SAMPLES:
            raise _ContractError("localization.cnn.center.all.pointing_rate.n")
        center_pointing = _number(
            _at(summary, "localization", "cnn", "center", "all", "pointing_rate", "mean"),
            "localization.cnn.center.all.pointing_rate.mean",
        )
        vit_center = _number(
            _at(summary, "localization", "vit", "center", "all", "pointing_rate", "mean"),
            "localization.vit.center.all.pointing_rate.mean",
        )
        vit_center_n = _integer(
            _at(summary, "localization", "vit", "center", "all", "pointing_rate", "n"),
            "localization.vit.center.all.pointing_rate.n",
        )
        if vit_center != center_pointing or vit_center_n != attribution_samples:
            raise _ContractError("localization.vit.center.all.pointing_rate")

        models = {
            key: _model_evidence(root, summary, key, attribution_samples) for key in MODEL_KEYS
        }
        exact_states, scheduler_status, not_full_scale = _canary_contract(canary)
        return EvidenceDashboard(
            center_pointing=center_pointing,
            attribution_samples=attribution_samples,
            spurious_patch_energy_max=_spurious_patch_energy_max(summary),
            models=MappingProxyType(models),
            canary_exact_states=exact_states,
            canary_scheduler_status=scheduler_status,
            canary_not_full_scale=not_full_scale,
        )
    except _ContractError:
        raise EvidenceError(
            "results/derived/summary.json or release/cuda-resume-canary.json "
            "does not match the required public evidence contract"
        ) from None
