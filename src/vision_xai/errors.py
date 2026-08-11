"""Exception hierarchy. The CLI catches :class:`VisionXAIError` for clean exits."""

from __future__ import annotations


class VisionXAIError(Exception):
    """Base class for all expected, user-facing failures."""


class ConfigError(VisionXAIError):
    """A config file is missing, unparsable, or fails schema validation."""


class DataPreparationError(VisionXAIError):
    """The data-preparation pipeline hit an unrecoverable problem."""


class ResumeStateError(VisionXAIError):
    """A resume checkpoint exists but does not match the current run."""


class DatasetIntegrityError(VisionXAIError):
    """The on-disk dataset contradicts its documented structure or semantics."""


class TrainingError(VisionXAIError):
    """Training cannot start or continue (missing manifest, bad checkpoint, ...)."""


class ExplainError(VisionXAIError):
    """An explanation method cannot run (incompatible arch, missing checkpoint, ...)."""


class EvaluationError(VisionXAIError):
    """Evaluation cannot run (no explanation runs found, missing artifacts, ...)."""


class ReportError(VisionXAIError):
    """Report generation cannot run (no eval results, missing README markers, ...)."""
