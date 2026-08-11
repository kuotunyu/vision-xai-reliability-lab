from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import ConfigFactory
from vision_xai.config import AppConfig, ExplainConfig
from vision_xai.data.prepare import prepare_data
from vision_xai.errors import ExplainError, TrainingError
from vision_xai.explain.base import check_compatibility, explain
from vision_xai.explain.baselines import baseline_attributions
from vision_xai.explain.runner import run_explanations
from vision_xai.models.factory import build_model
from vision_xai.train.loop import train_model
from vision_xai.utils.io import read_jsonl

SMALL = ExplainConfig.model_validate(
    {"ig_steps": 4, "occlusion_window": 16, "occlusion_stride": 8, "batch_size": 4}
)


def test_compatibility_rules() -> None:
    with pytest.raises(ExplainError, match="not compatible"):
        check_compatibility("gradcam", "vit")
    with pytest.raises(ExplainError, match="unknown method"):
        check_compatibility("shapley", "cnn")
    check_compatibility("integrated_gradients", "vit")  # must not raise


@pytest.mark.parametrize("method", ["gradcam", "integrated_gradients", "occlusion"])
def test_real_methods_output_shape(method: str) -> None:
    bundle = build_model("cnn", pretrained=False)
    images = torch.rand(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    batch = explain(bundle.model, images, targets, method, SMALL, "cnn")
    assert batch.attributions.shape == (2, 32, 32)
    assert torch.isfinite(batch.attributions).all()


def test_ig_caps_captum_internal_batch_size() -> None:
    """IG must bound captum's internal expansion to the incoming batch size.

    Without ``internal_batch_size``, captum runs all ``batch * n_steps``
    expanded examples as a single forward/backward — at full-config scale
    (16 x 32) that is 512 images' worth of activations *with* input gradients,
    which OOMs a 22 GB GPU. This is memory-only: the maths is unchanged.
    """
    from captum.attr import IntegratedGradients

    from vision_xai.explain.integrated_gradients import ig_attributions

    seen: dict[str, object] = {}
    original = IntegratedGradients.attribute

    def spy(self: IntegratedGradients, *args: object, **kwargs: object) -> torch.Tensor:
        seen.update(kwargs)
        return original(self, *args, **kwargs)  # type: ignore[no-any-return]

    bundle = build_model("cnn", pretrained=False)
    images = torch.rand(3, 3, 32, 32)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(IntegratedGradients, "attribute", spy)
        ig_attributions(bundle.model, images, torch.tensor([0, 1, 2]), n_steps=4)

    assert seen["internal_batch_size"] == images.shape[0]


def test_baselines_are_deterministic_and_distinct() -> None:
    first = baseline_attributions("random", 2, 16, 16, ["a", "b"], seed=42)
    second = baseline_attributions("random", 2, 16, 16, ["a", "b"], seed=42)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])  # different sample ids differ

    uniform = baseline_attributions("uniform", 1, 16, 16, ["a"], seed=42)[0]
    assert torch.allclose(uniform, uniform.mean())

    center = baseline_attributions("center", 1, 17, 17, ["a"], seed=42)[0]
    assert center.argmax() == (17 * 17) // 2  # peak at the exact center


def test_baseline_via_explain_requires_sample_ids() -> None:
    bundle = build_model("cnn", pretrained=False)
    images = torch.rand(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    with pytest.raises(ExplainError, match="sample_id"):
        explain(bundle.model, images, targets, "random", SMALL, "cnn")
    batch = explain(bundle.model, images, targets, "random", SMALL, "cnn", sample_ids=["x", "y"])
    assert batch.attributions.shape == (2, 32, 32)


@pytest.fixture()
def trained_config(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AppConfig:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts"))
    cfg = config_factory(
        synthetic_data_dir,
        top_level={
            "train": {"epochs": 1, "batch_size": 8, "amp": False, "pretrained": False},
            "explain": {
                "split": "test",
                "num_samples": 6,
                "batch_size": 4,
                "ig_steps": 4,
                "occlusion_window": 16,
                "occlusion_stride": 8,
            },
        },
    )
    assert prepare_data(cfg).completed
    train_model(cfg, "cnn")
    return cfg


def test_runner_end_to_end_and_resume(trained_config: AppConfig) -> None:
    result = run_explanations(trained_config, "cnn", "integrated_gradients")
    assert result.num_processed_this_run == 6
    meta = read_jsonl(result.out_dir / "meta.jsonl")
    assert len(meta) == 6
    assert {"sample_id", "target", "prediction", "correct", "latency_s"} <= set(meta[0])
    npz_files = sorted((result.out_dir / "attr").glob("*.npz"))
    assert len(npz_files) == 6
    loaded = np.load(npz_files[0])["attribution"]
    assert loaded.shape == (32, 32)
    assert loaded.dtype == np.float16
    assert (result.out_dir / "run.json").is_file()

    with pytest.raises(ExplainError, match="already exist"):
        run_explanations(trained_config, "cnn", "integrated_gradients")
    resumed = run_explanations(trained_config, "cnn", "integrated_gradients", resume=True)
    assert resumed.num_processed_this_run == 0
    assert resumed.num_total == 6


def test_runner_resume_merges_a_genuine_partial_run(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unlike test_runner_end_to_end_and_resume (which only resumes an
    already-100%-done run), this drives a real partial-then-resume: 3 of 6
    samples explained, then the config's num_samples is raised to 6 and
    --resume is used to backfill exactly the missing 3, mirroring how
    tests/test_prepare_resume.py exercises the data pipeline's --resume."""
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts"))
    train_top_level = {"train": {"epochs": 1, "batch_size": 8, "amp": False, "pretrained": False}}
    small_cfg = config_factory(
        synthetic_data_dir,
        top_level={
            **train_top_level,
            "explain": {
                "split": "test",
                "num_samples": 3,
                "batch_size": 4,
                "ig_steps": 4,
                "occlusion_window": 16,
                "occlusion_stride": 8,
            },
        },
    )
    assert prepare_data(small_cfg).completed
    train_model(small_cfg, "cnn")

    partial = run_explanations(small_cfg, "cnn", "integrated_gradients")
    assert partial.num_processed_this_run == 3
    assert partial.num_total == 3
    partial_ids = {row["sample_id"] for row in read_jsonl(partial.out_dir / "meta.jsonl")}

    full_cfg = config_factory(
        synthetic_data_dir,
        top_level={
            **train_top_level,
            "explain": {
                "split": "test",
                "num_samples": 6,
                "batch_size": 4,
                "ig_steps": 4,
                "occlusion_window": 16,
                "occlusion_stride": 8,
            },
        },
    )
    resumed = run_explanations(full_cfg, "cnn", "integrated_gradients", resume=True)
    assert resumed.num_processed_this_run == 3  # only the 3 newly-added samples
    assert resumed.num_total == 6

    meta = read_jsonl(resumed.out_dir / "meta.jsonl")
    sample_ids = [row["sample_id"] for row in meta]
    assert len(sample_ids) == 6
    assert len(set(sample_ids)) == 6  # no duplicates from the first, partial run
    assert partial_ids <= set(sample_ids)  # the original 3 are preserved, not redone
    npz_files = sorted((resumed.out_dir / "attr").glob("*.npz"))
    assert {path.stem for path in npz_files} == set(sample_ids)


def test_runner_with_patched_test_variant(trained_config: AppConfig) -> None:
    result = run_explanations(trained_config, "cnn", "random", test_variant="correlated")
    assert result.out_dir.name == "random__correlated"
    assert result.num_processed_this_run == 6


def test_runner_requires_checkpoint(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "none"))
    cfg = config_factory(
        synthetic_data_dir,
        top_level={"train": {"pretrained": False}},
    )
    assert prepare_data(cfg).completed
    with pytest.raises(TrainingError, match="checkpoint not found"):
        run_explanations(cfg, "cnn", "integrated_gradients")
