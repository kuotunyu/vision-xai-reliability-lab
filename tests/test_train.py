from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from conftest import ConfigFactory
from vision_xai.config import AppConfig, config_hash, train_config_hash
from vision_xai.data.prepare import prepare_data
from vision_xai.errors import ResumeStateError, TrainingError
from vision_xai.paths import raw_train_dir, resolve_checkpoints_dir
from vision_xai.train.loop import (
    CHECKPOINT_SCHEMA_VERSION,
    _resolve_start_epoch,
    load_head_checkpoint,
    train_model,
)
from vision_xai.utils.io import read_jsonl


def _train_config(config_factory: ConfigFactory, data_dir: Path, epochs: int = 1) -> AppConfig:
    return config_factory(
        data_dir,
        top_level={
            "train": {
                "epochs": epochs,
                "batch_size": 8,
                "num_workers": 0,
                "amp": False,
                "pretrained": False,  # never download weights in tests
            }
        },
    )


@pytest.fixture()
def prepared_config(
    synthetic_data_dir: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AppConfig:
    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts"))
    cfg = _train_config(config_factory, synthetic_data_dir)
    assert prepare_data(cfg).completed
    return cfg


def test_train_produces_checkpoint_history_and_final(prepared_config: AppConfig) -> None:
    result = train_model(prepared_config, "cnn")
    assert result.variant == "cnn"
    assert result.device == "cpu"
    assert result.peak_vram_mb is None  # never fabricated on CPU
    assert 0.0 <= result.final_val_accuracy <= 1.0
    assert 0.0 <= result.final_val_ece <= 1.0
    assert result.checkpoint_path.is_file()

    history = read_jsonl(raw_train_dir(prepared_config, "cnn") / "history.jsonl")
    assert len(history) == 1
    assert set(history[0]) >= {"epoch", "train_loss", "val_accuracy", "val_macro_f1", "val_ece"}

    final_path = raw_train_dir(prepared_config, "cnn") / "final.json"
    assert final_path.is_file()
    assert 0.0 <= json.loads(final_path.read_text(encoding="utf-8"))["val_ece"] <= 1.0

    payload = load_head_checkpoint(result.checkpoint_path)
    assert payload["variant"] == "cnn"
    assert payload["epochs_completed"] == 1
    # amp=False in this config -> GradScaler is disabled -> its state_dict() is
    # {} by torch's own convention, but the key must still be present so a
    # resumed AMP run can distinguish "no scaler state" from "not tracked yet".
    assert payload["scaler_state"] == {}


def test_fresh_run_refuses_existing_checkpoint(prepared_config: AppConfig) -> None:
    train_model(prepared_config, "cnn")
    with pytest.raises(ResumeStateError, match="already exists"):
        train_model(prepared_config, "cnn")


def test_resume_extends_epochs(
    prepared_config: AppConfig,
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
) -> None:
    train_model(prepared_config, "cnn")
    extended = _train_config(config_factory, synthetic_data_dir, epochs=2)
    result = train_model(extended, "cnn", resume=True)
    assert result.epochs_run_this_call == 1
    assert result.total_epochs_completed == 2
    history = read_jsonl(raw_train_dir(extended, "cnn") / "history.jsonl")
    assert [row["epoch"] for row in history] == [1, 2]


def test_resume_when_already_complete_raises(prepared_config: AppConfig) -> None:
    train_model(prepared_config, "cnn")
    with pytest.raises(TrainingError, match="already covers"):
        train_model(prepared_config, "cnn", resume=True)


def test_resume_rejects_changed_seed(
    prepared_config: AppConfig,
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
) -> None:
    train_model(prepared_config, "cnn")
    changed: dict[str, Any] = {
        "train": {"epochs": 2, "batch_size": 8, "amp": False, "pretrained": False}
    }
    reseeded = config_factory(synthetic_data_dir, top_level={**changed, "seed": 7})
    with pytest.raises(ResumeStateError, match="config changed"):
        train_model(reseeded, "cnn", resume=True)


def test_resume_rejects_changed_data_config(
    prepared_config: AppConfig,
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
) -> None:
    """A genuine `data` section change (not just seed) must also be rejected."""
    train_model(prepared_config, "cnn")
    changed_data = config_factory(
        synthetic_data_dir,
        image_size=36,  # differs from prepared_config's 32 (min allowed is 32)
        resize_size=40,
        top_level={"train": {"epochs": 2, "batch_size": 8, "amp": False, "pretrained": False}},
    )
    with pytest.raises(ResumeStateError, match="config changed"):
        train_model(changed_data, "cnn", resume=True)


def test_resume_rejects_changed_train_hyperparameters(
    prepared_config: AppConfig,
    config_factory: ConfigFactory,
    synthetic_data_dir: Path,
) -> None:
    """Changing lr (or batch_size/weight_decay/amp/pretrained) across a resume
    would not reproduce an uninterrupted run, so it must be rejected even
    though config_hash (experiment/seed/data) is unchanged."""
    train_model(prepared_config, "cnn")
    different_lr = config_factory(
        synthetic_data_dir,
        top_level={
            "train": {
                "epochs": 2,
                "batch_size": 8,
                "amp": False,
                "pretrained": False,
                "lr": 5e-4,
            }
        },
    )
    with pytest.raises(ResumeStateError, match="hyperparameters"):
        train_model(different_lr, "cnn", resume=True)


def test_patched_variant_trains(prepared_config: AppConfig) -> None:
    result = train_model(prepared_config, "cnn", patched=True)
    assert result.variant == "cnn_patched"
    assert (resolve_checkpoints_dir(prepared_config) / "cnn_patched" / "last.pt").is_file()


def test_train_without_manifest_raises(config_factory: ConfigFactory, tmp_path: Path) -> None:
    cfg = _train_config(config_factory, tmp_path / "empty")
    with pytest.raises(TrainingError, match="manifest not found"):
        train_model(cfg, "cnn")


def test_resume_matches_an_uninterrupted_run(
    tmp_path: Path,
    config_factory: ConfigFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1-epoch-then-resumed-to-2 run must reproduce a continuous 2-epoch run
    exactly: same per-epoch history and bit-identical final head weights.

    Exercises the fix for three sources of resume nondeterminism that a
    brand-new `train_model()` call (a resume) would otherwise reintroduce:
    the DataLoader shuffle order (now seeded per-epoch, not per-process), the
    frozen backbone's internal stochasticity (now run in eval mode, since
    head-only training has no trainable dropout/stochastic-depth), and the
    RandomHorizontalFlip augmentation (now decided deterministically per
    sample_id instead of drawn from torch's global RNG).
    """
    from conftest import make_synthetic_pet_tree

    tree_continuous = tmp_path / "continuous"
    tree_resumed = tmp_path / "resumed"
    make_synthetic_pet_tree(tree_continuous)
    make_synthetic_pet_tree(tree_resumed)

    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts_continuous"))
    cfg_continuous = _train_config(config_factory, tree_continuous, epochs=2)
    assert prepare_data(cfg_continuous).completed
    continuous_result = train_model(cfg_continuous, "cnn")

    monkeypatch.setenv("VISION_XAI_CHECKPOINTS_DIR", str(tmp_path / "ckpts_resumed"))
    cfg_one_epoch = _train_config(config_factory, tree_resumed, epochs=1)
    assert prepare_data(cfg_one_epoch).completed
    train_model(cfg_one_epoch, "cnn")
    cfg_two_epochs = _train_config(config_factory, tree_resumed, epochs=2)
    resumed_result = train_model(cfg_two_epochs, "cnn", resume=True)

    continuous_history = read_jsonl(raw_train_dir(cfg_continuous, "cnn") / "history.jsonl")
    resumed_history = read_jsonl(raw_train_dir(cfg_two_epochs, "cnn") / "history.jsonl")
    assert [row["train_loss"] for row in continuous_history] == [
        row["train_loss"] for row in resumed_history
    ]
    assert [row["val_accuracy"] for row in continuous_history] == [
        row["val_accuracy"] for row in resumed_history
    ]

    continuous_ckpt = load_head_checkpoint(continuous_result.checkpoint_path)
    resumed_ckpt = load_head_checkpoint(resumed_result.checkpoint_path)
    for key in continuous_ckpt["head_state"]:
        assert torch.equal(continuous_ckpt["head_state"][key], resumed_ckpt["head_state"][key])


def _amp_checkpoint_payload(
    cfg: AppConfig, *, scaler_state: dict[str, Any] | None
) -> dict[str, Any]:
    """A minimal checkpoint payload for exercising _resolve_start_epoch's
    GradScaler handling directly, without a real CUDA device: only the keys
    that function actually reads are populated."""
    head = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(head.parameters())
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_name": "cnn",
        "variant": "cnn",
        "config_hash": config_hash(cfg),
        "train_config_hash": train_config_hash(cfg),
        "epochs_completed": 1,
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scaler_state is not None:
        payload["scaler_state"] = scaler_state
    return payload


def test_resume_restores_gradscaler_state_when_amp_enabled(
    config_factory: ConfigFactory, tmp_path: Path
) -> None:
    """GradScaler's loss scale must actually be restored on --resume, not left
    at its default — otherwise a resumed AMP run's optimizer steps diverge
    from an uninterrupted run's the moment the scale trajectories differ."""
    cfg = config_factory(tmp_path / "data", top_level={"train": {"amp": True}})
    saved_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=512.0)
    ckpt_path = tmp_path / "amp.pt"
    torch.save(_amp_checkpoint_payload(cfg, scaler_state=saved_scaler.state_dict()), ckpt_path)

    fresh_head = torch.nn.Linear(2, 2)
    fresh_optimizer = torch.optim.AdamW(fresh_head.parameters())
    fresh_scaler = torch.amp.GradScaler("cpu", enabled=True)  # default init_scale != 512.0
    assert fresh_scaler.get_scale() != 512.0

    start_epoch = _resolve_start_epoch(
        True,
        ckpt_path,
        config_hash(cfg),
        train_config_hash(cfg),
        "cnn",
        "cnn",
        fresh_head,
        fresh_optimizer,
        fresh_scaler,
        True,
        2,
    )
    assert start_epoch == 1
    assert fresh_scaler.get_scale() == 512.0


def test_resume_with_amp_rejects_checkpoint_missing_scaler_state(
    config_factory: ConfigFactory, tmp_path: Path
) -> None:
    """A checkpoint written before GradScaler state was tracked (or under
    amp=False) cannot reproduce a continuous AMP run if resumed under
    amp=True — this must fail loudly rather than silently restart the scale
    schedule from its default."""
    cfg = config_factory(tmp_path / "data", top_level={"train": {"amp": True}})
    ckpt_path = tmp_path / "amp.pt"
    torch.save(_amp_checkpoint_payload(cfg, scaler_state=None), ckpt_path)

    head = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(head.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=True)

    with pytest.raises(ResumeStateError, match="GradScaler"):
        _resolve_start_epoch(
            True,
            ckpt_path,
            config_hash(cfg),
            train_config_hash(cfg),
            "cnn",
            "cnn",
            head,
            optimizer,
            scaler,
            True,
            2,
        )
