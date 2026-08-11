from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from vision_xai.config import AppConfig, config_hash, load_config
from vision_xai.errors import ConfigError
from vision_xai.paths import DATA_DIR_ENV, resolve_data_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_raw() -> dict[str, Any]:
    return {
        "experiment_name": "unit",
        "seed": 7,
        "paths": {"data_dir": "data", "results_dir": "results"},
        "data": {"download": False},
    }


def _load_from_dict(tmp_path: Path, raw: dict[str, Any]) -> AppConfig:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


@pytest.mark.parametrize("name", ["smoke.yaml", "full.yaml"])
def test_repo_configs_parse(name: str) -> None:
    cfg = load_config(REPO_ROOT / "configs" / name)
    assert cfg.data.dataset == "oxford-iiit-pet"
    assert cfg.data.resize_size >= cfg.data.image_size
    if name == "smoke.yaml":
        assert cfg.experiment_name == "smoke"
        assert cfg.data.limit_per_class == 3
    else:
        assert cfg.experiment_name == "full"
        assert cfg.data.limit_per_class is None


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(REPO_ROOT / "configs" / "does-not-exist.yaml")


def test_invalid_enum_names_field(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["data"]["mask_policy"] = "bogus"
    with pytest.raises(ConfigError, match="mask_policy"):
        _load_from_dict(tmp_path, raw)


def test_out_of_range_fraction_names_field(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["data"]["split"] = {"val_fraction": 1.5}
    with pytest.raises(ConfigError, match="val_fraction"):
        _load_from_dict(tmp_path, raw)


def test_unknown_key_inside_data_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["data"]["not_a_real_option"] = 1
    with pytest.raises(ConfigError, match="not_a_real_option"):
        _load_from_dict(tmp_path, raw)


def test_resize_smaller_than_crop_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["data"]["resize_size"] = 128
    raw["data"]["image_size"] = 224
    with pytest.raises(ConfigError, match="resize_size"):
        _load_from_dict(tmp_path, raw)


def test_unknown_top_level_section_tolerated(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["train"] = {"epochs": 3}  # arrives in Stage 2; must not break Stage 1
    cfg = _load_from_dict(tmp_path, raw)
    assert cfg.experiment_name == "unit"


def test_config_hash_ignores_paths_but_not_data(tmp_path: Path) -> None:
    base = _load_from_dict(tmp_path, _valid_raw())

    moved_raw = _valid_raw()
    moved_raw["paths"] = {"data_dir": "elsewhere", "results_dir": "other"}
    moved = _load_from_dict(tmp_path, moved_raw)
    assert config_hash(base) == config_hash(moved)

    reseeded_raw = _valid_raw()
    reseeded_raw["seed"] = 8
    reseeded = _load_from_dict(tmp_path, reseeded_raw)
    assert config_hash(base) != config_hash(reseeded)


def test_data_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_from_dict(tmp_path, _valid_raw())
    override = tmp_path / "override"
    monkeypatch.setenv(DATA_DIR_ENV, str(override))
    assert resolve_data_dir(cfg) == override
