from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click import unstyle
from typer.testing import CliRunner

from vision_xai.cli import app
from vision_xai.config import load_config
from vision_xai.paths import raw_results_dir

runner = CliRunner()


def test_root_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in unstyle(result.output)


def test_data_prepare_help_exits_zero() -> None:
    result = runner.invoke(app, ["data", "prepare", "--help"])
    assert result.exit_code == 0
    assert "--resume" in unstyle(result.output)


def test_self_check_passes_on_cpu() -> None:
    result = runner.invoke(app, ["self-check"])
    assert result.exit_code == 0, result.output
    assert "self-check OK" in result.output


@pytest.mark.parametrize("command", ["train", "explain", "evaluate", "report"])
def test_missing_required_config_option_exits_2(command: str) -> None:
    """These commands have no default --config; Click/Typer reports a usage
    error (exit 2) before our code runs. `serve` is excluded: it defaults
    --config to configs/smoke.yaml and would actually start a blocking
    uvicorn server if invoked bare — see test_serve_help_exits_zero below."""
    result = runner.invoke(app, [command])
    assert result.exit_code == 2


def test_serve_help_exits_zero_without_starting_a_server() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--config" in unstyle(result.output)


def test_serve_missing_config_file_exits_1_without_starting_a_server() -> None:
    """Config loading fails before uvicorn.run() is ever reached."""
    result = runner.invoke(app, ["serve", "--config", "does-not-exist.yaml"])
    assert result.exit_code == 1


def test_missing_config_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "prepare", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1


def test_data_prepare_end_to_end(synthetic_data_dir: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "cli-e2e",
                "seed": 42,
                "paths": {
                    "data_dir": str(synthetic_data_dir),
                    "results_dir": str(tmp_path / "results"),
                },
                "data": {
                    "download": False,
                    "image_size": 32,
                    "resize_size": 40,
                    "incremental_save_every": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["data", "prepare", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "data prepare complete" in result.output

    fingerprint = raw_results_dir(load_config(config_path)) / "fingerprint.json"
    assert fingerprint.is_file()
