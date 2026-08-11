from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tools.cuda_resume_canary import (
    CanaryError,
    _gpu_lease,
    _safe_run_directory,
    _state_differences,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cuda_resume_canary_exposes_safe_output_root_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "cuda_resume_canary.py"), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output-root" in result.stdout
    assert ".artifacts/cuda-resume-canary" in "".join(result.stdout.split())


def test_state_comparison_requires_exact_nested_tensor_and_scaler_values() -> None:
    reference = {
        "optimizer": {"state": {0: {"step": torch.tensor(3), "exp_avg": torch.ones(2)}}},
        "scaler": {"scale": 65536.0, "_growth_tracker": 3},
    }
    exact = {
        "optimizer": {"state": {0: {"step": torch.tensor(3), "exp_avg": torch.ones(2)}}},
        "scaler": {"scale": 65536.0, "_growth_tracker": 3},
    }
    changed = {
        "optimizer": {"state": {0: {"step": torch.tensor(3), "exp_avg": torch.zeros(2)}}},
        "scaler": {"scale": 65536.0, "_growth_tracker": 0},
    }

    assert _state_differences(reference, exact) == []
    assert _state_differences(reference, changed) == [
        "state.optimizer.state.0.exp_avg: tensor values differ",
        "state.scaler._growth_tracker: values differ",
    ]


def test_canary_output_must_stay_in_gitignored_root(tmp_path: Path) -> None:
    with pytest.raises(CanaryError, match="must stay under"):
        _safe_run_directory(tmp_path, Path("results"), "unsafe")

    safe = _safe_run_directory(tmp_path, Path(".artifacts/cuda-resume-canary"), "safe-run")
    assert safe == tmp_path / ".artifacts" / "cuda-resume-canary" / "safe-run"


def test_gpu_lease_is_atomic_and_released_on_exception(tmp_path: Path) -> None:
    lease_path = tmp_path / "gpu.lease"
    with (
        pytest.raises(RuntimeError, match="controlled interruption"),
        _gpu_lease(lease_path) as lease_token,
    ):
        assert len(lease_token) == 32
        assert lease_path.is_file()
        with pytest.raises(CanaryError, match="already exists"), _gpu_lease(lease_path):
            pass
        raise RuntimeError("controlled interruption")
    assert not lease_path.exists()


def test_external_gpu_lease_is_never_removed(tmp_path: Path) -> None:
    lease_path = tmp_path / "gpu.lease"
    lease_path.write_text('{"token": "external"}\n', encoding="utf-8")

    with pytest.raises(CanaryError, match="already exists"), _gpu_lease(lease_path):
        pass

    assert lease_path.read_text(encoding="utf-8") == '{"token": "external"}\n'
