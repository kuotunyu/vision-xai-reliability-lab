# Compatibility notes and negative results

This file records public, reproducible lessons that affect how the repository
is operated. It contains no private progress log or machine-specific path.

## Windows: isolate the Python interpreter

A virtual environment created from an Anaconda base interpreter failed to load
PyTorch's native libraries because incompatible OpenMP/MKL DLLs were visible on
the process search path. Reinstalling the wheels did not change the failure.

The repository therefore pins Python 3.11 in `.python-version` and sets
`python-preference = "only-managed"` for uv. A fresh `uv sync --frozen` uses an
isolated uv-managed interpreter. Docker deliberately overrides this setting to
`only-system` because the official `python:3.11-slim` image already provides an
isolated interpreter.

## CUDA notebooks: editable installs are not visible immediately

In a running notebook kernel, `pip install -e .` can add a `.pth` file that is
only processed when Python starts. Importing the package in the same kernel can
therefore fail. For notebook environments, use a regular
`pip install . --no-deps` after confirming that the preinstalled CUDA
torch/torchvision pair satisfies this project's version floors.

## Full-scale spurious-patch experiment: negative result

The patched-trained models performed almost identically on correlated,
no-patch, and counter-correlated test variants, while attribution energy on the
patch stayed near zero. Under this frozen-backbone, linear-head training regime,
the models did not learn the intended shortcut. This is a real negative result,
not evidence that vision models generally resist spurious cues. Exact aggregate
metrics are in `results/derived/summary.json`.

## CUDA resume evidence has a narrow scope

The release canary compares a tiny synthetic, head-only training run completed
continuously with the same run interrupted at an epoch checkpoint and resumed.
It verifies the actual CUDA AMP/GradScaler checkpoint path, but it is not a
repeat of the full Oxford-IIIT Pet experiment and does not validate every GPU,
driver, or PyTorch release.
