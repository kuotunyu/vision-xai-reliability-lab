"""Deterministic seeding helpers.

Per-sample decisions (e.g. spurious-patch assignment) must not depend on iteration
order, subset choice, or platform, so they derive an RNG from a sha256 of
``(seed, namespace, sample_id)``. Python's builtin ``hash()`` is salted per process
and is never used for this.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the global RNGs (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def per_sample_rng(seed: int, sample_id: str, namespace: str) -> np.random.Generator:
    """Return an RNG that is a pure function of ``(seed, namespace, sample_id)``."""
    digest = hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))
