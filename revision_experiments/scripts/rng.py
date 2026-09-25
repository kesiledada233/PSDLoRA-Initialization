"""Explicitly separated initialization and training random streams."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch


def derive_init_seed(run_seed: int, method: str) -> int:
    payload = f"revision-init-v1:{int(run_seed)}:{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_initialization(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_training(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def training_probe() -> dict:
    """Small serializable probe used to verify paired RNG reset behavior."""
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(())),
    }


def probe_hash(probe: dict) -> str:
    return hashlib.sha256(json.dumps(probe, sort_keys=True).encode("utf-8")).hexdigest()
