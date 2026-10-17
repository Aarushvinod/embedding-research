# Random-number control for reproducible ByteResearch runs.
# seed_everything seeds Python, NumPy and torch (CPU and CUDA) and can switch torch into
# deterministic mode, which also disables TF32 and sets the cuBLAS workspace variable.
# rng_snapshot and rng_restore capture and reinstate the complete generator state of all
# four sources, so evaluation inside training never perturbs the training random stream.
# The rng_preserved context manager wraps that pattern for any block of code.
# Interpretability code calls seed_everything with deterministic=True and fp32 matmuls.

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    # Inputs: integer seed and whether to force deterministic kernels. Seeds python, numpy and
    # torch (all CUDA devices); with deterministic=True also disables TF32 and cuDNN autotuning and
    # enables torch deterministic algorithms (warn-only). No output.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        set_deterministic()


def set_deterministic() -> None:
    # Switches torch to deterministic, fp32-exact settings: cuBLAS workspace config, no TF32,
    # deterministic cuDNN without benchmarking, deterministic algorithms in warn-only mode.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_snapshot() -> dict:
    # Output: a dict holding the python, numpy, torch CPU and (if available) all CUDA RNG states.
    snap = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def rng_restore(snap: dict) -> None:
    # Input: a dict from rng_snapshot. Restores every generator state it contains. No output.
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["torch"])
    if "cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["cuda"])


@contextlib.contextmanager
def rng_preserved():
    # Context manager: snapshots all RNG states on entry and restores them on exit (also after an
    # exception), so the wrapped block leaves the random streams untouched.
    snap = rng_snapshot()
    try:
        yield
    finally:
        rng_restore(snap)
