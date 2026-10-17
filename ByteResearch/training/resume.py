# Checkpoint bookkeeping for exact, safe resumption of fine-tuning runs.
# finalize_checkpoint writes byteresearch_meta.json and then, last of all, the COMPLETE marker, so a
# directory without COMPLETE is by definition half-written and latest_complete_checkpoint skips it.
# check_resume_compat refuses to resume when the config hash, data hash, world size, per-device
# batch, gradient accumulation or tag ids differ from the run being started.
# finished_state tells whether a checkpoint already ended training (step budget or early stopping),
# so a rerun after a crash goes straight to the export instead of training past the stop.
# export_best copies the best checkpoint's model and tokenizer to ROOT/checkpoints/{run_id}/best/
# with fresh meta and COMPLETE, the layout modeling.loading.load_trained reads.

from __future__ import annotations

import os
import shutil
from pathlib import Path

from transformers import TrainerState

from ..core.io import read_json, write_json_atomic
from ..core.paths import COMPLETE_MARKER, META_FILE
from ..core.results import now_iso

COMPAT_KEYS = ("config_hash", "data_hash", "world_size", "per_device_batch", "grad_accum", "tags")
STATE_FILE = "trainer_state.json"
EXPORT_SKIP = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state*.pth", "training_args.bin",
               STATE_FILE, META_FILE, COMPLETE_MARKER)


class ResumeMismatch(ValueError):
    # Raised when a complete checkpoint exists but belongs to an incompatible run setup.
    pass


def finalize_checkpoint(ckpt, meta: dict) -> Path:
    # Inputs: a checkpoint directory and its meta dict. Writes byteresearch_meta.json atomically and
    # then the COMPLETE marker (always last). Output: the directory.
    ckpt = Path(ckpt)
    write_json_atomic(ckpt / META_FILE, {**meta, "finalized": now_iso()})
    (ckpt / COMPLETE_MARKER).write_text("ok", encoding="utf-8")
    return ckpt


def latest_complete_checkpoint(paths, rid: str):
    # Inputs: Paths and run id. Output: the newest checkpoint-N directory holding COMPLETE, or None;
    # directories without the marker are ignored.
    done = paths.complete_checkpoints(rid)
    return done[-1][1] if done else None


def remove_incomplete(paths, rid: str) -> list:
    # Inputs: Paths and run id. Deletes every checkpoint-* directory without COMPLETE (leftovers of a
    # killed save) so checkpoint rotation never counts them. Output: the removed paths.
    base = paths.checkpoints_dir(rid)
    stale = [p for p in base.glob("checkpoint-*") if p.is_dir() and not (p / COMPLETE_MARKER).exists()]
    for p in stale:
        shutil.rmtree(p, ignore_errors=True)
    return stale


def read_meta(ckpt) -> dict:
    # Input: a checkpoint directory. Output: its byteresearch_meta.json; FileNotFoundError if missing.
    meta = read_json(Path(ckpt) / META_FILE)
    if meta is None:
        raise FileNotFoundError(f"missing or unreadable {Path(ckpt) / META_FILE}")
    return meta


def check_resume_compat(stored: dict, expected: dict) -> None:
    # Inputs: the meta of the checkpoint to resume and the meta of the run being started. Raises
    # ResumeMismatch naming every differing compatibility key. No output.
    bad = {k: (stored.get(k), expected.get(k)) for k in COMPAT_KEYS if stored.get(k) != expected.get(k)}
    if bad:
        detail = "; ".join(f"{k}: checkpoint={a!r} now={b!r}" for k, (a, b) in bad.items())
        raise ResumeMismatch(f"cannot resume, run setup changed ({detail}); rerun with --force to start over")


def load_state(ckpt) -> TrainerState:
    # Input: a checkpoint directory. Output: its TrainerState.
    return TrainerState.load_from_json(str(Path(ckpt) / STATE_FILE))


def finished_state(state: TrainerState, max_steps: int, patience) -> bool:
    # Inputs: a TrainerState, the step budget and the early-stopping patience (None when disabled).
    # Output: whether that state already ended training.
    if state.global_step >= max_steps:
        return True
    early = (state.stateful_callbacks or {}).get("EarlyStoppingCallback") or {}
    early = early[-1] if isinstance(early, list) and early else early
    counter = (early.get("attributes") or {}).get("early_stopping_patience_counter", 0) if early else 0
    return patience is not None and counter >= patience


def best_checkpoint_dir(paths, rid: str, state: TrainerState):
    # Inputs: Paths, run id and TrainerState. Output: the checkpoint-N directory of the best step, or
    # None when no best step is recorded.
    return None if state.best_global_step is None else paths.checkpoint(rid, state.best_global_step)


def export_best(paths, rid: str, state: TrainerState, meta: dict) -> Path:
    # Inputs: Paths, run id, final TrainerState and the run meta. Copies the best checkpoint's model,
    # config and tokenizer into a temporary directory, writes meta (step, metric) and COMPLETE, then
    # swaps it in as ROOT/checkpoints/{rid}/best. Output: the best directory.
    src = best_checkpoint_dir(paths, rid, state)
    if src is None or not (src / COMPLETE_MARKER).exists():
        raise FileNotFoundError(f"best checkpoint {src} is missing or incomplete")
    dest, tmp = paths.best_checkpoint(rid), paths.checkpoints_dir(rid) / "best.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(*EXPORT_SKIP))
    finalize_checkpoint(tmp, {**meta, "step": state.best_global_step, "best_metric": state.best_metric,
                              "source": src.name})
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest
