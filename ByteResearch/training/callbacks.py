# Trainer callbacks and the train.json measurement built from the trainer's log history.
# CheckpointMetaCallback runs after every save: all ranks meet at a barrier (so every rank's RNG
# file is on disk), then rank 0 writes byteresearch_meta.json and finally the COMPLETE marker.
# ResultsCallback makes rank 0 rewrite ROOT/results/{train|sweep}/{run_id}/train.json after every evaluation
# (the run's end writes it once more, complete).
# The file is rebuilt from state.log_history each time, and a resume restores that history from the
# checkpoint, so entries logged after the last checkpoint are replaced rather than duplicated.
# DriveWatchdogCallback, on a Drive-backed root, makes every save first confirm with Drive's servers that the
# previous checkpoint arrived (training.drive_watch), stopping the run on every rank when it has not.
# train_results turns the history into columnar per_item blocks (train, eval, final) and an
# aggregate with the best and final dev chrF++, stop reason, wall time and examples seen.

from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

from ..core.dist import barrier, gather_objects
from .args import METRIC
from .drive_watch import DriveUploadError, confirm
from .resume import finalize_checkpoint

EVAL_KEY = f"eval_{METRIC}"


class CheckpointMetaCallback(TrainerCallback):
    # on_save hook that finalizes each checkpoint with meta and COMPLETE. Construct with
    # meta_fn(state) -> dict giving the meta for the checkpoint at state.global_step.

    def __init__(self, meta_fn):
        # Input: the meta factory.
        self.meta_fn = meta_fn

    def on_save(self, args, state, control, **kwargs):
        # Inputs: Trainer callback arguments. Waits for all ranks, lets rank 0 write meta then COMPLETE
        # into checkpoint-{global_step}, and waits again so no rank runs ahead of the marker.
        barrier()
        if state.is_world_process_zero:
            finalize_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", self.meta_fn(state))
        barrier()


class DriveWatchdogCallback(TrainerCallback):
    # on_save hook that, after each new checkpoint, confirms on rank 0 that Drive holds the previous one and
    # raises DriveUploadError on every rank when it does not. Construct with a DriveChecker (None on other
    # ranks) and the timeout in seconds.

    def __init__(self, checker, timeout: float):
        # Inputs: the DriveChecker (or None) and the timeout.
        self.checker = checker
        self.timeout = timeout
        self.previous = None

    def on_save(self, args, state, control, **kwargs):
        # Inputs: Trainer callback arguments. Checks the checkpoint saved before this one, remembers this one
        # for the next save and stops training when Drive does not hold the previous checkpoint.
        error = confirm(self.checker, self.previous, self.timeout) if state.is_world_process_zero else None
        self.previous = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        raise_first([e for e in gather_objects(error) if e])


def raise_first(errors: list) -> None:
    # Input: error messages gathered from every rank. Raises DriveUploadError with the first one, if any.
    if errors:
        raise DriveUploadError(errors[0])


class ResultsCallback(TrainerCallback):
    # on_evaluate hook that rewrites the run's train.json on rank 0. Construct with
    # write_fn(state, complete=False).

    def __init__(self, write_fn):
        # Input: the writer function.
        self.write_fn = write_fn

    def on_evaluate(self, args, state, control, **kwargs):
        # Inputs: Trainer callback arguments. Rank 0 rewrites the (incomplete) result from state, whose log
        # history holds every training log and evaluation so far.
        if state.is_world_process_zero:
            self.write_fn(state, complete=False)


def columnar(entries) -> dict:
    # Input: a list of log dicts. Output: {key: [value per entry]} over the union of keys in first-seen
    # order, None where an entry lacks a key.
    keys = list(dict.fromkeys(k for e in entries for k in e))
    return {k: [e.get(k) for e in entries] for k in keys}


def split_history(history) -> dict:
    # Input: TrainerState.log_history. Output: {"train", "eval", "final"} lists of entries (training
    # logs carry "loss", dev logs carry eval_chrf_pp, the end-of-training summary carries train_runtime).
    return {"train": [e for e in history if "loss" in e], "eval": [e for e in history if EVAL_KEY in e],
            "final": [e for e in history if "train_runtime" in e]}


def train_results(state, info: dict, stop_reason=None) -> tuple:
    # Inputs: TrainerState, run info {"max_steps", "global_batch", "n_train", "target", ...} and the stop
    # reason (None while running). Output: (per_item, aggregate) for train.json.
    parts = split_history(state.log_history)
    evals = parts["eval"]
    last = evals[-1] if evals else None
    aggregate = {
        "best_step": state.best_global_step, "best_metric": state.best_metric,
        "final_step": last and last["step"], "final_metric": last and last[EVAL_KEY],
        "global_step": state.global_step, "epoch": state.epoch,
        "examples_seen": state.global_step * info["global_batch"], "stop_reason": stop_reason,
        "wall_time_s": sum(e.get("interval_s") or 0.0 for e in state.log_history), **info,
    }
    return {k: columnar(v) for k, v in parts.items()}, aggregate
