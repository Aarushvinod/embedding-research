# The learning-rate sweep and its per-model selection.
# Stage `sweep` trains every (model, condition, lr) cell for a fixed number of steps with the same
# warmup, batch and data order as the full runs, but without early stopping or a best export, and
# keeps two checkpoints so a killed cell still resumes; checkpoints are deleted once the cell is done.
# Each cell writes ROOT/results/sweep/{model}__{condition}__lr{lr}/train.json.
# Stage `sweep.select` picks, per model, the LR with the best mean final dev chrF++ across the study
# conditions (C1-C3, or the smoke conditions) and writes ROOT/results/sweep/{model}/selection.json.
# The full `train` stage reads that selection whenever --lr is not given.

from __future__ import annotations

import shutil

from ..core.config import TrainConfig, resolve_conditions, resolve_lrs
from ..core.dist import is_main
from ..core.paths import run_id
from ..core.registry import opt, register_stage
from ..core.results import result_done, result_file
from .dataset import data_requires
from .train import MEASUREMENT, OPTIONS, SWEEP_EXP, RunSpec, run_training


def sweep_done(ctx, unit) -> bool:
    # Inputs: Ctx and Unit (model, condition, lr). Output: whether that cell's train.json is complete.
    return result_done(ctx, SWEEP_EXP, unit.run_id, MEASUREMENT)


@register_stage("sweep", grid=("model", "condition", "lr"), is_done=sweep_done, requires=data_requires(),
                ddp=True, order=250,
                options=OPTIONS + (opt("--keep-checkpoints", action="store_true", help="keep sweep checkpoints"),))
def sweep(ctx, unit) -> None:
    # Inputs: Ctx and Unit (model, condition, lr). Runs one sweep cell, then removes its checkpoints
    # unless --keep-checkpoints.
    spec = RunSpec(unit.model, unit.condition, float(unit.lr), experiment=SWEEP_EXP, sweep=True)
    run_training(ctx, spec)
    if is_main() and not ctx.opt("keep_checkpoints", False):
        shutil.rmtree(ctx.paths.checkpoints_dir(spec.run_id), ignore_errors=True)


def final_metric(ctx, model: str, condition: str, lr: float):
    # Inputs: Ctx, model, condition and lr. Output: the cell's final dev chrF++, or None when its
    # train.json is missing or incomplete.
    data = result_file(ctx, SWEEP_EXP, run_id(model, condition, lr), MEASUREMENT).load()
    if not data or not data.get("status", {}).get("complete"):
        return None
    return data["aggregate"].get("final_metric")


def selection(ctx, model: str) -> dict:
    # Inputs: Ctx and model key. Output: {"lr", "mean_chrf_pp", "by_lr": {lr: {condition: chrF++,
    # "mean": ...}}, "conditions", "lrs"}; raises RuntimeError listing missing sweep cells.
    conds = resolve_conditions("all", ctx.smoke)
    lrs = resolve_lrs(ctx.sel.get("lr"), ctx.config(TrainConfig))
    table = {lr: {c: final_metric(ctx, model, c, lr) for c in conds} for lr in lrs}
    missing = [run_id(model, c, lr) for lr, row in table.items() for c, v in row.items() if v is None]
    if missing:
        raise RuntimeError(f"sweep cells missing or incomplete for {model}: {missing}; run `sweep` first")
    by_lr = {f"{lr:g}": {**row, "mean": sum(row.values()) / len(row)} for lr, row in table.items()}
    best = max(lrs, key=lambda lr: (by_lr[f"{lr:g}"]["mean"], -lr))
    return {"lr": best, "mean_chrf_pp": by_lr[f"{best:g}"]["mean"], "by_lr": by_lr, "conditions": conds,
            "lrs": lrs}


def select_done(ctx, unit) -> bool:
    # Inputs: Ctx and Unit (model). Output: whether selection.json is complete and still agrees with
    # the current sweep cells (a rerun or missing cell makes it stale, so it is recomputed).
    data = result_file(ctx, SWEEP_EXP, unit.model, "selection").load()
    if not (data and data.get("status", {}).get("complete")):
        return False
    try:
        chosen = selection(ctx, unit.model)
    except RuntimeError:
        return False
    return data["per_item"].get("by_lr") == chosen["by_lr"] and data["aggregate"].get("lr") == chosen["lr"]


@register_stage("sweep.select", grid=("model",), is_done=select_done, requires=("sweep",), order=260)
def select(ctx, unit) -> None:
    # Inputs: Ctx and Unit (model). Writes ROOT/results/sweep/{model}/selection.json.
    chosen = selection(ctx, unit.model)
    rf = result_file(ctx, SWEEP_EXP, unit.model, "selection", unit=unit)
    rf.write({"by_lr": chosen["by_lr"]}, {k: v for k, v in chosen.items() if k != "by_lr"},
             config={"conditions": chosen["conditions"], "lrs": chosen["lrs"], "criterion": "mean final dev chrF++"},
             complete=True)
