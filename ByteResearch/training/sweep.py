# The learning-rate sweep and its per-architecture selection.
# Stage `sweep` trains every (model, sweep direction, lr) cell for a fixed number of steps with the same
# warmup, batch and data order as the full runs, but without early stopping, a best export or any checkpoint
# (to spare a Drive-backed ROOT the writes), so a killed cell restarts from the beginning.
# The sweep directions are the fixed representative (condition, target) pairs of TrainConfig.sweep_directions;
# --condition and --target only narrow them (so DDP relaunches can pin one cell), and each cell writes
# ROOT/results/sweep/{model}__{condition}__{target}__lr{lr}/train.json.
# Stage `sweep.select` picks, per model key, the LR with the best mean final dev chrF++ across the sweep
# directions and writes ROOT/results/sweep/{model}/selection.json, recomputed whenever a sweep cell changes.
# The full `train` stage reads that selection for every direction of the model whenever --lr is not given.

from __future__ import annotations

from ..core.config import TARGETS, TrainConfig, resolve_conditions, resolve_langs, resolve_lrs, resolve_models, \
    smoke_overrides
from ..core.paths import run_id
from ..core.registry import Unit, register_stage
from ..core.results import result_done, result_file
from .dataset import data_requires
from .train import MEASUREMENT, OPTIONS, SWEEP_EXP, RunSpec, run_training


def sweep_directions(smoke: bool) -> list:
    # Input: the smoke flag. Output: the (condition, target) sweep directions of the (smoke-aware) TrainConfig.
    cfg = smoke_overrides(TrainConfig()) if smoke else TrainConfig()
    return [tuple(d) for d in cfg.sweep_directions]


def sweep_units(sel: dict, smoke: bool) -> list:
    # Inputs: the CLI selections and the smoke flag. Output: one Unit per selected model, sweep direction
    # within the --condition and --target selections, and selected lr (the sweep grid when --lr is not given).
    lrs = resolve_lrs(sel.get("lr"), smoke_overrides(TrainConfig()) if smoke else TrainConfig())
    conds = resolve_conditions(sel.get("condition"), smoke)
    targets = resolve_langs(sel.get("target"), TARGETS, smoke)
    dirs = [(c, t) for c, t in sweep_directions(smoke) if c in conds and t in targets]
    return [Unit(model=m, condition=c, target=t, lr=lr) for m in resolve_models(sel.get("model"), smoke)
            for c, t in dirs for lr in lrs]


def sweep_done(ctx, unit) -> bool:
    # Inputs: Ctx and Unit (model, condition, target, lr). Output: whether that cell's train.json is complete.
    return result_done(ctx, SWEEP_EXP, unit.run_id, MEASUREMENT)


@register_stage("sweep", grid=sweep_units, is_done=sweep_done, requires=data_requires(), ddp=True, order=250,
                options=OPTIONS)
def sweep(ctx, unit) -> None:
    # Inputs: Ctx and Unit (model, condition, target, lr). Runs one sweep cell from its first step.
    run_training(ctx, RunSpec(unit.model, unit.condition, unit.target, float(unit.lr), experiment=SWEEP_EXP,
                              sweep=True))


def final_metric(ctx, model: str, condition: str, target: str, lr: float):
    # Inputs: Ctx, model, condition, target and lr. Output: the cell's final dev chrF++, or None when its
    # train.json is missing or incomplete.
    data = result_file(ctx, SWEEP_EXP, run_id(model, condition, target, lr), MEASUREMENT).load()
    if not data or not data.get("status", {}).get("complete"):
        return None
    return data["aggregate"].get("final_metric")


def selection(ctx, model: str) -> dict:
    # Inputs: Ctx and model key. Output: {"lr", "mean_chrf_pp", "by_lr": {lr: {"condition__target": chrF++,
    # "mean": ...}}, "directions", "lrs"}; raises RuntimeError listing missing sweep cells.
    dirs = sweep_directions(ctx.smoke)
    lrs = resolve_lrs(ctx.sel.get("lr"), ctx.config(TrainConfig))
    table = {lr: {(c, t): final_metric(ctx, model, c, t, lr) for c, t in dirs} for lr in lrs}
    missing = [run_id(model, c, t, lr) for lr, row in table.items() for (c, t), v in row.items() if v is None]
    if missing:
        raise RuntimeError(f"sweep cells missing or incomplete for {model}: {missing}; run `sweep` first")
    by_lr = {f"{lr:g}": {**{f"{c}__{t}": v for (c, t), v in row.items()}, "mean": sum(row.values()) / len(row)}
             for lr, row in table.items()}
    best = max(lrs, key=lambda lr: (by_lr[f"{lr:g}"]["mean"], -lr))
    return {"lr": best, "mean_chrf_pp": by_lr[f"{best:g}"]["mean"], "by_lr": by_lr,
            "directions": [list(d) for d in dirs], "lrs": lrs}


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
             config={"directions": chosen["directions"], "lrs": chosen["lrs"],
                     "criterion": "mean final dev chrF++ over the sweep directions"}, complete=True)
