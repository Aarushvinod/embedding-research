# ByT5 versus mT5 comparison of English-specific steering damage within one size and condition.
# Raw effect sizes are not comparable across architectures, so the comparison is made on the
# control-normalized excess damage: the English drop minus the language-control mean, minus the
# random control, and minus each language control.
# Both models are scored on the same FLORES devtest rows, so one shared bootstrap index matrix pairs
# their replicates and the difference ByT5 minus mT5 gets a paired interval and p-value.
# Setting groups are matched by label, so "best:c1.0" compares each model at its own chosen layer.
# This module also declares the interp.steering composite, which leaves out the estimate dry run.

from __future__ import annotations

from ...core.config import InterpConfig, model_for
from ...core.registry import Unit, composite, register_stage
from ...core.results import run_done, run_result
from .measure import EXP, OPTIONS, complete, make_plan
from .summary import READOUTS, interval, readout_table

ARCH_PAIR = ("byt5", "mt5")
MEMBERS = ("interp.steering.directions", "interp.steering.select", "interp.steering.logprob",
           "interp.steering.generate", "interp.steering.metrics", "interp.steering.check",
           "interp.steering.summary", "interp.steering.compare")


def both_archs(unit) -> list:
    # Input: a (size, condition) Unit. Output: the ByT5 and mT5 (model, condition) units of that size,
    # whose summaries the comparison needs whatever --model selected.
    return [Unit(model=model_for(a, unit.size), condition=unit.condition) for a in ARCH_PAIR]


def pair_plans(ctx, unit) -> dict:
    # Inputs: Ctx and a (size, condition) Unit. Output: {arch: Plan} for the ByT5 and mT5 models of
    # that size; raises if their devtest rows differ (the bootstrap must be paired).
    plans = {u.model.split("-")[0]: make_plan(ctx, u) for u in both_archs(unit)}
    rows = {a: (complete(ctx, p, "logprob") or {}).get("config", {}).get("rows") for a, p in plans.items()}
    if rows[ARCH_PAIR[0]] != rows[ARCH_PAIR[1]]:
        raise ValueError(f"{unit.run_id}: ByT5 and mT5 logprob rows differ; rerun both on the same rows")
    return plans


def compare_names(controls) -> tuple:
    # Input: the language-control codes. Output: the compared contrasts: excess over the control
    # mean, over the random control and over each language control, plus the raw English drop.
    return ("excess:langmean", "excess:rand", *(f"excess:{c}" for c in controls), "drop:en")


def compare_tables(tables: dict, alpha: float, names) -> dict:
    # Inputs: {arch: readout table} for the two models, alpha and the contrast names. Output: {group:
    # {target: {name: {byt5, mt5, diff}}}} over the groups, targets and contrasts both models share.
    a, b = (tables[x] for x in ARCH_PAIR)
    out = {}
    for g in [g for g in a if g in b]:
        targets = [x for x in a[g] if x in b[g]]
        out[g] = {x: {k: {ARCH_PAIR[0]: interval(a[g][x][k], alpha), ARCH_PAIR[1]: interval(b[g][x][k], alpha),
                          "diff": interval(a[g][x][k] - b[g][x][k], alpha)}
                      for k in names if k in a[g][x] and k in b[g][x]} for x in targets}
    return out


def compare(ctx, unit) -> None:
    # Inputs: Ctx and a (size, condition) Unit. Writes ByT5 and mT5 excess damage and their paired
    # bootstrap difference for every readout both models have.
    icfg = ctx.config(InterpConfig)
    plans = pair_plans(ctx, unit)
    names = compare_names(plans[ARCH_PAIR[0]].controls)
    agg = {}
    for r in READOUTS:
        tables = {a: readout_table(ctx, p, r, icfg.n_boot, icfg.seed) for a, p in plans.items()}
        if all(t is not None for t in tables.values()):
            agg[r] = compare_tables(tables, icfg.alpha, names)
    cfg = {"n_boot": icfg.n_boot, "seed": icfg.seed, "alpha": icfg.alpha, "models": {a: p.run_id for a, p in plans.items()}}
    run_result(ctx, EXP, unit.run_id, "compare", unit=unit).write(
        {}, {"readouts": agg, "names": list(names)}, config=cfg, complete=True)


register_stage("interp.steering.compare", grid=("size", "condition"),
               is_done=lambda ctx, unit: run_done(ctx, EXP, unit.run_id, "compare"),
               requires=(("interp.steering.summary", both_archs),), options=OPTIONS, order=607)(compare)
composite("interp.steering", MEMBERS)
