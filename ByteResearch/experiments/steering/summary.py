# English-specific steering damage tables with paired bootstrap confidence intervals.
# For every readout (per-char log-prob, chrF++, spBLEU, COMET, MetricX), setting group and target,
# the drop under a direction is base minus edited (edited minus base for MetricX, where higher is
# worse), so a positive drop is damage.
# English-specific damage is the English drop minus each control's drop, minus the mean of the three
# language controls, and minus the random control's drop.
# Every system is bootstrapped with one shared row-index matrix, so contrasts are paired, and pooled
# rows average the per-target replicates, which resamples FLORES rows jointly across targets.
# A value is carried as one vector [point, replicate_1, ...] so contrasts are plain arithmetic, and
# readout_table is shared with the ByT5-versus-mT5 comparison.

from __future__ import annotations

import numpy as np

from ...core.config import InterpConfig
from ...core.registry import register_stage
from ...core.results import load_array
from ...interp.units import fingerprint
from ...stats.bootstrap import CORPUS_SCORE_FNS as CORPUS
from ...stats.bootstrap import boot_index, percentile_interval, point_corpus, point_mean, stat_corpus, stat_mean
from .measure import EXP, GRID, OPTIONS, complete, make_plan, result
from .settings import BASE_ID, group_label, parse_setting

READOUTS = ("logprob", "chrf_pp", "spbleu", "comet", "metricx")
SIGN = {"metricx": -1.0}
INPUTS = ("logprob", "metrics")


def logprob_data(ctx, plan) -> dict | None:
    # Inputs: Ctx and Plan. Output: {target: {sid: per-char log-probs [n]}} from the logprob result,
    # or None when it is not complete.
    data = complete(ctx, plan, "logprob")
    if data is None:
        return None
    out = {}
    for key, item in data["per_item"].items():
        sid, x = key.split("|")
        out.setdefault(x, {})[sid] = np.asarray(item["per_char"], dtype=np.float64)
    return out


def metric_data(ctx, plan, metric: str) -> dict | None:
    # Inputs: Ctx, Plan and a metric. Output: {target: {sid: values [n] or corpus stats [n, k]}} from
    # the metrics result, or None when that metric was not scored.
    data = result(ctx, plan, "metrics").load() or {}
    items = {k.split("|")[1]: v for k, v in (data.get("per_item") or {}).items() if k.split("|")[0] == metric}
    if not items:
        return None
    out = {}
    for x, item in items.items():
        with load_array(ctx, item["array"]) as z:
            arr = z["stats"] if metric in CORPUS else z["values"]
        out[x] = {sid: np.asarray(arr[i], dtype=np.float64) for i, sid in enumerate(item["settings"])}
    return out


def readout_data(ctx, plan, readout: str) -> dict | None:
    # Inputs: Ctx, Plan and a readout name. Output: its {target: {sid: data}} or None if unavailable.
    return logprob_data(ctx, plan) if readout == "logprob" else metric_data(ctx, plan, readout)


def system_values(data_x: dict, idx: np.ndarray, score_fn) -> dict:
    # Inputs: {sid: per-row values or stats} of one target, the shared index matrix and a corpus
    # score_fn (None for mean metrics). Output: {sid: [point, replicates...]}.
    if score_fn is None:
        return {sid: np.concatenate([[point_mean(a)], stat_mean(a, idx)]) for sid, a in data_x.items()}
    return {sid: np.concatenate([[point_corpus(a, score_fn)], stat_corpus(a, idx, score_fn)]) for sid, a in data_x.items()}


def setting_groups(sids) -> dict:
    # Input: setting ids. Output: {group label: {direction: sid}} for every non-base setting.
    groups = {}
    for sid in sids:
        s = parse_setting(sid)
        if s.scope != BASE_ID:
            groups.setdefault(group_label(s), {})[s.direction] = sid
    return groups


def contrasts(values: dict, members: dict, controls, sign: float) -> dict:
    # Inputs: {sid: vector} of one target, {direction: sid} of one group, the language controls and
    # the readout sign. Output: {name: vector} with scores, drops and English-specific damages.
    base = values[BASE_ID]
    drop = {d: sign * (base - values[sid]) for d, sid in members.items()}
    ctrls = [c for c in controls if c in drop]
    out = {"score:base": base, **{f"score:{d}": values[sid] for d, sid in members.items()}}
    out.update({f"drop:{d}": v for d, v in drop.items()})
    out.update({f"excess:{c}": drop["en"] - drop[c] for c in ctrls})
    if ctrls:
        out["excess:langmean"] = drop["en"] - np.mean([drop[c] for c in ctrls], axis=0)
    if "rand" in drop:
        out["excess:rand"] = drop["en"] - drop["rand"]
    return out


def readout_table(ctx, plan, readout: str, n_boot: int, seed: int) -> dict | None:
    # Inputs: Ctx, Plan, readout, bootstrap size and seed. Output: {group: {target | "pooled": {name:
    # vector}}} with one shared row resampling for all systems and targets, or None if unavailable.
    data = readout_data(ctx, plan, readout)
    targets = [x for x in plan.targets if data and x in data]
    if not targets:
        return None
    n = len(next(iter(data[targets[0]].values())))
    idx = boot_index(n, n_boot, seed)
    score_fn = CORPUS[readout]() if readout in CORPUS else None
    values = {x: system_values(data[x], idx, score_fn) for x in targets}
    table = {}
    for label, members in setting_groups(data[targets[0]]).items():
        rows = {x: contrasts(values[x], members, plan.controls, SIGN.get(readout, 1.0)) for x in targets}
        rows["pooled"] = {k: np.mean([rows[x][k] for x in targets], axis=0) for k in rows[targets[0]]}
        table[label] = rows
    return table


def interval(vec: np.ndarray, alpha: float) -> dict:
    # Inputs: a [point, replicates...] vector and alpha. Output: {point, ci_low, ci_high, p_value}
    # with a percentile CI and a two-sided bootstrap p-value clamped to [1/n_boot, 1].
    ci = percentile_interval(vec[1:], alpha)
    return {"point": float(vec[0]), **{k: ci[k] for k in ("ci_low", "ci_high", "p_value")}}


def format_table(table: dict, alpha: float) -> dict:
    # Inputs: a readout table and alpha. Output: the same nesting with every vector as an interval.
    return {g: {x: {k: interval(v, alpha) for k, v in names.items()} for x, names in rows.items()}
            for g, rows in table.items()}


def summary_config(ctx, plan) -> dict:
    # Inputs: Ctx and Plan. Output: the summary config, including the fingerprint and update time of
    # each input result, so new logprob or metric results make the summary stale.
    icfg = ctx.config(InterpConfig)
    inputs = {}
    for m in INPUTS:
        data = result(ctx, plan, m).load() or {}
        inputs[m] = [data.get("status", {}).get("fingerprint"), data.get("provenance", {}).get("updated")]
    return {"n_boot": icfg.n_boot, "seed": icfg.seed, "alpha": icfg.alpha, "targets": list(plan.targets), "inputs": inputs}


def summary_done(ctx, unit) -> bool:
    # Inputs: Ctx and Unit. Output: whether a complete summary exists for the current inputs.
    plan = make_plan(ctx, unit)
    data = complete(ctx, plan, "summary")
    return bool(data) and data.get("status", {}).get("fingerprint") == fingerprint(summary_config(ctx, plan))


def summary(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Writes the damage tables for every available readout,
    # per target and pooled, with paired bootstrap intervals.
    plan = make_plan(ctx, unit)
    cfg = summary_config(ctx, plan)
    tables = {r: readout_table(ctx, plan, r, cfg["n_boot"], cfg["seed"]) for r in READOUTS}
    agg = {r: format_table(t, cfg["alpha"]) for r, t in tables.items() if t is not None}
    result(ctx, plan, "summary").write({}, {"readouts": agg, "targets": list(plan.targets)}, config=cfg, complete=True,
                                       extra={"status": {"fingerprint": fingerprint(cfg)}})


register_stage("interp.steering.summary", grid=GRID, is_done=summary_done,
               requires=("interp.steering.logprob", "interp.steering.metrics"), options=OPTIONS, order=606,
               experiment=EXP)(summary)
