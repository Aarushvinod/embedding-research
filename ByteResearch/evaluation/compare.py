# ByT5 versus mT5 comparisons of scored translations, with paired bootstrap significance.
# For one size, condition and test set, the scores come from the per-direction translate files
# results/translate/{model}__{condition}__{target}/{testset}.json of both architectures.
# Each (target, metric) cell compares the two direction models on the same sentence ids; corpus metrics
# resample sacreBLEU statistics and mean metrics resample values.
# The pooled cell macro-averages over the condition's targets while resampling the test-set rows jointly
# across targets, because the rows are parallel; all cells use stats.bootstrap.paired_bootstrap with fixed
# seeds. Results go to results/compare/{size}__{condition}/{testset}.json (with a __pilot suffix under --pilot).

from __future__ import annotations

import numpy as np

from ..core.config import ARCHS, CONDITIONS, TARGETS, InterpConfig, config_hash, model_for, resolve_langs
from ..core.paths import compare_id, run_id
from ..core.results import ResultFile, locate, run_result
from ..interp.units import measurement_done
from ..metrics import sacre
from ..metrics.api import METRICS, canonical
from ..stats.bootstrap import paired_bootstrap
from .generate import EXP as TRANSLATE_EXP

EXP = "compare"
CELL_FIELDS = ("target", "metric", "byt5", "mt5", "delta", "ci_low", "ci_high", "p_value", "n", "significant",
               "byt5_gain", "better")


def translate_file(ctx, model: str, condition: str, target: str, testset: str) -> ResultFile:
    # Inputs: Ctx, model key, condition, target and test set. Output: the ResultFile of that direction's
    # translate result.
    return ResultFile(locate(ctx, TRANSLATE_EXP, run_id(model, condition, target), testset)["result"])


def condition_targets(ctx, condition: str) -> list:
    # Inputs: Ctx and condition key. Output: the condition's targets a comparison covers (restricted to
    # the smoke targets in smoke mode).
    allowed = resolve_langs("all", TARGETS, ctx.smoke)
    return [t for t in CONDITIONS[condition].targets if t in allowed]


def load_scored(ctx, model: str, condition: str, testset: str, metric: str) -> dict:
    # Inputs: Ctx, model, condition, test set and canonical metric. Output: {target: {sid: value}} over
    # the condition's targets, with stats vectors for corpus metrics and sentence values for mean metrics;
    # raises when a direction's file lacks the metric.
    out = {}
    for target in condition_targets(ctx, condition):
        data = translate_file(ctx, model, condition, target, testset).load() or {}
        rows = data.get("per_item", {}).get("rows")
        scores = data.get("per_item", {}).get("scores", {}).get(metric)
        if not rows or not scores:
            raise FileNotFoundError(f"{run_id(model, condition, target)}/{testset} has no {metric} scores; "
                                    f"run the score stage")
        col = scores["stats"] if METRICS[metric]["kind"] == "corpus" else scores["values"]
        out[target] = dict(zip(rows["sid"], col))
    return out


def input_digests(ctx, runs, testset: str, metrics) -> dict:
    # Inputs: Ctx, (model, condition, target) directions, test set and metrics. Output: {"run_id|metric":
    # rows digest of the stored scores, or None}, so comparisons are redone when their inputs change.
    out = {}
    for model, cond, target in runs:
        data = translate_file(ctx, model, cond, target, testset).load() or {}
        scores = data.get("aggregate", {}).get("scores", {})
        out.update({f"{run_id(model, cond, target)}|{m}": (scores.get(m) or {}).get("rows_sha256") for m in metrics})
    return out


def aligned_arrays(maps) -> list:
    # Input: a list of {sid: value-or-stats} maps. Output: one float array per map over the sorted
    # sentence ids shared by all maps ([n] values or [n, k] statistics).
    sids = sorted(set.intersection(*(set(m) for m in maps)))
    if not sids:
        raise ValueError("no shared sentence ids to compare")
    return [np.asarray([m[s] for s in sids], dtype=np.float64) for m in maps]


def contrast(plus, minus, metric: str, fn, boot: dict, scale: float = 1.0) -> dict:
    # Inputs: lists of {sid: value} maps added and subtracted (one per system and target), metric,
    # corpus score_fn (None for mean metrics), bootstrap settings {n_boot, seed, alpha} and a scale.
    # Output: the paired bootstrap of macro(plus) - macro(minus) with rows resampled jointly across
    # every map, delta and CI multiplied by scale, plus byt5_gain (delta signed so that > 0 favours
    # the plus side under the metric's direction).
    arrs = aligned_arrays(list(plus) + list(minus))
    a, b = np.stack(arrs[:len(plus)]), np.stack(arrs[len(plus):])
    r = paired_bootstrap(a, b, METRICS[metric]["kind"], fn, **boot)
    lo, hi = sorted((r["ci_low"] * scale, r["ci_high"] * scale))
    delta = r["delta"] * scale
    return {**r, "delta": delta, "ci_low": lo, "ci_high": hi,
            "byt5_gain": delta if METRICS[metric]["higher"] else -delta}


def compare_cell(byt5: dict, mt5: dict, targets, metric: str, fn, boot: dict, label: str) -> dict:
    # Inputs: {target: {sid: value}} maps of both models, the targets to pool (one for a per-target
    # cell), metric, score_fn, bootstrap settings and the cell's target label. Output: one compare
    # cell with byt5/mt5 scores, delta = byt5 - mt5, CI, p-value and the significant winner.
    r = contrast([byt5[t] for t in targets], [mt5[t] for t in targets], metric, fn, boot)
    better = ("byt5" if r["byt5_gain"] > 0 else "mt5") if r["significant"] else None
    return {**r, "target": label, "metric": metric, "byt5": r["a_score"], "mt5": r["b_score"], "better": better}


def columnar(cells, fields) -> dict:
    # Inputs: cell dicts and the field names to keep. Output: {field: [values]} aligned over cells.
    return {f: [c.get(f) for c in cells] for f in fields}


def boot_settings(ctx) -> dict:
    # Input: Ctx. Output: {n_boot, seed, alpha} from InterpConfig and the global seed.
    cfg = ctx.config(InterpConfig)
    return {"n_boot": cfg.n_boot, "seed": ctx.seed, "alpha": cfg.alpha}


def metric_score_fn(ctx, metric: str):
    # Inputs: Ctx and canonical metric. Output: the corpus score_fn for chrF++/spBLEU, else None.
    return sacre.score_fn(sacre.metric_object(ctx, metric), metric) if METRICS[metric]["kind"] == "corpus" else None


def size_models(size: str) -> tuple:
    # Input: a model size. Output: (ByT5 key, mT5 key) of that size, in core.config.ARCHS order.
    return tuple(model_for(a, size) for a in ARCHS)


def compare_config(ctx, size: str, condition: str, testset: str, metrics) -> dict:
    # Inputs: Ctx, size, condition, test set and metrics. Output: the compare config (metrics,
    # bootstrap settings and input digests) whose hash fingerprints the result.
    runs = [(m, condition, t) for m in size_models(size) for t in condition_targets(ctx, condition)]
    return {"size": size, "condition": condition, "testset": testset, "metrics": list(metrics),
            **boot_settings(ctx), "inputs": input_digests(ctx, runs, testset, metrics)}


def compare_size_condition(ctx, size: str, condition: str, testset: str, metrics) -> dict:
    # Inputs: Ctx, size, condition, test set and metrics. Writes per-(target, metric) cells and pooled
    # aggregates to results/compare/{size}__{condition}/{testset}.json. Output: the envelope.
    metrics = [canonical(m) for m in metrics]
    cfg, boot = compare_config(ctx, size, condition, testset, metrics), boot_settings(ctx)
    bm, mm = size_models(size)
    cells, pooled, wins = [], {}, {}
    for metric in metrics:
        fn = metric_score_fn(ctx, metric)
        byt5, mt5 = (load_scored(ctx, m, condition, testset, metric) for m in (bm, mm))
        targets = [t for t in byt5 if t in mt5]
        per_target = [compare_cell(byt5, mt5, [t], metric, fn, boot, t) for t in targets]
        pooled[metric] = compare_cell(byt5, mt5, targets, metric, fn, boot, "pooled")
        wins[metric] = {k: sum(c["better"] == k for c in per_target) for k in ("byt5", "mt5", None)}
        cells += per_target
    rf = run_result(ctx, EXP, compare_id(size, condition), testset)
    aggregate = {"pooled": pooled, "wins": {m: {"byt5": w["byt5"], "mt5": w["mt5"], "ns": w[None]} for m, w in wins.items()},
                 "models": {"byt5": bm, "mt5": mm}}
    return rf.write({"cells": columnar(cells, CELL_FIELDS)}, aggregate, config=cfg, complete=True,
                    extra={"status": {"fingerprint": config_hash(cfg)}})


def compare_done(ctx, size: str, condition: str, testset: str, metrics) -> bool:
    # Inputs: Ctx, size, condition, test set and metrics. Output: whether the compare result is
    # complete for exactly this config (metrics, bootstrap settings and current input scores).
    cfg = compare_config(ctx, size, condition, testset, [canonical(m) for m in metrics])
    return measurement_done(locate(ctx, EXP, compare_id(size, condition), testset)["result"], config_hash(cfg))
