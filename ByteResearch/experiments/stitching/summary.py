# Summary stage of the stitching experiment: English-privilege gap curves per stitch layer.
# load_readouts gathers every readout of one recipient (per-char log-prob, chrF++, spBLEU, COMET and
# MetricX) as per-row arrays keyed by unit name and target, shared with the ByT5-vs-mT5 comparison.
# For each layer l the gap Q_en(l) - Q_ctrl(l) between English-derived and control-derived donor
# states gets a paired bootstrap per target and pooled, with FLORES rows resampled jointly across
# targets, and corpus metrics recomputed from resampled sentence statistics.
# Deficits of each donor stitch against the recipient's own forward (self) are tested the same way.
# Each donor's own translation quality is reported as context only, without any normalization, and a
# summary counts as done only while its inputs (logprob and metrics results) are unchanged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...core.config import InterpConfig
from ...core.registry import register_stage
from ...core.results import load_array, run_result
from ...interp.units import fingerprint
from ...stats.bootstrap import CORPUS_SCORE_FNS as CORPUS
from ...stats.bootstrap import paired_bootstrap, point_corpus, point_mean
from .measure import EXP, OPTIONS, ROLES, SELF, stored_result, context_name, stitch_grid, stitch_name, variant_of

LOWER_IS_BETTER = ("metricx",)


@dataclass(frozen=True)
class Readout:
    # One readout of one recipient: kind ("mean" over per-row values or "corpus" from per-row
    # statistics), which direction is better, the corpus score_fn (None for means) and the data
    # {unit name: {target: per-row array [n] or statistics [n, k]}}.
    kind: str
    better: str
    score_fn: Callable | None
    data: dict


def _by_unit(arrays: dict) -> dict:
    # Input: {"name|target": array}. Output: {name: {target: array}}.
    out = {}
    for key, arr in arrays.items():
        name, t = key.split("|")
        out.setdefault(name, {})[t] = np.asarray(arr)
    return out


def _metric_readout(ctx, metric: str, entry: dict) -> Readout:
    # Inputs: Ctx, metric name and its metrics-result entry. Output: the Readout (statistics for
    # chrF++/spBLEU when stored, per-row values otherwise).
    arrays = load_array(ctx, entry["array"])
    corpus = metric in CORPUS and "stats" in arrays.files
    block = arrays["stats"] if corpus else arrays["values"]
    data = _by_unit({k: block[i] for i, k in enumerate(entry["units"])})
    return Readout("corpus" if corpus else "mean", "lower" if metric in LOWER_IS_BETTER else "higher",
                   CORPUS[metric]() if corpus else None, data)


def load_readouts(ctx, rid: str, variant) -> tuple:
    # Inputs: Ctx, recipient run id and variant. Output: (logprob result config, {readout name:
    # Readout}); metrics are included when their result exists. A missing or incomplete logprob
    # result raises FileNotFoundError.
    lp = stored_result(ctx, rid, "logprob", variant)
    if not (lp and lp["status"].get("complete")):
        raise FileNotFoundError(f"stitching logprob result for {rid} ({variant}) is missing or incomplete")
    out = {"logprob": Readout("mean", "higher", None, _by_unit({k: v["per_char"] for k, v in lp["per_item"].items()}))}
    met = stored_result(ctx, rid, "metrics", variant) or {}
    out.update({m: _metric_readout(ctx, m, e) for m, e in (met.get("per_item") or {}).items()})
    return lp["config"], out


def finite_rows(a, b, kind: str):
    # Inputs: paired arrays [n] / [T, n] (or statistics) and the readout kind. Output: the arrays
    # restricted to rows that are finite in both (for every target); statistics pass unchanged.
    if kind != "mean":
        return a, b
    keep = np.isfinite(a).reshape(-1, a.shape[-1]).all(0) & np.isfinite(b).reshape(-1, b.shape[-1]).all(0)
    return a[..., keep], b[..., keep]


def stacked(by_target: dict, targets) -> np.ndarray:
    # Inputs: {target: per-row array} and the targets. Output: the arrays stacked as [T, n(, k)].
    return np.stack([by_target[t] for t in targets])


def paired(r: Readout, a: dict, b: dict, targets, ic: InterpConfig) -> dict:
    # Inputs: a Readout, two {target: array} dicts, the targets and InterpConfig. Output: {target:
    # paired bootstrap of a - b, "pooled": the same with rows resampled jointly across targets}.
    scopes = {t: (a[t], b[t]) for t in targets}
    scopes["pooled"] = (stacked(a, targets), stacked(b, targets))
    return {s: paired_bootstrap(*finite_rows(x, y, r.kind), kind=r.kind, score_fn=r.score_fn, n_boot=ic.n_boot,
                                seed=ic.seed, alpha=ic.alpha) for s, (x, y) in scopes.items()}


def point(r: Readout, by_target: dict, targets) -> dict:
    # Inputs: a Readout, {target: array} and the targets. Output: {target: score, "pooled": score}
    # (mean of finite values, or the corpus score; pooled is the mean or macro average over targets).
    score = (lambda x: point_mean(x[np.isfinite(x)])) if r.kind == "mean" else (lambda x: point_corpus(x, r.score_fn))
    out = {t: score(by_target[t]) for t in targets}
    out["pooled"] = point_mean([out[t] for t in targets]) if r.kind == "mean" else score(stacked(by_target, targets))
    return out


def summarize_readout(r: Readout, cfg: dict, ic: InterpConfig) -> dict:
    # Inputs: a Readout, the logprob config (layers, n_layers, targets) and InterpConfig. Output: the
    # per-layer gap (en - ctrl) and deficits (each donor - self) with bootstraps, the self and
    # context point scores, and the context gap.
    targets, L = cfg["targets"], cfg["n_layers"]
    d = r.data
    curve = [{"layer": l, "frac": l / L,
              "gap": paired(r, d[stitch_name("en", l)], d[stitch_name("ctrl", l)], targets, ic),
              "deficit_en": paired(r, d[stitch_name("en", l)], d[SELF], targets, ic),
              "deficit_ctrl": paired(r, d[stitch_name("ctrl", l)], d[SELF], targets, ic)} for l in cfg["layers"]]
    names = [SELF] + [context_name(role) for role in ROLES]
    return {"kind": r.kind, "better": r.better, "curve": curve,
            "points": {n: point(r, d[n], targets) for n in names},
            "context_gap": paired(r, d[context_name("en")], d[context_name("ctrl")], targets, ic)}


def inputs_fingerprint(ctx, rids, variant) -> str:
    # Inputs: Ctx, recipient run ids and variant. Output: the fingerprint of every input result's
    # fingerprint and update time plus the bootstrap settings, so a rerun input makes outputs stale.
    ic = ctx.config(InterpConfig)
    stamp = lambda d: [d.get("status", {}).get("fingerprint"), d.get("provenance", {}).get("updated")]
    state = {f"{rid}/{m}": stamp(stored_result(ctx, rid, m, variant) or {}) for rid in rids for m in ("logprob", "metrics")}
    return fingerprint({"inputs": state, "n_boot": ic.n_boot, "seed": ic.seed, "alpha": ic.alpha})


def fresh(ctx, rid: str, measurement: str, rids) -> bool:
    # Inputs: Ctx, the output run id, the output measurement and the input recipient run ids. Output:
    # whether the output is complete and was computed from the current inputs.
    variant = variant_of(ctx)
    status = (stored_result(ctx, rid, measurement, variant) or {}).get("status", {})
    return bool(status.get("complete")) and status.get("fingerprint") == inputs_fingerprint(ctx, rids, variant)


def summarize(ctx, rid: str, variant) -> tuple:
    # Inputs: Ctx, recipient run id and variant. Output: (config, aggregate) with one summary per
    # available readout plus the donors, layers and identity check of the logprob measurement.
    cfg, readouts = load_readouts(ctx, rid, variant)
    ic = ctx.config(InterpConfig)
    lp = stored_result(ctx, rid, "logprob", variant)
    agg = {"donors": cfg["donors"], "layers": cfg["layers"], "n_layers": cfg["n_layers"], "targets": cfg["targets"],
           "identity_check": lp["aggregate"].get("identity_check"),
           "readouts": {name: summarize_readout(r, cfg, ic) for name, r in readouts.items()}}
    config = {"n_boot": ic.n_boot, "seed": ic.seed, "alpha": ic.alpha, "readouts": list(readouts),
              "gap": "Q_en(l) - Q_ctrl(l)", "deficit": "Q_donor(l) - Q_self", "normalization": "none"}
    return config, agg


@register_stage("interp.stitching.summary", grid=stitch_grid,
                is_done=lambda ctx, unit: fresh(ctx, unit.run_id, "summary", [unit.run_id]),
                requires=("interp.stitching.logprob", "interp.stitching.metrics"), order=903, options=OPTIONS)
def summary(ctx, unit):
    # Inputs: Ctx and a recipient Unit. Writes summary[__variant].json with the gap curves per layer
    # and readout (per target and pooled), deficits against self and the context table.
    variant = variant_of(ctx)
    config, agg = summarize(ctx, unit.run_id, variant)
    status = {"fingerprint": inputs_fingerprint(ctx, [unit.run_id], variant)}
    run_result(ctx, EXP, unit.run_id, "summary", variant, unit).write({}, agg, config=config, complete=True,
                                                                      extra={"status": status})
