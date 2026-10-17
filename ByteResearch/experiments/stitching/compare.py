# ByT5-vs-mT5 comparison of the stitching English-privilege gap for one size and recipient condition.
# The two architectures have different encoder depths, so each model's gap curve Q_en(l) - Q_ctrl(l)
# is placed on the depth fraction l/L and linearly interpolated at the fractions 0, .25, .5, .75 and 1.
# The mean gap over depth is the trapezoid average of the curve over l/L.
# Both models are scored on the same FLORES devtest rows, so one bootstrap index matrix resamples the
# rows jointly for both models, every layer and (pooled) every target, and the interpolation is
# applied to each bootstrap sample to give confidence intervals for the ByT5 minus mT5 difference.
# Results go to interp_stitching/{size}__{condition}/compare[__variant].json and go stale when either
# recipient's logprob or metrics result changes.

from __future__ import annotations

import numpy as np

from ...core.config import InterpConfig, model_for
from ...core.registry import Unit, composite, product_units, register_stage
from ...core.results import run_result
from ...stats.bootstrap import boot_index, diff_stat, percentile_interval, point_corpus, point_mean
from .measure import EXP, OPTIONS, recipients, stitch_name, variant_of
from .summary import fresh, inputs_fingerprint, load_readouts, stacked

FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARCH_PAIR = ("byt5", "mt5")


def compare_grid(sel: dict, smoke: bool) -> list:
    # Inputs: CLI selections and smoke flag. Output: size x condition Units restricted to recipients.
    return [u for u in product_units(("size", "condition"), sel, smoke) if u.condition in recipients()]


def both_archs(unit) -> list:
    # Input: a (size, condition) Unit. Output: the ByT5 and mT5 recipient units of that size, whose
    # summaries the comparison needs whatever --model selected.
    return [Unit(model=model_for(a, unit.size), condition=unit.condition) for a in ARCH_PAIR]


def interp_matrix(frac, at) -> np.ndarray:
    # Inputs: increasing sample positions [m] and query positions [q]. Output: W [q, m] with
    # W @ y == np.interp(at, frac, y) for any y (linear interpolation as a matrix).
    eye = np.eye(len(frac))
    return np.stack([np.interp(at, frac, eye[j]) for j in range(len(frac))], axis=1)


def depth_weights(frac) -> np.ndarray:
    # Input: increasing sample positions [m]. Output: weights w [m] with w @ y the trapezoid average of
    # y over [frac[0], frac[-1]] (a single sample gets weight 1).
    frac = np.asarray(frac, dtype=np.float64)
    if len(frac) == 1:
        return np.ones(1)
    dx = np.diff(frac) / 2
    return (np.concatenate([dx, [0.0]]) + np.concatenate([[0.0], dx])) / (frac[-1] - frac[0])


def gap_series(r, layers, targets, idx) -> tuple:
    # Inputs: a Readout, stitch layers, the targets of this scope and the shared bootstrap index.
    # Output: (point gaps [n_layers], bootstrap gaps [n_layers, n_boot]) of en minus ctrl, macro over
    # the scope's targets with rows resampled jointly.
    points, boots = [], []
    for l in layers:
        a, b = stacked(r.data[stitch_name("en", l)], targets), stacked(r.data[stitch_name("ctrl", l)], targets)
        if r.kind == "mean" and not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"non-finite readout values at layer {l}")
        score = point_mean if r.kind == "mean" else (lambda x: point_corpus(x, r.score_fn))
        points.append(score(a) - score(b))
        boots.append(diff_stat(a, b, idx, r.kind, r.score_fn))
    return np.asarray(points), np.stack(boots)


def interval(point: float, boots: np.ndarray, ic: InterpConfig) -> dict:
    # Inputs: a point estimate, its bootstrap samples and InterpConfig. Output: {delta, ci_low, ci_high,
    # p_value, significant} with a percentile CI and a two-sided p clamped to [1/n_boot, 1].
    return {"delta": float(point), **percentile_interval(boots, ic.alpha)}


def projected(r, cfg: dict, targets, idx) -> tuple:
    # Inputs: a Readout, its logprob config, the scope's targets and the bootstrap index. Output:
    # (points [6], boots [6, n_boot]): the gap at FRACS followed by the mean gap over depth.
    frac = np.asarray(cfg["layers"], dtype=np.float64) / cfg["n_layers"]
    m = np.vstack([interp_matrix(frac, FRACS), depth_weights(frac)])
    points, boots = gap_series(r, cfg["layers"], targets, idx)
    return m @ points, m @ boots


def compare_scope(loaded: dict, name: str, targets, idx, ic: InterpConfig) -> dict:
    # Inputs: {arch: (config, readouts)}, a readout name, the scope's targets, the bootstrap index and
    # InterpConfig. Output: per fraction and for the mean over depth, each model's gap and the
    # ByT5 minus mT5 difference with bootstrap intervals.
    proj = {a: projected(loaded[a][1][name], loaded[a][0], targets, idx) for a in ARCH_PAIR}
    (pb, bb), (pm, bm) = proj["byt5"], proj["mt5"]
    rows = [{"byt5": interval(pb[i], bb[i], ic), "mt5": interval(pm[i], bm[i], ic),
             "diff": interval(pb[i] - pm[i], bb[i] - bm[i], ic)} for i in range(len(FRACS) + 1)]
    return {"fracs": [{"frac": f, **rows[i]} for i, f in enumerate(FRACS)], "mean_over_depth": rows[-1]}


def compare(ctx, unit) -> tuple:
    # Inputs: Ctx and a (size, condition) Unit. Output: (config, aggregate) comparing the ByT5 and mT5
    # recipients of that size and condition on every readout both have, per target and pooled.
    variant = variant_of(ctx)
    loaded = {u.model.split("-")[0]: load_readouts(ctx, u.run_id, variant) for u in both_archs(unit)}
    (cb, rb), (cm, rm) = loaded["byt5"], loaded["mt5"]
    if cb["rows"] != cm["rows"]:
        raise ValueError("ByT5 and mT5 stitching results cover different rows")
    targets = [t for t in cb["targets"] if t in cm["targets"]]
    ic = ctx.config(InterpConfig)
    idx = boot_index(len(cb["rows"]), ic.n_boot, ic.seed)
    scopes = {**{t: [t] for t in targets}, "pooled": targets}
    agg = {name: {"better": rb[name].better,
                  **{s: compare_scope(loaded, name, ts, idx, ic) for s, ts in scopes.items()}}
           for name in rb if name in rm}
    config = {"fracs": list(FRACS), "n_boot": ic.n_boot, "seed": ic.seed, "alpha": ic.alpha, "targets": targets,
              "layers": {"byt5": cb["layers"], "mt5": cm["layers"]},
              "n_layers": {"byt5": cb["n_layers"], "mt5": cm["n_layers"]},
              "diff": "gap_byt5 - gap_mt5 at matched depth fraction l/L", "mean": "trapezoid average over l/L"}
    return config, agg


@register_stage("interp.stitching.compare", grid=compare_grid,
                is_done=lambda ctx, unit: fresh(ctx, unit.run_id, "compare", [u.run_id for u in both_archs(unit)]),
                requires=(("interp.stitching.summary", both_archs),), order=904, options=OPTIONS)
def compare_stage(ctx, unit):
    # Inputs: Ctx and a (size, condition) Unit. Writes compare[__variant].json under {size}__{condition}.
    variant = variant_of(ctx)
    config, agg = compare(ctx, unit)
    status = {"fingerprint": inputs_fingerprint(ctx, [u.run_id for u in both_archs(unit)], variant)}
    run_result(ctx, EXP, unit.run_id, "compare", variant, unit).write({}, agg, config=config, complete=True,
                                                                      extra={"status": status})


composite("interp.stitching", ("interp.stitching.logprob", "interp.stitching.generate", "interp.stitching.metrics",
                               "interp.stitching.summary", "interp.stitching.compare"))
