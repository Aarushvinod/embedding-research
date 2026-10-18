# Pure numpy statistics behind the LAPE language-neuron experiment (Tang et al. 2024).
# Activation probabilities are fired counts divided by non-padding positions per language, and
# LAPE is the entropy of each neuron's L1-normalised probability vector (an all-zero row is +inf).
# Selection follows Tang's identify.py: a kth-value 95th-percentile threshold, the lowest k% of all
# neurons by LAPE among neurons whose best language passes it, and assignment to every language
# above it.
# The module also holds the hub-target overlap and column-centred correlation summaries, seeded
# size-matched random neuron sets, the mean-ablation replacement values and the paired-bootstrap
# damage comparison against the baseline and the random control.
# Nothing here touches torch or files, so every function is unit-testable on small arrays.

from __future__ import annotations

import zlib

import numpy as np
from scipy.stats import rankdata

from ..core.config import LANGUAGES
from ..stats.bootstrap import boot_index, percentile_interval, point_corpus, point_mean, stat_corpus, stat_mean

ABLATION_SETS = ("shared", "en_only", "rand_shared", "rand_en_only")
RANDOM_OF = {"rand_shared": "shared", "rand_en_only": "en_only"}


def activation_probs(fired, positions) -> np.ndarray:
    # Inputs: fired counts [L, U] and non-padding position counts [L] per language. Output: the
    # activation-probability matrix P [U, L] in float64 (0 for a language with no positions).
    fired = np.asarray(fired, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    return (fired / np.where(pos > 0, pos, 1.0)[:, None]).T


def lape_entropy(P) -> np.ndarray:
    # Input: P [U, L]. Output: [U] entropy of each row after L1 normalisation (natural log, 0 log 0
    # = 0); rows that are all zero get +inf so they can never be selected.
    P = np.asarray(P, dtype=np.float64)
    tot = P.sum(axis=1, keepdims=True)
    q = P / np.where(tot > 0, tot, 1.0)
    logs = np.log(np.where(q > 0, q, 1.0))
    ent = -(q * logs).sum(axis=1)
    return np.where(tot[:, 0] > 0, ent, np.inf)


def tang_threshold(P, percentile: float = 95.0) -> float:
    # Inputs: P [U, L] and the percentile. Output: the kth smallest of all probabilities with
    # k = round(N * percentile / 100) (1-based), exactly as torch.kthvalue in Tang's identify.py.
    flat = np.sort(np.asarray(P, dtype=np.float64).ravel())
    k = max(1, round(flat.size * percentile / 100.0))
    return float(flat[k - 1])


def passes_filter(P, thr: float) -> np.ndarray:
    # Inputs: P [U, L] and the threshold. Output: bool [U], true when some language's probability
    # strictly exceeds the threshold.
    return (np.asarray(P) > thr).any(axis=1)


def n_keep(n_units: int, k_percent: float) -> int:
    # Inputs: total neuron count and k in percent. Output: Tang's selection size round(N * k / 100).
    return int(round(n_units * float(k_percent) / 100.0))


def select_neurons(lape, passes, k_percent: float) -> np.ndarray:
    # Inputs: LAPE [U], the passes-filter mask [U] and k in percent. Output: sorted neuron ids of the
    # n_keep lowest-LAPE neurons among those passing the filter (fewer when too few pass).
    ent = np.where(np.asarray(passes, dtype=bool), np.asarray(lape, dtype=np.float64), np.inf)
    order = np.argsort(ent, kind="stable")[: n_keep(ent.size, k_percent)]
    return np.sort(order[np.isfinite(ent[order])])


def assign_languages(P, ids, thr: float) -> np.ndarray:
    # Inputs: P [U, L], selected ids and the threshold. Output: bool [n_selected, L], true for every
    # language whose probability strictly exceeds the threshold.
    return np.asarray(P)[np.asarray(ids, dtype=np.int64)] > thr


def language_sets(ids, assign, langs) -> dict:
    # Inputs: selected ids, the assignment matrix and the language codes (columns). Output:
    # {lang: sorted int array of neuron ids assigned to it}.
    ids = np.asarray(ids, dtype=np.int64)
    assign = np.asarray(assign, dtype=bool).reshape(len(ids), len(langs))
    return {lang: np.sort(ids[assign[:, j]]) for j, lang in enumerate(langs)}


def layer_counts(ids, d_ff: int, n_layers: int) -> np.ndarray:
    # Inputs: flattened neuron ids (layer * d_ff + unit), d_ff and the layer count. Output: int
    # counts per layer [n_layers].
    return np.bincount(np.asarray(ids, dtype=np.int64) // int(d_ff), minlength=int(n_layers))


def different_script(hub: str, targets) -> list:
    # Inputs: a hub code and target codes. Output: the targets whose script differs from the hub's
    # (for English this drops the Latin-script Swahili, Albanian and Estonian).
    return [t for t in targets if LANGUAGES[t].script != LANGUAGES[hub].script]


def overlap_frac(a, b):
    # Inputs: the hub's neuron set A and a target's set B. Output: |A & B| / |A|, or None when A is empty.
    a = set(int(x) for x in a)
    return None if not a else len(a & set(int(x) for x in b)) / len(a)


def mean_or_none(values):
    # Input: numbers possibly containing None. Output: the mean of the non-None values, or None.
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def overlap_summary(sets: dict, hub: str, targets, n_selected: int) -> dict:
    # Inputs: {lang: ids}, a hub code, the target codes and the selected-neuron count. Output: per
    # target overlap |A&B|/|A| and chance |B|/n_selected, plus their means over all targets and over
    # the different-script targets.
    hub_set = sets.get(hub, ())
    per = {t: {"overlap": overlap_frac(hub_set, sets.get(t, ())), "size": len(sets.get(t, ())),
               "chance": len(sets.get(t, ())) / n_selected if n_selected else None} for t in targets}
    diff = different_script(hub, targets)
    return {"hub_size": len(hub_set), "per_target": per, "diff_script_targets": diff,
            "mean_all": mean_or_none(per[t]["overlap"] for t in targets),
            "mean_diff_script": mean_or_none(per[t]["overlap"] for t in diff),
            "chance_all": mean_or_none(per[t]["chance"] for t in targets),
            "chance_diff_script": mean_or_none(per[t]["chance"] for t in diff)}


def center_columns(P) -> np.ndarray:
    # Input: P [U, L]. Output: P with each neuron's mean across languages subtracted (row-wise mean
    # removal, which centres every language column on the shared per-neuron level).
    P = np.asarray(P, dtype=np.float64)
    return P - P.mean(axis=1, keepdims=True)


def _pearson_columns(X) -> np.ndarray:
    # Input: X [n, L]. Output: [L, L] Pearson correlations between columns (NaN for a constant column).
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (Xc.T @ Xc) / np.outer(norms, norms)


def cosine_matrix(X) -> np.ndarray:
    # Input: X [n, L]. Output: [L, L] cosine similarities between columns (NaN for a zero column).
    X = np.asarray(X, dtype=np.float64)
    norms = np.sqrt((X ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (X.T @ X) / np.outer(norms, norms)


def spearman_matrix(X) -> np.ndarray:
    # Input: X [n, L]. Output: [L, L] Spearman correlations between columns (Pearson of average ranks).
    return _pearson_columns(rankdata(np.asarray(X, dtype=np.float64), axis=0))


def hub_means(mat, langs, hubs, targets) -> dict:
    # Inputs: an [L, L] matrix over langs, hub codes and target codes. Output: per hub the mean
    # entry over all targets and over the different-script targets, plus the per-target values.
    ix = {c: i for i, c in enumerate(langs)}
    out = {}
    for hub in hubs:
        vals = {t: _finite(mat[ix[hub], ix[t]]) for t in targets}
        out[hub] = {"per_target": vals, "mean_all": mean_or_none(vals.values()),
                    "mean_diff_script": mean_or_none(vals[t] for t in different_script(hub, targets))}
    return out


def _finite(x):
    # Input: a number. Output: it as float, or None when it is NaN or infinite.
    x = float(x)
    return x if np.isfinite(x) else None


def matrix_json(mat) -> list:
    # Input: a 2-D array. Output: nested lists with NaN/inf entries replaced by None.
    return [[_finite(v) for v in row] for row in np.asarray(mat)]


def correlation_summary(P, langs, hubs, targets, n_layers: int, d_ff: int) -> dict:
    # Inputs: P [U, L], language codes, hub and target codes, layer count and d_ff. Output: per layer
    # and pooled Spearman and cosine [L, L] matrices of the column-centred probability profiles, with
    # the per-hub means over all and different-script targets.
    Pc = center_columns(P)
    scopes = {str(l): Pc[l * d_ff:(l + 1) * d_ff] for l in range(n_layers)}
    scopes["pooled"] = Pc
    out = {}
    for name, X in scopes.items():
        mats = {"spearman": spearman_matrix(X), "cosine": cosine_matrix(X)}
        out[name] = {kind: {"matrix": matrix_json(m), "hubs": hub_means(m, langs, hubs, targets)}
                     for kind, m in mats.items()}
    return out


def seeded_rng(seed: int, *parts) -> np.random.Generator:
    # Inputs: the base seed and any strings/numbers (e.g. target, set name, k). Output: a numpy
    # Generator seeded deterministically from them (CRC32 of each part, stable across processes).
    keys = [int(seed)] + [zlib.crc32(str(p).encode("utf-8")) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(keys))


def random_matched(ids, d_ff: int, n_layers: int, rng: np.random.Generator) -> np.ndarray:
    # Inputs: a neuron set, d_ff, layer count and a Generator. Output: sorted random neuron ids with
    # the same count in every layer, drawn without replacement from units outside the given set.
    ids = np.asarray(ids, dtype=np.int64)
    taken = set(ids.tolist())
    out = []
    for layer, count in enumerate(layer_counts(ids, d_ff, n_layers)):
        pool = np.array([u for u in range(layer * d_ff, (layer + 1) * d_ff) if u not in taken], dtype=np.int64)
        if count > pool.size:
            raise ValueError(f"layer {layer}: cannot draw {count} random units from {pool.size}")
        out.append(rng.choice(pool, size=int(count), replace=False))
    return np.sort(np.concatenate(out)) if out else ids[:0]


def ablation_sets(sets: dict, x: str, k_label: str, seed: int, d_ff: int, n_layers: int, hub: str = "en") -> dict:
    # Inputs: {lang: ids}, the target X, the k label, the seed, d_ff, layer count and the English
    # code. Output: {set name: sorted ids} for shared (X and en), en_only (en, not X) and their
    # size-matched random controls seeded from (seed, X, set, k).
    a, b = set(sets[hub].tolist()), set(sets[x].tolist())
    real = {"shared": np.array(sorted(a & b), dtype=np.int64), "en_only": np.array(sorted(a - b), dtype=np.int64)}
    rand = {r: random_matched(real[s], d_ff, n_layers, seeded_rng(seed, x, r, k_label)) for r, s in RANDOM_OF.items()}
    return {**real, **rand}


def mean_replacement(act_sum, positions, langs, exclude) -> np.ndarray:
    # Inputs: per-language activation sums [L, U], position counts [L], language codes and codes to
    # leave out (X and English). Output: [U] unweighted mean over the remaining languages of each
    # language's mean activation act_sum / positions.
    keep = [i for i, c in enumerate(langs) if c not in set(exclude)]
    act_sum = np.asarray(act_sum, dtype=np.float64)[keep]
    pos = np.asarray(positions, dtype=np.float64)[keep]
    return (act_sum / np.where(pos > 0, pos, 1.0)[:, None]).mean(axis=0)


def _interval(point, boots, alpha: float) -> dict:
    # Inputs: a point estimate, bootstrap replicates and alpha. Output: {point, ci_low, ci_high,
    # p_value} with a percentile CI (NaN replicates ignored) and a two-sided sign p-value.
    ci = percentile_interval(boots, alpha, drop_nan=True)
    return {"point": _finite(point), **{k: ci[k] for k in ("ci_low", "ci_high", "p_value")}}


def _ratio(a, b):
    # Inputs: numerator and denominator (scalars or arrays). Output: a / b with inf/NaN where b == 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)


def damage_compare(arrays: dict, kind: str = "mean", score_fn=None, n_boot: int = 10_000, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    # Inputs: {set name: per-row values [n] / [T, n] (kind "mean") or statistics [n, k] / [T, n, k]
    # (kind "corpus")} including "none", the score_fn for corpus metrics and bootstrap settings.
    # Output: per ablated set the score, damage m(set) - m(none), and for the real sets the damage
    # relative to the size-matched random set (difference and ratio), each with a paired bootstrap
    # CI that resamples rows jointly across all sets (and targets); the ratio carries no p-value.
    arrs = {s: np.asarray(a, dtype=np.float64) for s, a in arrays.items()}
    corpus = kind == "corpus"
    n = arrs["none"].shape[-2 if corpus else -1]
    idx = boot_index(n, n_boot, seed)
    point = {s: (point_corpus(a, score_fn) if corpus else point_mean(a)) for s, a in arrs.items()}
    boots = {s: (stat_corpus(a, idx, score_fn) if corpus else stat_mean(a, idx)) for s, a in arrs.items()}
    dmg = {s: (point[s] - point["none"], boots[s] - boots["none"]) for s in arrs if s != "none"}
    out = {"none": {"score": point["none"]}}
    for s, (p, b) in dmg.items():
        out[s] = {"score": point[s], "damage": _interval(p, b, alpha)}
    for r, s in RANDOM_OF.items():
        if s in dmg and r in dmg:
            out[s]["vs_rand_diff"] = _interval(dmg[s][0] - dmg[r][0], dmg[s][1] - dmg[r][1], alpha)
            ratio = _interval(_ratio(dmg[s][0], dmg[r][0]), _ratio(dmg[s][1], dmg[r][1]), alpha)
            out[s]["vs_rand_ratio"] = {**ratio, "p_value": None}
    return out
