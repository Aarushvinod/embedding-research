# Paired bootstrap significance testing and corpus metrics from sufficient statistics.
# Two systems scored on the same sentences are compared by resampling sentence ids with
# replacement (Koehn 2004).
# Mean metrics resample per-sentence values, while corpus metrics resample per-sentence
# chrF++/spBLEU statistics and recompute the corpus score for every sample.
# Pooled statistics over several targets resample FLORES rows jointly, because the rows are
# parallel across languages, by sharing one index matrix across the target axis.
# Resampling is expressed as a count matrix, so every bootstrap sample is a matrix product and no
# [n_boot, n, k] tensor is ever built.
# Corpus scores are vectorised replicas of sacreBLEU's formulas, unit-tested against sacreBLEU.

from __future__ import annotations

import numpy as np

_BLEU_ORDER = 4
_TINY = 1e-16


def boot_index(n_rows: int, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    # Inputs: number of rows, bootstrap samples and seed. Output: int index matrix [n_boot, n_rows]
    # of rows drawn with replacement.
    return np.random.default_rng(seed).integers(0, n_rows, size=(n_boot, n_rows))


def boot_counts(idx: np.ndarray, n_rows: int) -> np.ndarray:
    # Inputs: an index matrix [n_boot, m] and the row count. Output: float64 counts [n_boot, n_rows]
    # with counts[b, i] = how often row i was drawn in sample b.
    n_boot = idx.shape[0]
    flat = (idx + n_rows * np.arange(n_boot)[:, None]).ravel()
    return np.bincount(flat, minlength=n_boot * n_rows).reshape(n_boot, n_rows).astype(np.float64)


def stat_mean(per_row, idx: np.ndarray) -> np.ndarray:
    # Inputs: per-row values [n] or [T, n] (T targets sharing parallel rows) and an index matrix
    # [n_boot, n]. Output: [n_boot] bootstrap means; for [T, n] the mean over targets and the
    # jointly resampled rows.
    vals = np.asarray(per_row, dtype=np.float64)
    counts = boot_counts(idx, vals.shape[-1])
    sums = counts @ np.atleast_2d(vals).T
    return sums.mean(axis=1) / idx.shape[1]


def stat_corpus(stats, idx: np.ndarray, score_fn) -> np.ndarray:
    # Inputs: per-row sufficient statistics [n, k] or [T, n, k], an index matrix [n_boot, n] and a
    # vectorised score_fn([B, k]) -> [B]. Output: [n_boot] corpus scores (macro-averaged over T).
    arr = np.asarray(stats, dtype=np.float64)
    arr = arr[None] if arr.ndim == 2 else arr
    counts = boot_counts(idx, arr.shape[1])
    return np.mean([score_fn(counts @ arr[t]) for t in range(arr.shape[0])], axis=0)


def point_mean(per_row) -> float:
    # Input: per-row values [n] or [T, n]. Output: their overall mean (mean over targets of row means).
    return float(np.asarray(per_row, dtype=np.float64).mean())


def point_corpus(stats, score_fn) -> float:
    # Inputs: statistics [n, k] or [T, n, k] and score_fn. Output: the corpus score on all rows
    # (macro-averaged over T).
    arr = np.asarray(stats, dtype=np.float64)
    arr = arr[None] if arr.ndim == 2 else arr
    return float(np.mean(score_fn(arr.sum(axis=1))))


def diff_stat(a, b, idx: np.ndarray, kind: str = "mean", score_fn=None) -> np.ndarray:
    # Inputs: paired per-row data a and b (values or stats, same shape), an index matrix, the kind
    # ("mean" or "corpus") and score_fn for corpus. Output: [n_boot] bootstrap differences a - b.
    if kind == "mean":
        return stat_mean(a, idx) - stat_mean(b, idx)
    return stat_corpus(a, idx, score_fn) - stat_corpus(b, idx, score_fn)


def _align(a, b):
    # Inputs: two dicts {id: value-or-stats} or two arrays. Output: two arrays over the sorted
    # shared ids (dicts) or the arrays unchanged; raises ValueError on mismatched shapes.
    if isinstance(a, dict):
        keys = sorted(set(a) & set(b))
        if not keys:
            raise ValueError("no shared ids to pair")
        a, b = np.array([a[k] for k in keys], dtype=np.float64), np.array([b[k] for k in keys], dtype=np.float64)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired data must have equal shapes, got {a.shape} and {b.shape}")
    return a, b


def paired_bootstrap(a, b, kind: str = "mean", score_fn=None, n_boot: int = 10_000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    # Inputs: paired data a and b (dicts keyed by sentence id, or arrays: values [n]/[T, n] for
    # kind="mean", statistics [n, k]/[T, n, k] for kind="corpus"), score_fn for corpus, number of
    # samples, seed and alpha. Output: {delta, ci_low, ci_high, p_value, a_score, b_score, n,
    # significant} for a - b, with a percentile CI and a two-sided p clamped to [1/n_boot, 1].
    if kind not in ("mean", "corpus") or (kind == "corpus" and score_fn is None):
        raise ValueError("kind must be 'mean', or 'corpus' with a score_fn")
    a, b = _align(a, b)
    n = a.shape[1] if (kind == "mean" and a.ndim == 2) or (kind == "corpus" and a.ndim == 3) else a.shape[0]
    idx = boot_index(n, n_boot, seed)
    boots = diff_stat(a, b, idx, kind, score_fn)
    point = point_mean if kind == "mean" else (lambda x: point_corpus(x, score_fn))
    a_score, b_score = point(a), point(b)
    return {"delta": a_score - b_score, **percentile_interval(boots, alpha), "a_score": a_score,
            "b_score": b_score, "n": int(n)}


def percentile_interval(boots, alpha: float = 0.05, drop_nan: bool = False) -> dict:
    # Inputs: bootstrap replicates of a statistic, alpha and whether to ignore non-finite replicates.
    # Output: {ci_low, ci_high, p_value, significant}: the percentile CI and the two-sided sign
    # p-value 2*min(P(d<=0), P(d>=0)) clamped to [1/n_boot, 1] (all None when no replicate is left).
    b = np.asarray(boots, dtype=np.float64)
    b = b[np.isfinite(b)] if drop_nan else b
    if b.size == 0:
        return {"ci_low": None, "ci_high": None, "p_value": None, "significant": None}
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(float((b <= 0).mean()), float((b >= 0).mean()))
    return {"ci_low": float(lo), "ci_high": float(hi), "p_value": min(1.0, max(p, 1.0 / b.size)),
            "significant": bool(lo > 0 or hi < 0)}


def chrf_metric():
    # Output: the sacreBLEU chrF++ metric object (CHRF with word_order=2).
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2)


def spbleu_metric(tokenize: str = "flores200"):
    # Input: sacreBLEU tokenizer name (default flores200 SPM). Output: the BLEU metric object.
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize=tokenize)


def sentence_stats(metric, hyps, refs) -> np.ndarray:
    # Inputs: a sacreBLEU CHRF/BLEU object, hypotheses and single references. Output: per-sentence
    # sufficient statistics [n, k] as int64 (sacreBLEU's own extraction).
    return np.asarray(metric._extract_corpus_statistics(list(hyps), [list(refs)]), dtype=np.int64)


def chrf_from_stats(sums, char_order: int = 6, word_order: int = 2, beta: float = 2.0) -> np.ndarray:
    # Inputs: summed chrF statistics [B, 3*(char_order+word_order)] (or [k]) laid out as
    # [hyp, ref, match] per order, and the metric parameters. Output: [B] corpus chrF scores in
    # [0, 100], replicating sacreBLEU's effective-order (non-eps) computation.
    s = np.atleast_2d(np.asarray(sums, dtype=np.float64))
    order = char_order + word_order
    hyp, ref, match = s[:, 0:3 * order:3], s[:, 1:3 * order:3], s[:, 2:3 * order:3]
    valid = (hyp > 0) & (ref > 0)
    prec = np.where(valid, match / np.where(hyp > 0, hyp, 1), 0.0)
    rec = np.where(valid, match / np.where(ref > 0, ref, 1), 0.0)
    eff = valid.sum(axis=1)
    avg_p = np.where(eff > 0, prec.sum(axis=1) / np.maximum(eff, 1), 0.0)
    avg_r = np.where(eff > 0, rec.sum(axis=1) / np.maximum(eff, 1), 0.0)
    f = beta ** 2
    denom = f * avg_p + avg_r
    return np.where(avg_p + avg_r > 0, 100 * (1 + f) * avg_p * avg_r / np.where(denom > 0, denom, 1), 0.0)


def bleu_from_stats(sums, smooth_method: str = "exp") -> np.ndarray:
    # Inputs: summed BLEU statistics [B, 2 + 2*4] (or [k]) laid out as [sys_len, ref_len,
    # correct_1..4, total_1..4] and the smoothing method ("exp" or "none"). Output: [B] corpus
    # BLEU scores replicating sacreBLEU's compute_bleu (max order 4, no effective order).
    if smooth_method not in ("exp", "none"):
        raise ValueError(f"unsupported smooth_method {smooth_method!r}")
    s = np.atleast_2d(np.asarray(sums, dtype=np.float64))
    sys_len, ref_len = s[:, 0], s[:, 1]
    correct, total = s[:, 2:2 + _BLEU_ORDER], s[:, 2 + _BLEU_ORDER:2 + 2 * _BLEU_ORDER]
    bp = np.where(sys_len < ref_len, np.exp(1 - ref_len / np.where(sys_len > 0, sys_len, 1)), 1.0)
    bp = np.where((sys_len < ref_len) & (sys_len <= 0), 0.0, bp)
    zero = correct == 0
    smooth = np.cumprod(np.where(zero, 2.0, 1.0), axis=1) if smooth_method == "exp" else None
    safe_total = np.where(total > 0, total, 1)
    prec = 100.0 * correct / safe_total
    if smooth is not None:
        prec = np.where(zero, 100.0 / (smooth * safe_total), prec)
    ok = (total > 0).all(axis=1) & (prec > 0).all(axis=1) & ~zero.all(axis=1)
    logs = np.log(np.where(prec > 0, prec, 1.0)).mean(axis=1)
    return np.where(ok, bp * np.exp(logs), 0.0)


def chrf_score_fn(metric=None):
    # Input: an optional sacreBLEU CHRF object (default chrF++). Output: a vectorised
    # score_fn([B, k]) -> [B] for paired_bootstrap(kind="corpus").
    m = metric or chrf_metric()
    if m.eps_smoothing:
        raise ValueError("eps_smoothing chrF is not supported")
    return lambda sums: chrf_from_stats(sums, m.char_order, m.word_order, m.beta)


def bleu_score_fn(metric=None):
    # Input: an optional sacreBLEU BLEU object (default: spBLEU's corpus settings, exp smoothing and
    # max order 4, which need no tokenizer). Output: a vectorised score_fn([B, k]) -> [B] for
    # paired_bootstrap(kind="corpus").
    if metric is None:
        return lambda sums: bleu_from_stats(sums, "exp")
    if metric.effective_order or metric.max_ngram_order != _BLEU_ORDER:
        raise ValueError("only max order 4 without effective order is supported")
    return lambda sums: bleu_from_stats(sums, metric.smooth_method)


CORPUS_SCORE_FNS = {"chrf_pp": chrf_score_fn, "bleu": bleu_score_fn, "spbleu": bleu_score_fn}
