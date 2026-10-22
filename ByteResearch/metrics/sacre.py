# sacreBLEU metrics: chrF++ (CHRF with word_order=2), BLEU (sacreBLEU's default 13a tokenizer) and spBLEU
# (BLEU with the flores200 SPM, the FLORES convention). All are computed on detokenized text and keep
# per-sentence sufficient statistics extracted by sacreBLEU itself, the same path its paired significance
# tests use.
# Summing the statistics over any subset of rows reproduces the corpus score of that subset, so
# corpus scores, per-target scores and bootstrap samples all come from one stats matrix.
# Sentence values are the same corpus formula applied to a single row's statistics.
# The flores200 SPM model is cached under ROOT/data/sacrebleu (the data.sacrebleu stage), and
# sacreBLEU is pointed there through corpora.assets before the tokenizer is built.

from __future__ import annotations

import numpy as np

from ..corpora.assets import use_sacrebleu_cache
from ..stats import bootstrap as B


def metric_object(ctx, name: str):
    # Inputs: Ctx and "chrf_pp", "bleu" or "spbleu". Output: the sacreBLEU metric object (CHRF word_order=2,
    # BLEU tokenize=13a or BLEU tokenize=flores200), with the SPM cache pointed at ROOT/data/sacrebleu.
    use_sacrebleu_cache(ctx)
    if name == "chrf_pp":
        return B.chrf_metric()
    return B.spbleu_metric("13a" if name == "bleu" else "flores200")


def score_fn(metric, name: str):
    # Inputs: a sacreBLEU metric object and its name. Output: the vectorised corpus score_fn([B, k])
    # -> [B] from stats.bootstrap.
    return B.chrf_score_fn(metric) if name == "chrf_pp" else B.bleu_score_fn(metric)


def sentence_scores(ctx, name: str, hyps, refs) -> dict:
    # Inputs: Ctx, metric name, hypotheses and single references. Output: {"values": per-sentence
    # scores, "stats": int stats [n, k], "signature": sacreBLEU signature, "score_fn": corpus fn}.
    metric = metric_object(ctx, name)
    stats = B.sentence_stats(metric, hyps, refs)
    fn = score_fn(metric, name)
    values = fn(stats) if len(stats) else np.zeros(0)
    return {"values": values, "stats": stats, "signature": str(metric.get_signature()), "score_fn": fn}


def corpus_from_stats(stats, fn) -> float:
    # Inputs: stats [n, k] (or [T, n, k] for a macro average) and a corpus score_fn. Output: the
    # corpus score of the summed statistics.
    return B.point_corpus(stats, fn)
