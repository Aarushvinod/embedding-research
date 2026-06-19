"""Evaluation: CSLS retrieval and Bilingual Lexicon Induction (BLI).

BLI is the standard, training-free probe of cross-lingual alignment quality: after
mapping source vectors into the target space, how often is the correct translation the
nearest neighbour? We also return per-source-word correctness so the reliability gate
can be CALIBRATED against true alignment error.
"""
from __future__ import annotations

import numpy as np


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _mean_topk_sim(query: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    """For each row of `query`, mean cosine to its k nearest neighbours in `bank`.
    Inputs assumed L2-normalized so dot == cosine."""
    sims = query @ bank.T
    # partial top-k per row
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    topk = np.take_along_axis(sims, idx, axis=1)
    return topk.mean(axis=1)


def csls_scores(src: np.ndarray, tgt: np.ndarray, k: int = 10) -> np.ndarray:
    """CSLS score matrix [n_src, n_tgt] (Conneau et al. 2018), mitigates hubness."""
    src, tgt = l2norm(src), l2norm(tgt)
    r_t = _mean_topk_sim(src, tgt, k)  # source rows vs target bank
    r_s = _mean_topk_sim(tgt, src, k)  # target rows vs source bank
    sims = src @ tgt.T
    return 2.0 * sims - r_t[:, None] - r_s[None, :]


def bli(
    src_words: list[str],
    src_emb: np.ndarray,
    tgt_words: list[str],
    tgt_emb: np.ndarray,
    dictionary: dict[str, set[str]],
    csls_k: int = 10,
) -> dict:
    """Bilingual lexicon induction over source words that appear in `dictionary`.

    Returns p@1, p@5, the number of evaluated words, and `per_word_correct`
    ({src_word: bool}) for gate calibration.
    """
    tgt_index = {w: i for i, w in enumerate(tgt_words)}
    eval_rows, gold = [], []
    for i, w in enumerate(src_words):
        golds = dictionary.get(w)
        if not golds:
            continue
        gold_ids = [tgt_index[g] for g in golds if g in tgt_index]
        if gold_ids:
            eval_rows.append(i)
            gold.append(set(gold_ids))
    if not eval_rows:
        return {"p_at_1": float("nan"), "p_at_5": float("nan"), "n": 0,
                "per_word_correct": {}}

    scores = csls_scores(src_emb[eval_rows], tgt_emb, k=csls_k)
    top5 = np.argpartition(-scores, kth=5, axis=1)[:, :5]
    # order the 5 by score
    order = np.argsort(-np.take_along_axis(scores, top5, axis=1), axis=1)
    top5 = np.take_along_axis(top5, order, axis=1)

    per_word, hit1, hit5 = {}, 0, 0
    for r, row_i in enumerate(eval_rows):
        preds = top5[r]
        c1 = preds[0] in gold[r]
        c5 = any(p in gold[r] for p in preds)
        hit1 += int(c1)
        hit5 += int(c5)
        per_word[src_words[row_i]] = bool(c1)
    n = len(eval_rows)
    return {"p_at_1": hit1 / n, "p_at_5": hit5 / n, "n": n,
            "per_word_correct": per_word}
