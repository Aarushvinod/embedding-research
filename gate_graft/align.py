"""Bitext-free alignment of two fastText spaces.

Primary method (L0 — "weak orthographic anchors"): seed from IDENTICAL STRINGS shared by
both vocabularies (numerals, shared cognates/proper nouns), fit an orthogonal Procrustes
map, then refine self-learning style with CSLS-induced dictionaries (Artetxe et al. 2018).
This is robust and is what makes the near languages actually align.

Fallback method (L1 — strictly no anchors): entropic Gromov-Wasserstein structure-matching
(Alvarez-Melis & Jaakkola 2018), used when too few identical strings exist (e.g. distinct
scripts). GW is finicky and typically much weaker than the anchored path — which is itself
an informative feasibility result.
"""
from __future__ import annotations

import numpy as np
import ot  # POT

from common.eval import csls_scores, l2norm


def procrustes(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Orthogonal W minimizing ||A W - B|| (optionally weighted). Returns W [d, d]."""
    if w is not None:
        B = B * w[:, None]
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def identical_string_seed(src_words, tgt_words, translit=False, max_pairs=2000):
    """Pairs (i, j) where the surface string matches across vocabularies.

    With translit=True, both sides are romanized (unidecode) + lowercased before matching,
    so cross-script cognates / loanwords / numerals can seed the alignment for non-Latin
    scripts (e.g. Bengali).
    """
    if translit:
        from unidecode import unidecode

        def norm(w):
            return unidecode(w).lower()
    else:
        def norm(w):
            return w

    tgt_index = {}
    for j, w in enumerate(tgt_words):
        tgt_index.setdefault(norm(w), j)
    pairs, used = [], set()
    for i, w in enumerate(src_words):
        key = norm(w)
        if len(key) < 2 and not key.isdigit():  # keep numerals (good anchors), drop lone letters
            continue
        j = tgt_index.get(key)
        if j is not None and i not in used:
            pairs.append((i, j))
            used.add(i)
            if len(pairs) >= max_pairs:
                break
    return pairs


def muse_seed_pairs(src_words, tgt_words, dictionary, max_pairs=100):
    """Index pairs (i, j) from a bilingual dictionary where both words are in vocab.
    A tiny seed (~100 pairs) = weak supervision, vs the unsupervised identical-string anchors.

    Both sides are lowercased before matching: fastText `cc.*.vec` tokens are NOT lowercased
    while MUSE dictionary entries are, so a verbatim match silently drops capitalized words.
    First occurrence wins (fastText is frequency-sorted, so that keeps the most frequent form).
    """
    src_index, tgt_index = {}, {}
    for i, w in enumerate(src_words):
        src_index.setdefault(w.lower(), i)
    for j, w in enumerate(tgt_words):
        tgt_index.setdefault(w.lower(), j)
    pairs = []
    for s, tgts in dictionary.items():
        i = src_index.get(s.lower())
        if i is None:
            continue
        for t in tgts:
            j = tgt_index.get(t.lower())
            if j is not None:
                pairs.append((i, j))
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


def _self_learn(Xs, Xt, W, iters=5, csls_k=10):
    """Iterative refinement: induce a CSLS dictionary under W, refit Procrustes, repeat."""
    fwd = None
    for _ in range(iters):
        fwd = csls_scores(Xs @ W, Xt, k=csls_k).argmax(1)
        W = procrustes(Xs, Xt[fwd])
    return W, fwd


def _cos_dist(X):
    S = l2norm(X) @ l2norm(X).T
    D = 1.0 - S
    return D / (D.max() + 1e-9)


def gw_coupling(Xs, Xt, epsilon=5e-3, max_iter=500):
    p = np.full(len(Xs), 1.0 / len(Xs))
    q = np.full(len(Xt), 1.0 / len(Xt))
    T = ot.gromov.entropic_gromov_wasserstein(
        _cos_dist(Xs), _cos_dist(Xt), p, q, "square_loss",
        epsilon=epsilon, max_iter=max_iter, tol=1e-9)
    return np.asarray(T)


def align(Xs, Xt, src_words=None, tgt_words=None, method="anchored", translit=False,
          seed_pairs=None, epsilon=5e-3, iters=5, csls_k=10, min_seed=20):
    """Return {W, match, seed_size, method}. Defaults to the anchored path with GW fallback.

    If `seed_pairs` (weak supervision, ~100 dictionary pairs) is given, it seeds Procrustes
    directly (then self-learning), bypassing the unsupervised identical-string anchors.
    """
    if seed_pairs is not None and len(seed_pairs) >= min_seed:
        si = [p[0] for p in seed_pairs]
        ti = [p[1] for p in seed_pairs]
        W, fwd = _self_learn(Xs, Xt, procrustes(Xs[si], Xt[ti]), iters=iters, csls_k=csls_k)
        return {"W": W, "match": fwd, "seed_size": len(seed_pairs), "method": "seed"}
    if seed_pairs is not None:  # seed requested but too few pairs -> FAIL LOUDLY, never silently
        raise ValueError(                                    # fall through to anchored (was a bug)
            f"seed_pairs requested but only {len(seed_pairs)} usable pairs "
            f"(< min_seed={min_seed}); no usable bilingual dictionary for this language")

    if method == "anchored" and src_words is not None and tgt_words is not None:
        seed = identical_string_seed(src_words, tgt_words, translit=translit)
        if len(seed) >= min_seed:
            si = [p[0] for p in seed]
            ti = [p[1] for p in seed]
            W = procrustes(Xs[si], Xt[ti])
            W, fwd = _self_learn(Xs, Xt, W, iters=iters, csls_k=csls_k)
            return {"W": W, "match": fwd, "seed_size": len(seed), "method": "anchored"}

    # fallback: Gromov-Wasserstein (few/no anchors, e.g. distinct scripts)
    T = gw_coupling(Xs, Xt, epsilon=epsilon)
    match = T.argmax(1)
    W = procrustes(Xs, Xt[match], w=T[np.arange(len(T)), match])
    W, fwd = _self_learn(Xs, Xt, W, iters=max(1, iters // 2), csls_k=csls_k)
    return {"W": W, "match": fwd, "seed_size": 0, "method": "gw"}
