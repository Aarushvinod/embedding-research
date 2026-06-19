"""Bitext-free alignment: entropic Gromov-Wasserstein -> pseudo-dictionary -> Procrustes.

No translation pairs are used. GW matches the two spaces by their INTERNAL pairwise
similarity structure (Alvarez-Melis & Jaakkola 2018); from the resulting coupling we
read a pseudo-dictionary and fit an orthogonal map, optionally refined self-learning
style (Artetxe et al. 2018).
"""
from __future__ import annotations

import numpy as np
import ot  # POT

from common.eval import l2norm


def _cos_dist(X: np.ndarray) -> np.ndarray:
    S = l2norm(X) @ l2norm(X).T
    D = 1.0 - S
    return D / (D.max() + 1e-9)


def gw_coupling(Xs: np.ndarray, Xt: np.ndarray, epsilon: float = 5e-3,
                max_iter: int = 500) -> np.ndarray:
    """Entropic GW coupling T[n_src, n_tgt] from intra-space cosine-distance matrices."""
    Cs, Ct = _cos_dist(Xs), _cos_dist(Xt)
    p = np.full(len(Xs), 1.0 / len(Xs))
    q = np.full(len(Xt), 1.0 / len(Xt))
    T = ot.gromov.entropic_gromov_wasserstein(
        Cs, Ct, p, q, "square_loss", epsilon=epsilon, max_iter=max_iter, tol=1e-9,
    )
    return np.asarray(T)


def procrustes(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Orthogonal W minimizing ||A W - B|| (optionally weighted). Returns W [d, d]."""
    if w is not None:
        B = B * w[:, None]
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def align(Xs: np.ndarray, Xt: np.ndarray, epsilon: float = 5e-3,
          refine_iters: int = 1, csls_k: int = 10) -> dict:
    """Full bitext-free alignment. Returns W, coupling, and the GW match indices."""
    T = gw_coupling(Xs, Xt, epsilon=epsilon)
    match = T.argmax(axis=1)
    conf = T[np.arange(len(T)), match]
    W = procrustes(Xs, Xt[match], w=conf)

    # light self-learning refinement: re-mine NN matches under current W, refit
    for _ in range(max(0, refine_iters)):
        Xs_m = l2norm(Xs @ W)
        nn = (Xs_m @ l2norm(Xt).T).argmax(axis=1)
        W = procrustes(Xs, Xt[nn])
        match, conf = nn, T[np.arange(len(T)), T.argmax(axis=1)]

    return {"W": W, "coupling": T, "match": match, "match_conf": conf}
