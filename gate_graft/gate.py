"""The reliability gate: per-source-word signals of local isomorphism quality.

All signals are bitext-free and aligner-agnostic (no GW coupling required):
  - margin     : CSLS top1-top2 gap for the word's retrieval (confident match?).
  - mutual_knn : does the map preserve the word's local neighbourhood? (isomorphism).
  - cycle      : does map-there-and-back retrieve the original word?

A well-calibrated gate ranks words so that high-gate words have high BLI accuracy, letting
us trust the graft where reliable and fall back / abstain where not.
"""
from __future__ import annotations

import numpy as np

from common.eval import csls_scores, l2norm


def _rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)) / (len(x) - 1 + 1e-9)


def reliability(Xs, Xt, W, k: int = 10, weights=(0.5, 0.3, 0.2)) -> dict:
    Xs_n, Xt_n = l2norm(Xs), l2norm(Xt)
    S = csls_scores(Xs @ W, Xt, k=k)        # [n_src, n_tgt]
    match = S.argmax(1)

    # 1) retrieval margin (top1 - top2), rank-normalized to [0,1]
    top2 = np.partition(S, -2, axis=1)[:, -2:]
    margin = _rank01(top2[:, 1] - top2[:, 0])

    # 2) mutual-kNN neighbourhood preservation
    src_sim = Xs_n @ Xs_n.T
    np.fill_diagonal(src_sim, -np.inf)
    src_nn = np.argpartition(-src_sim, kth=k, axis=1)[:, :k]
    tgt_nn = np.argpartition(-S, kth=k, axis=1)[:, :k]
    mutual = np.empty(len(Xs))
    for i in range(len(Xs)):
        mutual[i] = len(set(match[src_nn[i]].tolist()) & set(tgt_nn[i].tolist())) / k

    # 3) retrieval cycle-consistency (W orthogonal -> map back with W.T)
    back = l2norm(Xt[match] @ W.T) @ Xs_n.T
    cycle = (back.argmax(1) == np.arange(len(Xs))).astype(float)

    r = weights[0] * mutual + weights[1] * margin + weights[2] * cycle
    return {"margin": margin, "mutual_knn": mutual, "cycle": cycle, "r": r}
