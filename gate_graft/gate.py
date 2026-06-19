"""The reliability gate: per-source-word signals of local isomorphism quality.

For each source word we estimate how trustworthy its cross-lingual mapping is, using
only bitext-free signals. A well-calibrated gate should rank words so that high-gate
words have high BLI accuracy and low-gate words do not — letting us trust the graft
where it is reliable and fall back (or abstain) where it is not.
"""
from __future__ import annotations

import numpy as np

from common.eval import l2norm


def _rank01(x: np.ndarray) -> np.ndarray:
    """Map to [0,1] by rank (robust to scale/outliers)."""
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1 + 1e-9)


def reliability(Xs: np.ndarray, Xt: np.ndarray, W: np.ndarray, coupling: np.ndarray,
                k: int = 10, weights=(0.5, 0.3, 0.2)) -> dict:
    """Return per-source-word gate signals and a combined score r in [0,1].

    Signals (all bitext-free):
      - peakiness:      how concentrated the GW coupling row is (confident match).
      - mutual_knn:     does the map preserve the local neighbourhood? (isomorphism).
      - cycle:          does map-there-and-back retrieve the original word?
    """
    Xs_n, Xt_n = l2norm(Xs), l2norm(Xt)
    Xs_m = l2norm(Xs @ W)
    match = coupling.argmax(axis=1)

    # 1) coupling peakiness: max mass / row entropy
    row = coupling / (coupling.sum(axis=1, keepdims=True) + 1e-12)
    entropy = -(row * np.log(row + 1e-12)).sum(axis=1)
    peakiness = _rank01(-entropy)  # lower entropy -> higher reliability

    # 2) mutual-kNN neighbourhood preservation
    src_sim = Xs_n @ Xs_n.T
    np.fill_diagonal(src_sim, -np.inf)
    src_nn = np.argpartition(-src_sim, kth=k, axis=1)[:, :k]
    tgt_sim_m = Xs_m @ Xt_n.T
    tgt_nn = np.argpartition(-tgt_sim_m, kth=k, axis=1)[:, :k]
    mutual = np.empty(len(Xs))
    for i in range(len(Xs)):
        mapped_src_neighbors = set(match[src_nn[i]].tolist())
        mutual[i] = len(mapped_src_neighbors & set(tgt_nn[i].tolist())) / k

    # 3) retrieval cycle-consistency (W orthogonal -> map back with W.T)
    fwd = tgt_sim_m.argmax(axis=1)               # src -> nearest target
    back = l2norm(Xt[fwd] @ W.T) @ Xs_n.T        # target -> back to source space
    cycle = (back.argmax(axis=1) == np.arange(len(Xs))).astype(float)

    r = weights[0] * mutual + weights[1] * peakiness + weights[2] * cycle
    return {"peakiness": peakiness, "mutual_knn": mutual, "cycle": cycle, "r": r}
