"""GATE-GRAFT feasibility run (LOCAL, no training, no GPU needed).

For each target language it: (1) loads top-k fastText for the language and English,
(2) aligns them bitext-free with GW+Procrustes, (3) computes the per-word reliability
gate, (4) evaluates bilingual lexicon induction (MUSE dict, eval-only), and reports the
three feasibility signals:

    p@1 (ungated)     : does bitext-free alignment work, and decay with distance?
    gate AUC          : does the gate predict which words are correctly aligned?
    P@coverage        : if we only answer for the highest-gate words, is precision high?

A positive result: p@1 decays from `ca`->`eu`, gate AUC stays > 0.5 (ideally >0.65),
and P@25% >> P@100% (the gate buys reliable answers on a confident subset).

    python -m gate_graft.run_feasibility --langs ca tr bn eu --k 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common.data import load_fasttext_topk, load_muse_dict
from common.eval import bli, l2norm
from gate_graft import align as A
from gate_graft import config as C
from gate_graft import gate as G


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def run_language(lang: str, k: int, epsilon: float, knn: int, translit: bool = False) -> dict:
    t0 = time.time()
    src_words, Xs = load_fasttext_topk(lang, k=k)
    en_words, Xe = load_fasttext_topk("en", k=k)
    dictionary = load_muse_dict(lang, "en")  # EVAL ONLY

    res = A.align(Xs, Xe, src_words=src_words, tgt_words=en_words, method="anchored",
                  translit=translit, epsilon=epsilon, csls_k=knn)
    W = res["W"]
    gate = G.reliability(Xs, Xe, W, k=knn)

    Xs_mapped = l2norm(Xs @ W)
    b = bli(src_words, Xs_mapped, en_words, Xe, dictionary, csls_k=knn)

    out = {"lang": lang, "tier": C.TIERS.get(lang, "?"), "k": k,
           "method": res["method"], "seed_size": res["seed_size"],
           "n_eval": b["n"], "p_at_1": b["p_at_1"], "p_at_5": b["p_at_5"],
           "seconds": round(time.time() - t0, 1)}

    if b["n"] >= 10:
        idx = {w: i for i, w in enumerate(src_words)}
        ew = [w for w in b["per_word_correct"] if w in idx]
        r_eval = np.array([gate["r"][idx[w]] for w in ew])
        correct = np.array([b["per_word_correct"][w] for w in ew])
        out["gate_auc"] = round(auc(r_eval, correct), 3)
        order = np.argsort(-r_eval)
        cs = correct[order]
        out["P@25%"] = round(float(cs[: max(1, len(cs) // 4)].mean()), 3)
        out["P@50%"] = round(float(cs[: max(1, len(cs) // 2)].mean()), 3)
        out["P@100%"] = round(float(cs.mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=C.DEFAULT_LANGS)
    ap.add_argument("--k", type=int, default=C.DEFAULT_K)
    ap.add_argument("--epsilon", type=float, default=C.DEFAULT_EPSILON)
    ap.add_argument("--knn", type=int, default=C.DEFAULT_KNN)
    ap.add_argument("--translit", action="store_true",
                    help="romanize target words before seeding (helps non-Latin scripts)")
    ap.add_argument("--out", default="results/gate_graft.json")
    args = ap.parse_args()

    rows = []
    for lang in args.langs:
        print(f"\n=== {lang} ===")
        try:
            r = run_language(lang, args.k, args.epsilon, args.knn, translit=args.translit)
            rows.append(r)
            print(json.dumps(r, indent=2, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001 — feasibility script, keep going
            print(f"  FAILED for {lang}: {type(e).__name__}: {e}")
            rows.append({"lang": lang, "error": str(e)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print(f"{'lang':5} {'n':>5} {'p@1':>6} {'p@5':>6} {'gateAUC':>8} "
          f"{'P@25%':>7} {'P@100%':>7}  tier")
    for r in rows:
        if "p_at_1" not in r:
            print(f"{r.get('lang','?'):5}  (failed)")
            continue
        print(f"{r['lang']:5} {r['n_eval']:>5} {r['p_at_1']:>6.3f} {r['p_at_5']:>6.3f} "
              f"{r.get('gate_auc', float('nan')):>8} {r.get('P@25%','-'):>7} "
              f"{r.get('P@100%','-'):>7}  {r['tier']}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
