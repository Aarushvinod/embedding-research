"""Sentence-level GATE-GRAFT: graft the bitext-free alignment into a FROZEN English
sentence encoder (e5) and test whether the reliability gate improves cross-lingual
sentence retrieval. Goes beyond word-level BLI.

Pipeline (no tokenizer surgery; the graft is a learned monolingual lift):
  1. Align target fastText -> English fastText (anchored, bitext-free)  -> W, gate r.
  2. Fit a lift L : English fastText bag-of-words -> e5 sentence space, on a DISJOINT set of
     English sentences (monolingual, no bitext).
  3. For a target sentence: average its mapped fastText word vectors (uniformly = ungated,
     or weighted by the gate r = gated), apply L -> a vector in e5's space.
  4. Retrieve against e5-encoded English sentences (FLORES, eval-only). Compare gated vs
     ungated P@1, and report precision at sentence-confidence coverage R(s).

The frozen English e5 only ever encodes English; the target language reaches e5's space
purely through the bitext-free graft.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from common.eval import l2norm
from gate_graft import align as A
from gate_graft import gate as G

_TOK = re.compile(r"\w+", re.UNICODE)


def _bow(sentences, vecs, word_index, gate_r=None):
    d = vecs.shape[1]
    out = np.zeros((len(sentences), d), dtype=np.float32)
    for si, s in enumerate(sentences):
        idxs = [word_index[t] for t in _TOK.findall(s.lower()) if t in word_index]
        if not idxs:
            continue
        V = vecs[idxs]
        if gate_r is not None:
            w = gate_r[idxs] + 1e-3
            out[si] = (V * (w / w.sum())[:, None]).sum(0)
        else:
            out[si] = V.mean(0)
    return out


def _confidence(sentences, word_index, gate_r):
    out = np.zeros(len(sentences))
    for si, s in enumerate(sentences):
        rs = [gate_r[word_index[t]] for t in _TOK.findall(s.lower()) if t in word_index]
        out[si] = float(np.mean(rs)) if rs else 0.0
    return out


def _p1(P, B):
    return float(((l2norm(P) @ l2norm(B).T).argmax(1) == np.arange(len(P))).mean())


def run(lang, align_k=10000, n_eval=400, device="cuda", out="results/sentence_eval.json"):
    from sentence_transformers import SentenceTransformer

    from byte_embed.data import load_parallel, load_sentences
    from common.data import load_fasttext_topk

    src_words, Xs = load_fasttext_topk(lang, k=align_k)
    en_words, Xe = load_fasttext_topk("en", k=align_k)
    res = A.align(Xs, Xe, src_words=src_words, tgt_words=en_words, method="anchored")
    W, gate = res["W"], G.reliability(Xs, Xe, res["W"])
    Xs_mapped = l2norm(Xs @ W)
    src_index = {w: i for i, w in enumerate(src_words)}
    en_index = {w: i for i, w in enumerate(en_words)}

    print(f"[{lang}] seed={res['seed_size']} method={res['method']}; loading frozen e5 ...")
    e5 = SentenceTransformer("intfloat/e5-base-v2", device=device)

    # --- fit the lift L on a disjoint English set (monolingual) ---
    fit_en = load_sentences(["en"], n_per_lang=1500)
    B_fit = _bow(fit_en, Xe, en_index)
    keep = np.linalg.norm(B_fit, axis=1) > 0
    fit_en = [s for s, k in zip(fit_en, keep) if k]
    B_fit = B_fit[keep]
    E_fit = e5.encode(["query: " + s for s in fit_en], normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
    d = B_fit.shape[1]
    L = np.linalg.solve(B_fit.T @ B_fit + 1e-1 * np.eye(d), B_fit.T @ E_fit)  # [300,768]

    # --- eval on FLORES parallel (eval only) ---
    par = load_parallel(lang, n=n_eval)
    if lang not in par or "en" not in par:
        print(f"[{lang}] FLORES parallel unavailable; skipping.")
        return {"lang": lang, "error": "no FLORES parallel"}
    E_en = e5.encode(["query: " + s for s in par["en"]], normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)

    P_u = _bow(par[lang], Xs_mapped, src_index) @ L
    P_g = _bow(par[lang], Xs_mapped, src_index, gate_r=gate["r"]) @ L
    Rconf = _confidence(par[lang], src_index, gate["r"])

    p1_u, p1_g = _p1(P_u, E_en), _p1(P_g, E_en)
    correct_g = (l2norm(P_g) @ l2norm(E_en).T).argmax(1) == np.arange(len(P_g))
    order = np.argsort(-Rconf)
    cov = {f"P@{int(c*100)}%": round(float(correct_g[order[:max(1, int(c * len(order)))]].mean()), 3)
           for c in (0.25, 0.5, 1.0)}

    result = {"lang": lang, "tier": __import__("gate_graft.config", fromlist=["TIERS"]).TIERS.get(lang, "?"),
              "align_k": align_k, "n_eval": len(par[lang]), "seed_size": res["seed_size"],
              "xling_sent_p1_ungated": round(p1_u, 3), "xling_sent_p1_gated": round(p1_g, 3),
              **cov}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out}")
    return result


def run_bow(lang, align_k=10000, n_eval=400, out="results/sentence_bow.json"):
    """Cleaner sentence test: gate-weighted bag-of-words retrieval DIRECTLY in the aligned
    fastText space (no e5, no lift). Isolates whether the gate helps at the sentence level
    without the lossy lift-into-e5 step. Pure numpy; no GPU."""
    from byte_embed.data import load_parallel
    from common.data import load_fasttext_topk
    from gate_graft import config as C

    src_words, Xs = load_fasttext_topk(lang, k=align_k)
    en_words, Xe = load_fasttext_topk("en", k=align_k)
    res = A.align(Xs, Xe, src_words=src_words, tgt_words=en_words, method="anchored")
    gate = G.reliability(Xs, Xe, res["W"])
    Xs_mapped = l2norm(Xs @ res["W"])
    src_index = {w: i for i, w in enumerate(src_words)}
    en_index = {w: i for i, w in enumerate(en_words)}

    par = load_parallel(lang, n=n_eval)
    if lang not in par or "en" not in par:
        print(f"[{lang}] parallel unavailable; skipping.")
        return {"lang": lang, "error": "no parallel"}

    B_en = _bow(par["en"], Xe, en_index)
    B_u = _bow(par[lang], Xs_mapped, src_index)
    B_g = _bow(par[lang], Xs_mapped, src_index, gate_r=gate["r"])
    keep = (np.linalg.norm(B_en, axis=1) > 0) & (np.linalg.norm(B_u, axis=1) > 0)
    B_en, B_u, B_g = B_en[keep], B_u[keep], B_g[keep]
    Rconf = _confidence(par[lang], src_index, gate["r"])[keep]

    p1_u, p1_g = _p1(B_u, B_en), _p1(B_g, B_en)
    correct_g = (l2norm(B_g) @ l2norm(B_en).T).argmax(1) == np.arange(len(B_g))
    order = np.argsort(-Rconf)
    cov = {f"P@{int(c * 100)}%": round(
        float(correct_g[order[:max(1, int(c * len(order)))]].mean()), 3)
        for c in (0.25, 0.5, 1.0)}
    result = {"lang": lang, "tier": C.TIERS.get(lang, "?"), "mode": "bow-fasttext",
              "align_k": align_k, "n_eval": int(keep.sum()), "seed_size": res["seed_size"],
              "xling_sent_p1_ungated": round(p1_u, 3), "xling_sent_p1_gated": round(p1_g, 3),
              **cov}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ca")
    ap.add_argument("--mode", choices=["bow", "lift"], default="bow",
                    help="bow = pure aligned-fastText retrieval (clean); lift = graft into e5")
    ap.add_argument("--align-k", type=int, default=10000, dest="align_k")
    ap.add_argument("--n-eval", type=int, default=400, dest="n_eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/sentence_eval.json")
    a = ap.parse_args()
    if a.mode == "bow":
        run_bow(a.lang, align_k=a.align_k, n_eval=a.n_eval, out=a.out)
    else:
        run(a.lang, align_k=a.align_k, n_eval=a.n_eval, device=a.device, out=a.out)


if __name__ == "__main__":
    main()
