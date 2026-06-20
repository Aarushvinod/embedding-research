"""Topline reference: target->en retrieval on the SAME OPUS-100 pairs, but encoded by a
MULTILINGUAL model that actually trained on these languages (multilingual-e5-base).

Contextualizes the bitext-free token-graft: how much of "a model that trained on the
language" does the graft recover, with no parallel data and a frozen English encoder?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.eval import l2norm


def run(langs, n_eval=400, device="cuda", mono=False, out="results/topline.json"):
    from sentence_transformers import SentenceTransformer

    from byte_embed.data import load_parallel

    m = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    rows = []
    for lang in langs:
        try:
            par = load_parallel(lang, n=n_eval)
            if lang not in par or "en" not in par:
                continue
            T = m.encode(["query: " + s for s in par[lang]], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
            E = m.encode(["query: " + s for s in par["en"]], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
            p1 = float(((l2norm(T) @ l2norm(E).T).argmax(1) == np.arange(len(T))).mean())
            row = {"lang": lang, "topline_p1": round(p1, 3), "n": len(par[lang])}
            if mono:
                from common.eval import sib_probe, sts_probe
                enc = lambda xs: m.encode(["query: " + x for x in xs], normalize_embeddings=True,  # noqa: E731
                                          convert_to_numpy=True, show_progress_bar=False)
                row["mono_sib"] = sib_probe(enc, lang)
                row["mono_sts"] = sts_probe(enc, lang)
            rows.append(row)
            print(f"  {lang}: topline p@1={p1:.3f}"
                  + (f"  mono_sib={row.get('mono_sib')} mono_sts={row.get('mono_sts')}"
                     if mono else ""))
        except Exception as e:  # noqa: BLE001 — one bad pair shouldn't kill the rest
            print(f"  {lang}: topline FAILED ({type(e).__name__})")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ca", "tr", "fi", "bn"])
    ap.add_argument("--mono", action="store_true", help="also run SIB-200 monolingual probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/topline.json")
    a = ap.parse_args()
    run(a.langs, device=a.device, mono=a.mono, out=a.out)


if __name__ == "__main__":
    main()
