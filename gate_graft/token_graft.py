"""Full GATE-GRAFT (faithful sentence-level test).

Graft a target language into a FROZEN English e5 at the INPUT-embedding level (WECHSEL-style)
so the frozen transformer composes target sentences natively -- not a bag-of-words average.

Pipeline:
  1. Bitext-free align target fastText -> English fastText (anchored) -> W, per-word gate r.
  2. Fit a monolingual lift  L_in : English fastText -> e5 INPUT-embedding space
     (target = mean of a word's WordPiece input-embeddings). Fit on English only.
  3. For each target word: grafted_row = L_in( W . fastText_target ).  Gate blend:
        row = r * grafted + (1 - r) * fallback     (fallback = e5's mean input embedding)
  4. Extend e5's tokenizer with the target words; set their input rows to the grafted rows;
     freeze the body. Encode target sentences -> the transformer composes them.
  5. Cross-lingual sentence retrieval (target -> English, OPUS-100), gated vs ungated, plus
     a sentence-confidence ranking from the gate.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import AddedToken

from common.eval import l2norm
from gate_graft import align as A
from gate_graft import gate as G

_TOK = re.compile(r"\w+", re.UNICODE)


@torch.no_grad()
def _encode(texts, tok, model, device, prefix="query: ", bs=64, max_len=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([prefix + t.lower() for t in texts[i:i + bs]], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        out.append(F.normalize(emb, dim=-1).float().cpu().numpy())
    return np.concatenate(out, 0)


def _p1(P, B):
    return float(((l2norm(P) @ l2norm(B).T).argmax(1) == np.arange(len(P))).mean())


def run(lang, align_k=15000, graft_n=15000, n_eval=400, device="cuda",
        out="results/token_graft.json"):
    from transformers import AutoModel, AutoTokenizer

    from byte_embed.data import load_parallel
    from common.data import load_fasttext_topk
    from gate_graft import config as C

    # 1) bitext-free alignment + gate
    src_words, Xs = load_fasttext_topk(lang, k=align_k)
    en_words, Xe = load_fasttext_topk("en", k=align_k)
    res = A.align(Xs, Xe, src_words=src_words, tgt_words=en_words, method="anchored")
    gate = G.reliability(Xs, Xe, res["W"])
    Xs_mapped = Xs @ res["W"]  # target fastText in English-fastText space (orthogonal W -> unit norm)
    r = gate["r"].astype(np.float32)

    # 2) frozen e5 + pristine English references (BEFORE extending the tokenizer)
    print(f"[{lang}] seed={res['seed_size']}; loading frozen e5 ...")
    tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    model = AutoModel.from_pretrained("intfloat/e5-base-v2").to(device).eval()
    orig_size = model.get_input_embeddings().weight.shape[0]
    input_np = model.get_input_embeddings().weight.detach().cpu().numpy()
    fallback = input_np.mean(0).astype(np.float32)

    par = load_parallel(lang, n=n_eval)
    if lang not in par or "en" not in par:
        print(f"[{lang}] parallel unavailable; skipping.")
        return {"lang": lang, "error": "no parallel"}
    E_en = _encode(par["en"], tok, model, device)

    # 3) fit L_in : English fastText -> e5 input-embedding space (mean of a word's subwords)
    rows, feats = [], []
    for w, fx in zip(en_words, Xe):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            rows.append(input_np[ids].mean(0))
            feats.append(fx)
    Bw, Fw = np.vstack(rows), np.vstack(feats)
    L_in = np.linalg.solve(Fw.T @ Fw + 1e-1 * np.eye(Fw.shape[1]), Fw.T @ Bw)  # [300,768]

    # 4) grafted rows (ungated + gate-blended)
    grafted = (Xs_mapped @ L_in).astype(np.float32)
    gated = r[:, None] * grafted + (1 - r[:, None]) * fallback

    # 5) extend tokenizer (lowercased to match the uncased e5 tokenizer); set new rows only
    seen, add_words, add_src = set(), [], []
    for i, w in enumerate(src_words[:graft_n]):
        wl = w.lower()
        if wl and wl not in seen and not any(c.isspace() for c in wl):
            seen.add(wl)
            add_words.append(wl)
            add_src.append(i)
    # normalized=False -> matched against raw (lowercased) text, bypassing the bert-uncased
    # accent-stripping normalizer that otherwise mangles accented / non-Latin tokens.
    tok.add_tokens([AddedToken(w, normalized=False) for w in add_words])
    model.resize_token_embeddings(len(tok))
    new_idx, graft_idx = [], []
    for wl, i in zip(add_words, add_src):
        tid = tok.convert_tokens_to_ids(wl)
        if tid >= orig_size:
            new_idx.append(tid)
            graft_idx.append(i)
    new_idx = torch.tensor(new_idx, device=model.get_input_embeddings().weight.device)
    graft_idx = np.array(graft_idx)

    def set_rows(mat):
        W = model.get_input_embeddings().weight.data
        W[new_idx] = torch.tensor(mat[graft_idx], dtype=W.dtype, device=W.device)

    set_rows(grafted)
    P_u = _encode(par[lang], tok, model, device)
    set_rows(gated)
    P_g = _encode(par[lang], tok, model, device)

    # sentence confidence (mean gate over grafted words); applied to the DEPLOYED (ungated) graft
    src_index = {w.lower(): i for i, w in enumerate(src_words)}
    Rconf = np.array([
        float(np.mean([r[src_index[t]] for t in _TOK.findall(s.lower()) if t in src_index]
                      or [0.0])) for s in par[lang]])
    correct_u = (l2norm(P_u) @ l2norm(E_en).T).argmax(1) == np.arange(len(P_u))
    order = np.argsort(-Rconf)
    cov = {f"conf_P@{int(c * 100)}%": round(
        float(correct_u[order[:max(1, int(c * len(order)))]].mean()), 3) for c in (0.25, 0.5, 1.0)}

    result = {"lang": lang, "tier": C.TIERS.get(lang, "?"), "align_k": align_k,
              "grafted_tokens": int(len(graft_idx)), "seed_size": res["seed_size"],
              "n_eval": len(par[lang]),
              "sent_p1": round(_p1(P_u, E_en), 3),               # primary: ungated graft
              "sent_p1_gateblend": round(_p1(P_g, E_en), 3),     # ablation: blend toward mean (hurts)
              **cov}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ca")
    ap.add_argument("--align-k", type=int, default=15000, dest="align_k")
    ap.add_argument("--graft-n", type=int, default=15000, dest="graft_n")
    ap.add_argument("--n-eval", type=int, default=400, dest="n_eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/token_graft.json")
    a = ap.parse_args()
    run(a.lang, align_k=a.align_k, graft_n=a.graft_n, n_eval=a.n_eval, device=a.device, out=a.out)


if __name__ == "__main__":
    main()
