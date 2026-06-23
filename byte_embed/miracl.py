"""MIRACL passage-retrieval eval (subset) — a REAL multilingual retrieval benchmark.

Tatoeba (bitext mining) is only a *proxy* for retrieval; MIRACL is query->passage retrieval with
graded relevance — the standard multilingual retriever benchmark. The full per-language corpora
are millions of passages (too big for a Colab session), so we score over a SHARED CANDIDATE POOL
built from the dev set's positive + hard-negative passages (optionally augmented with sampled
corpus passages as extra distractors): for each query we rank the whole pool and compute
nDCG@10 / recall@100 / MRR@10 against the gold positive docids. This is an honest, session-feasible
retrieval eval (a rerank-over-hard-pool setting); the pool size is reported so the difficulty is
explicit, and `corpus_extra` makes it as hard as the session budget allows.

  python -m byte_embed.miracl --selftest        # synthetic nDCG math + (optional) tiny real load
"""
from __future__ import annotations

import numpy as np

# MIRACL's 16 "known" languages (it also has 2 surprise langs); we default to a script-diverse subset.
MIRACL_LANGS = ["ar", "bn", "en", "es", "fa", "fi", "fr", "hi",
                "id", "ja", "ko", "ru", "sw", "te", "th", "zh"]
DEFAULT_EVAL = ["ar", "bn", "en", "fr", "hi", "ru", "sw", "zh"]  # diverse scripts, overlaps our langs


def _load_miracl_dev(lang):
    """miracl/miracl dev split (queries + positive/negative passages). Uses a loading script, so it
    needs trust_remote_code on newer `datasets`; tries with then without. May be GATED -> prints a
    clear hint to `huggingface-cli login` / set HF_TOKEN rather than failing silently."""
    from datasets import load_dataset

    last = None
    for kw in ({"trust_remote_code": True}, {}):
        try:
            return load_dataset("miracl/miracl", lang, split="dev", **kw)
        except TypeError:  # this datasets version rejects trust_remote_code
            continue
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"  [miracl] dev unavailable for {lang}: {type(last).__name__}: {last}\n"
          f"          (MIRACL is gated — on Colab run `huggingface-cli login` or set HF_TOKEN, "
          f"and accept the dataset terms at https://huggingface.co/datasets/miracl/miracl)")
    return None


def _sample_corpus(lang, n, exclude, seed=0):
    """Stream n extra distractor passages from miracl/miracl-corpus[lang] (skips `exclude` docids)."""
    if n <= 0:
        return []
    from datasets import load_dataset

    out = []
    try:
        ds = load_dataset("miracl/miracl-corpus", lang, split="train", streaming=True)
        for ex in ds:
            d = ex["docid"]
            if d in exclude:
                continue
            out.append((d, (ex.get("title", "") + " " + ex.get("text", "")).strip()))
            if len(out) >= n:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  [miracl] corpus sampling unavailable for {lang} ({type(e).__name__}); "
              f"using pos+neg pool only")
    return out


def _dcg(rels):
    rels = np.asarray(rels, dtype=float)
    return float((rels / np.log2(np.arange(2, len(rels) + 2))).sum())


def _ndcg_at_k(ranked_rel, k):
    actual = _dcg(ranked_rel[:k])
    ideal = _dcg(sorted(ranked_rel, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def eval_miracl(encode_fn, lang, n_queries=250, corpus_extra=0, seed=0):
    """Rank a shared pos+neg(+sampled-corpus) pool per query; return nDCG@10 / recall@100 / MRR@10.
    `encode_fn(list[str]) -> np.ndarray` (the model's sentence encoder). Returns None if unavailable."""
    from common.eval import l2norm

    ds = _load_miracl_dev(lang)
    if ds is None:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))[:min(n_queries, len(ds))]

    queries, qrels = [], []
    pool_text, pool_docid, seen = [], [], {}

    def _add(p):
        d = p["docid"]
        if d not in seen:
            seen[d] = len(pool_text)
            pool_text.append((p.get("title", "") + " " + p.get("text", "")).strip())
            pool_docid.append(d)

    for i in order:
        row = ds[int(i)]
        pos = row.get("positive_passages") or []
        neg = row.get("negative_passages") or []
        rel = {p["docid"] for p in pos}
        if not rel:
            continue
        for p in pos:
            _add(p)
        for p in neg:
            _add(p)
        queries.append(row["query"])
        qrels.append(rel)

    if not queries:
        return None
    if corpus_extra > 0:
        for d, t in _sample_corpus(lang, corpus_extra, exclude=set(pool_docid), seed=seed):
            seen[d] = len(pool_text)
            pool_text.append(t)
            pool_docid.append(d)

    q_emb = l2norm(encode_fn(queries))
    p_emb = l2norm(encode_fn(pool_text))
    docid = np.array(pool_docid)

    ndcgs, recalls, mrrs = [], [], []
    for qi, rel in enumerate(qrels):
        ranked = docid[np.argsort(-(q_emb[qi] @ p_emb.T))]
        ranked_rel = np.fromiter((1.0 if d in rel else 0.0 for d in ranked), dtype=float,
                                 count=len(ranked))
        ndcgs.append(_ndcg_at_k(ranked_rel, 10))
        recalls.append(float(ranked_rel[:100].sum()) / len(rel))
        hit = np.flatnonzero(ranked_rel[:10])
        mrrs.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)

    return {"ndcg@10": round(float(np.mean(ndcgs)), 4),
            "recall@100": round(float(np.mean(recalls)), 4),
            "mrr@10": round(float(np.mean(mrrs)), 4),
            "n_queries": len(queries), "pool": len(pool_text)}


def eval_miracl_langs(encode_fn, langs=None, n_queries=250, corpus_extra=0, seed=0):
    """{lang: metrics} + macro means across available langs."""
    langs = langs or DEFAULT_EVAL
    per = {}
    for lang in langs:
        m = eval_miracl(encode_fn, lang, n_queries=n_queries, corpus_extra=corpus_extra, seed=seed)
        per[lang] = m
        if m:
            print(f"  [miracl] {lang}: nDCG@10={m['ndcg@10']} recall@100={m['recall@100']} "
                  f"mrr@10={m['mrr@10']} (pool {m['pool']}, {m['n_queries']} q)")
    vals = [m["ndcg@10"] for m in per.values() if m]
    rec = [m["recall@100"] for m in per.values() if m]
    return {"per_lang": per,
            "ndcg@10_mean": round(float(np.mean(vals)), 4) if vals else None,
            "recall@100_mean": round(float(np.mean(rec)), 4) if rec else None}


def _selftest():
    # nDCG sanity: perfect ranking -> 1.0; worst -> < ideal
    assert abs(_ndcg_at_k([1, 1, 0, 0], 10) - 1.0) < 1e-9
    assert _ndcg_at_k([0, 0, 1, 1], 10) < 1.0
    assert _ndcg_at_k([0, 0, 0], 10) == 0.0
    # synthetic retrieval: a "good" encoder (identity-ish) should beat random nDCG
    rng = np.random.default_rng(0)
    dim, npool = 32, 200
    P = rng.standard_normal((npool, dim))
    # query = noisy copy of a random pool row -> that row should rank top
    good_ndcg = []
    for _ in range(50):
        j = rng.integers(npool)
        q = P[j] + 0.01 * rng.standard_normal(dim)
        ranked = np.argsort(-(q @ P.T))
        rel = (ranked == j).astype(float)
        good_ndcg.append(_ndcg_at_k(rel, 10))
    assert np.mean(good_ndcg) > 0.9, np.mean(good_ndcg)
    print(f"selftest OK: synthetic good-encoder nDCG@10={np.mean(good_ndcg):.3f} (>0.9)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lang", default=None, help="if set, try a tiny real MIRACL dev load (CPU ok)")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.lang:
        ds = _load_miracl_dev(a.lang)
        if ds is not None:
            print(f"loaded miracl/{a.lang} dev: {len(ds)} queries; cols={ds.column_names}")
            r = ds[0]
            print(f"  q0='{r['query'][:60]}...' "
                  f"#pos={len(r.get('positive_passages') or [])} "
                  f"#neg={len(r.get('negative_passages') or [])}")


if __name__ == "__main__":
    main()
