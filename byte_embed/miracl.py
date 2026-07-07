"""MIRACL passage-retrieval eval — a REAL multilingual retrieval benchmark (nDCG@10 / recall@100).

Loads from **mteb/MIRACLRetrieval**, the parquet-native MTEB reformat of MIRACL — it has NO loading
script, so (unlike `miracl/miracl`, `jinaai/miracl`, `mteb/miracl-hard-negatives`) it works on
`datasets>=4.0`, which dropped script-based datasets. No version pin, no gated-repo access needed.

Format is the standard retrieval triple per language: `{lang}-queries` (_id, text), `{lang}-qrels`
(query-id, corpus-id, score) and `{lang}-corpus` (_id, text, title; up to millions of passages).
Encoding a whole corpus per language is too heavy on a Colab budget, so we score a RERANK POOL:
for a sample of queries we rank ALL their relevant passages + a fixed sample of corpus passages as
distractors. The corpus is streamed ONCE per language and the pool is CACHED to disk, so the six
models in a run reuse it instead of re-streaming. Pool size is reported so difficulty is explicit.

  python -m byte_embed.miracl --selftest          # synthetic nDCG sanity
  python -m byte_embed.miracl --lang sw            # tiny real pool build (verifies mteb/MIRACLRetrieval)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DS = "mteb/MIRACLRetrieval"
MIRACL_LANGS = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id",
                "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]
# default eval = small/mid-corpus langs (a full streaming pass is fast), 6 scripts. Add big-corpus
# langs (en/ru/de/fr/es/ja/zh) explicitly if you want them and can spend the streaming time.
DEFAULT_EVAL = ["sw", "bn", "hi", "te", "th", "ko", "fi", "ar"]


def _dcg(rels):
    rels = np.asarray(rels, dtype=float)
    return float((rels / np.log2(np.arange(2, len(rels) + 2))).sum())


def _ndcg_at_k(ranked_rel, k):
    ideal = _dcg(sorted(ranked_rel, reverse=True)[:k])
    return _dcg(ranked_rel[:k]) / ideal if ideal > 0 else 0.0


def _encode_chunked(encode_fn, texts, chunk=8192, label=""):
    """Encode a large text list in chunks with progress prints (multi-hour full-corpus encodes would
    otherwise be silent in the job log). Small lists go through in one call."""
    if len(texts) <= 20000:
        return encode_fn(texts)
    parts = []
    for i in range(0, len(texts), chunk):
        parts.append(encode_fn(texts[i:i + chunk]))
        done = min(i + chunk, len(texts))
        if done // 100000 != i // 100000 or done == len(texts):
            print(f"    [encode{' ' + label if label else ''}] {done}/{len(texts)}")
    return np.concatenate(parts, axis=0)


def _build_pool(lang, n_queries, distractors, seed, cache_dir, full=False, corpus_cap=None):
    """Stream the corpus ONCE; return (queries{qid:text}, rel{qid:[docids-in-pool]}, pool_id, pool_text).
    Cached to disk so every model in a run reuses it. Returns None if the language is unavailable.

    Legacy mode (full=False): pool = sampled queries' relevant passages + `distractors` corpus passages.
    FULL-CORPUS mode (full=True): pool = the ENTIRE corpus (or the first `corpus_cap` passages, plus
    every relevant passage for the evaluated queries so qrels stay valid — the anchor-language cap).
    n_queries=None -> ALL dev queries with relevance (the official-protocol setting)."""
    tag = (f"full{corpus_cap or 'all'}_{n_queries or 'all'}q" if full
           else f"{n_queries}q_{distractors}d")
    cache = Path(cache_dir) / f"miracl_{lang}_{tag}_{seed}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["queries"], {k: set(v) for k, v in d["rel"].items()}, d["pool_id"], d["pool_text"]

    from datasets import load_dataset
    try:
        q = load_dataset(_DS, f"{lang}-queries", split="dev")
        qr = load_dataset(_DS, f"{lang}-qrels", split="dev")
    except Exception as e:  # noqa: BLE001
        print(f"  [miracl] {lang} queries/qrels unavailable: {type(e).__name__}: {e}")
        return None

    qid2text = {str(r["_id"]): r["text"] for r in q}
    rel = {}
    for r in qr:
        if r["score"] > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    rng = np.random.default_rng(seed)
    qids = [qid for qid in qid2text if qid in rel]
    rng.shuffle(qids)
    if n_queries:
        qids = qids[:n_queries]
    if not qids:
        return None
    needed = set().union(*(rel[qid] for qid in qids))

    id2text, distract, seen = {}, [], 0
    cap = None if not full else corpus_cap                    # full+no-cap -> whole corpus
    want_distract = float("inf") if full and cap is None else (cap if full else distractors)
    for d in load_dataset(_DS, f"{lang}-corpus", split="dev", streaming=True):
        seen += 1
        did = str(d["_id"])
        txt = (str(d.get("title") or "") + " " + str(d.get("text") or "")).strip()
        if did in needed and did not in id2text:
            id2text[did] = txt
        elif len(distract) < want_distract:
            distract.append((did, txt))
        if len(id2text) == len(needed) and len(distract) >= want_distract:
            break
        if full and seen % 1_000_000 == 0:
            print(f"    [miracl {lang}] streamed {seen} corpus rows "
                  f"({len(id2text)}/{len(needed)} relevant found)")

    found = set(id2text)
    queries = {qid: qid2text[qid] for qid in qids if rel[qid] & found}
    rel = {qid: sorted(rel[qid] & found) for qid in queries}
    pool_id = list(id2text) + [i for i, _ in distract]
    pool_text = list(id2text.values()) + [t for _, t in distract]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"queries": queries, "rel": rel,
                                 "pool_id": pool_id, "pool_text": pool_text}, ensure_ascii=False),
                     encoding="utf-8")
    return queries, {k: set(v) for k, v in rel.items()}, pool_id, pool_text


def eval_miracl(encode_fn, lang, n_queries=200, distractors=20000, seed=0, cache_dir="checkpoints",
                full=False, corpus_cap=None):
    """Rank each query's relevant passages against the pool; nDCG@10/P@10/R@10/R@100/MRR@10.
    full=True -> FULL-CORPUS retrieval (optionally capped at corpus_cap passages for the anchor
    languages) over ALL dev queries when n_queries=None — the literature-comparable protocol."""
    from common.eval import l2norm

    built = _build_pool(lang, n_queries, distractors, seed, cache_dir,
                        full=full, corpus_cap=corpus_cap)
    if not built:
        return None
    queries, rel, pool_id, pool_text = built
    qids = list(queries)
    if not qids:
        return None

    Q = l2norm(encode_fn([queries[qid] for qid in qids]))
    P = l2norm(_encode_chunked(encode_fn, pool_text, label=f"miracl-{lang}"))
    docid = np.array(pool_id)
    ndcgs, recalls, rec10s, prec10s, mrrs = [], [], [], [], []
    for k, qid in enumerate(qids):
        r = rel[qid]
        ranked = docid[np.argsort(-(Q[k] @ P.T))]
        ranked_rel = np.fromiter((1.0 if d in r else 0.0 for d in ranked), float, len(ranked))
        ndcgs.append(_ndcg_at_k(ranked_rel, 10))
        top10 = float(ranked_rel[:10].sum())
        prec10s.append(top10 / 10.0)
        rec10s.append(top10 / len(r))
        recalls.append(float(ranked_rel[:100].sum()) / len(r))
        hit = np.flatnonzero(ranked_rel[:10])
        mrrs.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4),
            "precision@10": round(float(np.mean(prec10s)), 4),
            "recall@10": round(float(np.mean(rec10s)), 4),
            "recall@100": round(float(np.mean(recalls)), 4),
            "mrr@10": round(float(np.mean(mrrs)), 4), "n_queries": len(qids), "pool": len(pool_text)}


def eval_miracl_langs(encode_fn, langs=None, n_queries=200, distractors=20000, seed=0,
                      cache_dir="checkpoints", full=False, corpus_caps=None):
    """corpus_caps (with full=True): {lang: cap-or-None}; a language absent from the dict (or mapped
    to None) uses its ENTIRE corpus; a cap keeps the pool to ~cap passages (anchor languages)."""
    langs = langs or DEFAULT_EVAL
    per = {}
    for lang in langs:
        m = eval_miracl(encode_fn, lang, n_queries=n_queries, distractors=distractors, seed=seed,
                        cache_dir=cache_dir, full=full,
                        corpus_cap=(corpus_caps or {}).get(lang))
        per[lang] = m
        if m:
            print(f"  [miracl] {lang}: nDCG@10={m['ndcg@10']} recall@100={m['recall@100']} "
                  f"mrr@10={m['mrr@10']} (pool {m['pool']}, {m['n_queries']}q)")
    vals = [m["ndcg@10"] for m in per.values() if m]
    rec = [m["recall@100"] for m in per.values() if m]
    return {"per_lang": per,
            "ndcg@10_mean": round(float(np.mean(vals)), 4) if vals else None,
            "recall@100_mean": round(float(np.mean(rec)), 4) if rec else None}


def _selftest():
    assert abs(_ndcg_at_k([1, 1, 0, 0], 10) - 1.0) < 1e-9
    assert _ndcg_at_k([0, 0, 1, 1], 10) < 1.0
    assert _ndcg_at_k([0, 0, 0], 10) == 0.0
    rng = np.random.default_rng(0)
    dim, npool, good = 32, 200, []
    P = rng.standard_normal((npool, dim))
    for _ in range(50):
        j = rng.integers(npool)
        ranked = np.argsort(-((P[j] + 0.01 * rng.standard_normal(dim)) @ P.T))
        good.append(_ndcg_at_k((ranked == j).astype(float), 10))
    assert np.mean(good) > 0.9, np.mean(good)
    print(f"selftest OK: synthetic good-encoder nDCG@10={np.mean(good):.3f} (>0.9)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--distractors", type=int, default=2000)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.lang:
        built = _build_pool(a.lang, 50, a.distractors, 0, "checkpoints")
        if built:
            queries, rel, pool_id, pool_text = built
            print(f"built mteb/MIRACLRetrieval {a.lang}: {len(queries)} queries, pool {len(pool_text)} "
                  f"(rel+{a.distractors} distractors); q0={list(queries.values())[0][:60]!r}")


if __name__ == "__main__":
    main()
