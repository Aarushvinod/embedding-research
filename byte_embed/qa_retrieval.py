"""QA open-retrieval eval — the RAG retriever's job: for a real question, rank the answer-bearing
passage against a relevant+distractor pool (the same rerank-pool proxy as `miracl.py`). This is the
retrieval step that bottlenecks multilingual RAG, and where byte-level embeddings win on deep retrieval.

Applied to the mteb QA-retrieval benchmarks that cover our low-resource languages:
  IndicQA (mteb/IndicQARetrieval): Marathi, Tamil, Telugu  — small corpora; the valuable NEW coverage
  Mr.TyDi (mteb/mrtidy):           Telugu                  — corroborates MIRACL te on a different corpus

All share MIRACL's `{lang}-corpus/-queries/-qrels` parquet layout, so one loader serves them. The
corpus *split* differs per dataset (IndicQA 'test', Mr.TyDi 'train', MIRACL 'dev'), so we try
candidates; a max-stream cap bounds very large corpora. AfriQA/CIRAL (Hausa) can be added here later.

  python -m byte_embed.qa_retrieval --selftest      # tiny pool build for indicqa/mr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from byte_embed.miracl import _ndcg_at_k

# benchmark -> (hf path, queries/qrels split, {our_lang: dataset_lang_config})
QA_BENCH = {
    "indicqa": ("mteb/IndicQARetrieval", "test", {"mr": "mr", "ta": "ta", "te": "te"}),
    "mrtydi":  ("mteb/mrtidy",           "test", {"te": "telugu"}),
}
_CORPUS_SPLITS = ("test", "train", "dev", "corpus")     # the corpus split varies by dataset


def _corpus_stream(path, cfg, prefer):
    from datasets import load_dataset
    for sp in (prefer, *_CORPUS_SPLITS):
        try:
            return load_dataset(path, f"{cfg}-corpus", split=sp, streaming=True)
        except Exception:  # noqa: BLE001
            continue
    return None


def _build_pool(path, cfg, split, key, n_queries, distractors, seed, cache_dir, max_stream=300000):
    """Stream the corpus once -> (queries, rel, pool_id, pool_text); cached. Returns None if unavailable
    or if no relevant docs surface within `max_stream` (bounds very large corpora)."""
    cache = Path(cache_dir) / f"qa_{key}_{n_queries}q_{distractors}d_{seed}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["queries"], {k: set(v) for k, v in d["rel"].items()}, d["pool_id"], d["pool_text"]

    from datasets import load_dataset
    try:
        q = load_dataset(path, f"{cfg}-queries", split=split)
        qr = load_dataset(path, f"{cfg}-qrels", split=split)
    except Exception as e:  # noqa: BLE001
        print(f"  [qa] {path}/{cfg} queries/qrels unavailable: {type(e).__name__}")
        return None

    qid2text = {str(r["_id"]): r["text"] for r in q}
    rel = {}
    for r in qr:
        if r.get("score", 1) > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    rng = np.random.default_rng(seed)
    qids = [qid for qid in qid2text if qid in rel]
    rng.shuffle(qids)
    qids = qids[:n_queries]
    if not qids:
        return None
    needed = set().union(*(rel[qid] for qid in qids))

    cs = _corpus_stream(path, cfg, split)
    if cs is None:
        print(f"  [qa] {path}/{cfg} corpus unavailable")
        return None
    id2text, distract, seen = {}, [], 0
    for d in cs:
        seen += 1
        did = str(d["_id"])
        txt = (str(d.get("title") or "") + " " + str(d.get("text") or "")).strip()
        if did in needed and did not in id2text:
            id2text[did] = txt
        elif len(distract) < distractors:
            distract.append((did, txt))
        if (len(id2text) == len(needed) and len(distract) >= distractors) or seen >= max_stream:
            break

    found = set(id2text)
    queries = {qid: qid2text[qid] for qid in qids if rel[qid] & found}
    rel = {qid: sorted(rel[qid] & found) for qid in queries}
    if not queries:
        print(f"  [qa] {key}: no relevant docs within {max_stream} streamed -> skip")
        return None
    pool_id = list(id2text) + [i for i, _ in distract]
    pool_text = list(id2text.values()) + [t for _, t in distract]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"queries": queries, "rel": rel, "pool_id": pool_id,
                                 "pool_text": pool_text}, ensure_ascii=False), encoding="utf-8")
    return queries, {k: set(v) for k, v in rel.items()}, pool_id, pool_text


def _eval_one(encode_fn, path, cfg, split, key, n_queries, distractors, seed, cache_dir):
    from common.eval import l2norm

    built = _build_pool(path, cfg, split, key, n_queries, distractors, seed, cache_dir)
    if not built:
        return None
    queries, rel, pool_id, pool_text = built
    qids = list(queries)
    Q = l2norm(encode_fn([queries[qid] for qid in qids]))
    P = l2norm(encode_fn(pool_text))
    docid = np.array(pool_id)
    ndcgs, recalls = [], []
    for k, qid in enumerate(qids):
        r = rel[qid]
        ranked = docid[np.argsort(-(Q[k] @ P.T))]
        rr = np.fromiter((1.0 if d in r else 0.0 for d in ranked), float, len(ranked))
        ndcgs.append(_ndcg_at_k(rr, 10))
        recalls.append(float(rr[:100].sum()) / len(r))
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4), "recall@100": round(float(np.mean(recalls)), 4),
            "n_queries": len(qids), "pool": len(pool_text)}


def eval_qa_retrieval(encode_fn, benchmarks=("indicqa", "mrtydi"), n_queries=250, distractors=20000,
                      seed=0, cache_dir="checkpoints"):
    """Per-benchmark, per-language nDCG@10 / recall@100 (+ benchmark means). The RAG-retrieval axis."""
    out = {}
    for bench in benchmarks:
        path, split, langmap = QA_BENCH[bench]
        per = {}
        for our_lang, cfg in langmap.items():
            m = _eval_one(encode_fn, path, cfg, split, f"{bench}_{our_lang}", n_queries, distractors,
                          seed, cache_dir)
            per[our_lang] = m
            if m:
                print(f"  [qa:{bench}] {our_lang}: nDCG@10={m['ndcg@10']} recall@100={m['recall@100']} "
                      f"(pool {m['pool']}, {m['n_queries']}q)")
        vals = [m["ndcg@10"] for m in per.values() if m]
        out[bench] = {"per_lang": per,
                      "ndcg@10_mean": round(float(np.mean(vals)), 4) if vals else None}
    return out


def _selftest():
    rng = np.random.default_rng(0)
    enc = lambda xs: rng.standard_normal((len(xs), 64)).astype(np.float32)  # noqa: E731
    print(eval_qa_retrieval(enc, benchmarks=("indicqa",), n_queries=20, distractors=500))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
