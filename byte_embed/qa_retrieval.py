"""QA open-retrieval eval — the RAG retriever's job: for a real question, rank the answer-bearing
passage against a relevant+distractor pool (the same rerank-pool proxy as `miracl.py`). This is the
retrieval step that bottlenecks multilingual RAG, and where byte-level embeddings win on deep retrieval.

Two dataset shapes are supported, both reduced to one scoring path (`_score_pool`):

* mteb `{cfg}-corpus/-queries/-qrels` layout (`QA_BENCH`, built by `_build_pool`):
    IndicQA (mteb/IndicQARetrieval): Marathi, Tamil, Telugu  — small corpora; the valuable NEW coverage
    Mr.TyDi (mteb/mrtidy):           Telugu                  — corroborates MIRACL te on a different corpus
  The corpus *split* differs per dataset (IndicQA 'test', Mr.TyDi 'train', MIRACL 'dev'), so we try
  candidates; a max-stream cap bounds very large corpora.

* flat (query, positive-passage) tables (`QA_FLAT`, built by `_build_pool_flat`):
    Amharic-Passage-Retrieval (rasyosef/...-V2): Amharic — monolingual, community-built from news.
    The corpus is the passages themselves (relevant + distractors), no separate corpus/qrels needed.

* inverted-relevance collections (`QA_INV`, built by `_build_pool_inv`):
    2AIRTC (rasyosef/2AIRTC-...): Amharic — the peer-reviewed Amharic ad-hoc IR test collection
    (240 topics, 12,587 docs). The HF dump stores relevance inverted (each doc lists the topics it is
    relevant to), so topics=queries + qrels + corpus are reconstructed from one split. This is the
    FORMAL Amharic benchmark; Amharic-PR corroborates it on a second (community) corpus.

This gives every low-resource language a retrieval signal: Belebele (shallow, all 9, in eval_battery)
plus deep retrieval for te/ta/mr (Indic) and am (Amharic, two corpora) here, en/zh/ar/te via MIRACL.
Hausa and Kinyarwanda have no public monolingual deep-retrieval corpus yet (AfriQA ships no passages);
they rest on Belebele until the cross-lingual AfriCLIRMatrix (English queries -> African docs, covers
Hausa) is wired in from its GitHub topics/qrels.

  python -m byte_embed.qa_retrieval --selftest      # tiny pool build for indicqa + amharicpr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from byte_embed.miracl import _ndcg_at_k

# mteb-layout benchmarks: name -> (hf path, queries/qrels split, {our_lang: dataset_lang_config})
QA_BENCH = {
    "indicqa": ("mteb/IndicQARetrieval", "test", {"mr": "mr", "ta": "ta", "te": "te"}),
    "mrtydi":  ("mteb/mrtidy",           "test", {"te": "telugu"}),
}
# flat (query,passage) benchmarks: name -> spec (corpus is the passages themselves)
QA_FLAT = {
    "amharicpr": {
        "path": "rasyosef/Amharic-Passage-Retrieval-Dataset-V2", "split": "test",
        "langs": ("am",), "qid": "query_id", "pid": "passage_id",
        "qcol": "query", "pcol": "passage", "extra_split": "train",
    },
}
# inverted-relevance benchmarks: name -> spec (each document lists the topics it is relevant to;
# reconstruct topics=queries + qrels + corpus from one split). 2AIRTC = the peer-reviewed Amharic
# ad-hoc IR test collection (Yeshambel et al. 2020) — the FORMAL Amharic benchmark.
QA_INV = {
    "2airtc": {
        "path": "rasyosef/2AIRTC-Amharic-Adhoc-Information-Retrieval-Test-Collection",
        "split": "documents", "langs": ("am",), "did": "doc_no", "dtext": "doc_text",
        "tno": "relevant_topic_nos", "ttitle": "relevant_topic_titles",
    },
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


def _build_pool_flat(spec, key, n_queries, distractors, seed, cache_dir, max_corpus=400000):
    """Flat (query, positive-passage) table -> (queries, rel, pool_id, pool_text); cached (same schema
    as `_build_pool`). The corpus is the passages themselves: every sampled query's relevant passage
    plus distractor passages, topped up by streaming `extra_split` when the eval split is small."""
    cache = Path(cache_dir) / f"qa_{key}_{n_queries}q_{distractors}d_{seed}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["queries"], {k: set(v) for k, v in d["rel"].items()}, d["pool_id"], d["pool_text"]

    from datasets import load_dataset
    qid, pid, qcol, pcol = spec["qid"], spec["pid"], spec["qcol"], spec["pcol"]
    try:
        ds = load_dataset(spec["path"], split=spec["split"])
    except Exception as e:  # noqa: BLE001
        print(f"  [qa] {spec['path']} unavailable: {type(e).__name__}")
        return None

    queries, rel, id2text = {}, {}, {}
    for r in ds:
        q, p = str(r[qid]), str(r[pid])
        queries.setdefault(q, str(r[qcol]))
        rel.setdefault(q, set()).add(p)
        id2text.setdefault(p, str(r[pcol]))

    rng = np.random.default_rng(seed)
    qids = list(queries)
    rng.shuffle(qids)
    qids = qids[:n_queries]
    if not qids:
        return None
    queries = {q: queries[q] for q in qids}
    rel = {q: rel[q] for q in qids}
    needed = set().union(*rel.values())

    pool_id = list(needed)
    distract = [p for p in id2text if p not in needed]
    rng.shuffle(distract)
    pool_id += distract[:distractors]
    if spec.get("extra_split") and len(pool_id) - len(needed) < distractors:   # top up from other split
        try:
            for r in load_dataset(spec["path"], split=spec["extra_split"], streaming=True):
                p = str(r[pid])
                if p not in id2text:
                    id2text[p] = str(r[pcol])
                    pool_id.append(p)
                if len(pool_id) - len(needed) >= distractors or len(id2text) >= max_corpus:
                    break
        except Exception:  # noqa: BLE001
            pass

    pool_text = [id2text[i] for i in pool_id]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"queries": queries, "rel": {k: sorted(v) for k, v in rel.items()},
                                 "pool_id": pool_id, "pool_text": pool_text}, ensure_ascii=False),
                     encoding="utf-8")
    return queries, {k: set(v) for k, v in rel.items()}, pool_id, pool_text


def _build_pool_inv(spec, key, n_queries, distractors, seed, cache_dir):
    """Inverted-relevance collection (each document lists the topics it is relevant to) -> reconstruct
    topics=queries + qrels + corpus, then pool = relevant docs + distractor docs (same schema as the
    other builders). The whole collection is the corpus, so distractors >= |collection| ranks every doc
    (true ad-hoc IR over the full collection)."""
    cache = Path(cache_dir) / f"qa_{key}_{n_queries}q_{distractors}d_{seed}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["queries"], {k: set(v) for k, v in d["rel"].items()}, d["pool_id"], d["pool_text"]

    from datasets import load_dataset
    did, dtext, tno_k, tt_k = spec["did"], spec["dtext"], spec["tno"], spec["ttitle"]
    try:
        ds = load_dataset(spec["path"], split=spec["split"])
    except Exception as e:  # noqa: BLE001
        print(f"  [qa] {spec['path']} unavailable: {type(e).__name__}")
        return None

    corpus, topics, qrels = {}, {}, {}
    for r in ds:
        dno = str(r[did])
        corpus[dno] = str(r[dtext])
        for tno, tt in zip(r.get(tno_k) or [], r.get(tt_k) or []):
            t = str(int(tno))
            topics.setdefault(t, str(tt))
            qrels.setdefault(t, set()).add(dno)

    rng = np.random.default_rng(seed)
    tids = [t for t in topics if qrels.get(t)]
    rng.shuffle(tids)
    tids = tids[:n_queries]
    if not tids:
        return None
    queries = {t: topics[t] for t in tids}
    rel = {t: qrels[t] for t in tids}
    needed = set().union(*rel.values())

    pool_id = list(needed)
    distract = [d for d in corpus if d not in needed]
    rng.shuffle(distract)
    pool_id += distract[:distractors]
    pool_text = [corpus[d] for d in pool_id]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"queries": queries, "rel": {k: sorted(v) for k, v in rel.items()},
                                 "pool_id": pool_id, "pool_text": pool_text}, ensure_ascii=False),
                     encoding="utf-8")
    return queries, {k: set(v) for k, v in rel.items()}, pool_id, pool_text


def _score_pool(encode_fn, built):
    """Encode queries + pool, return nDCG@10 / recall@100 over the (queries, rel, pool) tuple."""
    from common.eval import l2norm

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


def eval_qa_retrieval(encode_fn, benchmarks=("indicqa", "mrtydi", "amharicpr", "2airtc"), n_queries=250,
                      distractors=20000, seed=0, cache_dir="checkpoints"):
    """Per-benchmark, per-language nDCG@10 / recall@100 (+ benchmark means). The RAG-retrieval axis.
    Handles the mteb-layout benchmarks (`QA_BENCH`), flat query->passage ones (`QA_FLAT`), and
    inverted-relevance collections (`QA_INV`, e.g. the formal Amharic 2AIRTC)."""
    out = {}
    for bench in benchmarks:
        per = {}
        if bench in QA_BENCH:
            path, split, langmap = QA_BENCH[bench]
            for our_lang, cfg in langmap.items():
                built = _build_pool(path, cfg, split, f"{bench}_{our_lang}", n_queries, distractors,
                                    seed, cache_dir)
                per[our_lang] = _score_pool(encode_fn, built) if built else None
        elif bench in QA_FLAT:
            spec = QA_FLAT[bench]
            for our_lang in spec["langs"]:
                built = _build_pool_flat(spec, f"{bench}_{our_lang}", n_queries, distractors, seed,
                                         cache_dir)
                per[our_lang] = _score_pool(encode_fn, built) if built else None
        elif bench in QA_INV:
            spec = QA_INV[bench]
            for our_lang in spec["langs"]:
                built = _build_pool_inv(spec, f"{bench}_{our_lang}", n_queries, distractors, seed,
                                        cache_dir)
                per[our_lang] = _score_pool(encode_fn, built) if built else None
        else:
            print(f"  [qa] unknown benchmark {bench!r} -> skip")
            continue
        for our_lang, m in per.items():
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
    print(eval_qa_retrieval(enc, benchmarks=("indicqa", "amharicpr"), n_queries=20, distractors=500))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
