"""QA open-retrieval eval — the RAG retriever's job: for a real question, rank the answer-bearing
passage against a relevant+distractor pool (the same rerank-pool proxy as `miracl.py`). This is the
retrieval step that bottlenecks multilingual RAG, and where byte-level embeddings win on deep retrieval.

Four dataset shapes are supported, all reduced to one scoring path (`_score_pool`):

* mteb `{cfg}-corpus/-queries/-qrels` layout (`QA_BENCH`, built by `_build_pool`):
    Mr.TyDi (mteb/mrtidy):           Telugu + Swahili — monolingual, human TyDi questions
    IndicQA (mteb/IndicQARetrieval): Telugu           — small pool (~250), secondary only
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

* cross-lingual IR (`QA_CLIR`, built by `_build_pool_clir`) — English query -> African passage, always
  flagged as a different axis:
    CIRAL (SIGIR 2024) Test A: Hausa — the deep-research-verified African IR collection and the only
    public deep-retrieval signal for Hausa (715k news passages; never its 10-query dev split).
    AfriCLIRMatrix (Ogundepo et al. 2022): kept available (am/ha), off the default battery.

FINALIZED coverage — ONE deep benchmark per language (the one with the most passages available):
MIRACL (byte_embed/miracl.py) carries te/bn/sw/yo/en/zh/ar (49k-33M passages, monolingual);
Amharic-PR carries am (68,301 passages > 2AIRTC's 12,587); CIRAL carries ha (715k passages,
cross-lingual, flagged). So the DEFAULT here is just ("amharicpr", "ciral") — Mr.TyDi, IndicQA,
2AIRTC, and AfriCLIRMatrix stay wired for optional corroboration via benchmarks=. Belebele
(eval_battery) covers all languages shallow. Dropped from the study: rw (no deep benchmark exists —
AfriQA's gold passages are English/French text, not Kinyarwanda), ta/mr (IndicQA's ~250-doc pool is
not deep), so (CIRAL-only coverage + a Wikipedia too small to train on).

  python -m byte_embed.qa_retrieval --selftest      # tiny pool build for indicqa + amharicpr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from byte_embed.miracl import _ndcg_at_k

# mteb-layout benchmarks: name -> (hf path, queries/qrels split, {our_lang: dataset_lang_config})
QA_BENCH = {
    "indicqa": ("mteb/IndicQARetrieval", "test", {"te": "te"}),   # small pool (~250); te only (ta/mr left the study)
    "mrtydi":  ("mteb/mrtidy",           "test", {"te": "telugu", "sw": "swahili"}),
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
# cross-lingual IR benchmarks (`QA_CLIR`, built by `_build_pool_clir`): English topics (TSV) + TREC
# qrels fetched over HTTP, scored over a document corpus streamed as JSONL. Cross-lingual axis:
# English query -> African passage — a different task from the monolingual sets, always flagged.
#   ciral    — CIRAL (Adeyemi et al., SIGIR 2024): the deep-research-verified African IR collection;
#              Test A split (80-100 q/lang, shallow+deep judgments; NEVER the 10-query dev). The only
#              public deep-retrieval signal for Hausa. Corpus = real news passages (NOT the
#              machine-translated variant).
#   africlir — AfriCLIRMatrix (Ogundepo et al. 2022): kept available, off the default battery.
_AFRICLIR_GH = "https://raw.githubusercontent.com/castorini/africlirmatrix/main/test"
_CIRAL_HF = "https://huggingface.co/datasets/CIRAL/ciral/resolve/main"
QA_CLIR = {
    "ciral": {
        "topics_url": _CIRAL_HF + "/ciral-{name}/topics/topics.ciral-v1.0-{code}-test-a.tsv",
        "qrels_url":  _CIRAL_HF + "/ciral-{name}/qrels/qrels.ciral-v1.0-{code}-test-a.tsv",
        "corpus_url": "https://huggingface.co/datasets/CIRAL/ciral-corpus/resolve/main/"
                      "passages-v1.0/{name}_passages.jsonl",
        "langs": {"ha": ("hausa", "ha")},   # so/sw/yo also exist; sw/yo have monolingual MIRACL
        "did": "docid", "dtext": "text", "max_stream": 800000,     # ha corpus = 715k, full pass, cached
    },
    "africlir": {
        "topics_url": _AFRICLIR_GH + "/queries/topics.africlirmatrix-v1.0.en.{code}.tsv",
        "qrels_url":  _AFRICLIR_GH + "/qrels/qrels.africlirmatrix-v1.0.en.{code}.txt",
        "corpus_url": "https://huggingface.co/datasets/castorini/africlirmatrix/resolve/main/"
                      "africlirmatrix-v1.0-{name}/corpus.jsonl",
        "langs": {"am": ("amharic", "amh"), "ha": ("hausa", "hau")},   # our_lang -> (corpus_name, code)
    },
}
# REVERSED cross-lingual (XOR) benchmark: AfriQA gold passages — AFRICAN-language question -> ENGLISH
# gold passage (the direction a low-resource speaker querying English Wikipedia needs). Queries are
# native human questions from the masakhane GitHub release (JSONL); the pool = every gold context
# (train/dev/test) + English distractors streamed from the MIRACL en corpus. OFF the default battery —
# a viability probe for the reverse axis, on TRAINED languages only (ha/sw/yo).
QA_XOR = {
    "afriqa": {
        "url": "https://github.com/masakhane-io/afriqa/raw/main/data/gold_passages/"
               "{code}/gold_span_passages.afriqa.{code}.en.{split}.json",
        "langs": {"ha": "hau", "sw": "swa", "yo": "yor"},
        "query_splits": ("test",), "pool_splits": ("train", "dev", "test"),
        "distractor_corpus": ("mteb/MIRACLRetrieval", "en", "dev"),
    },
}
_CORPUS_SPLITS = ("test", "train", "dev", "corpus")     # the corpus split varies by dataset


def _cache_path(cache_dir, key, n_queries, distractors, seed):
    return Path(cache_dir) / f"qa_{key}_{n_queries}q_{distractors}d_{seed}.json"


def _cache_load(cache):
    """Return a cached (queries, rel, pool_id, pool_text) tuple (rel as sets), or None if not cached."""
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["queries"], {k: set(v) for k, v in d["rel"].items()}, d["pool_id"], d["pool_text"]
    return None


def _cache_save(cache, queries, rel, pool_id, pool_text):
    """Persist a built pool (rel serialized as sorted lists) and return it in-memory (rel as sets).
    `rel` may be a dict of sets or of lists — both are accepted."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"queries": queries, "rel": {k: sorted(v) for k, v in rel.items()},
                                 "pool_id": pool_id, "pool_text": pool_text}, ensure_ascii=False),
                     encoding="utf-8")
    return queries, {k: set(v) for k, v in rel.items()}, pool_id, pool_text


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
    cache = _cache_path(cache_dir, key, n_queries, distractors, seed)
    hit = _cache_load(cache)
    if hit is not None:
        return hit

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
    return _cache_save(cache, queries, rel, pool_id, pool_text)


def _build_pool_flat(spec, key, n_queries, distractors, seed, cache_dir, max_corpus=400000):
    """Flat (query, positive-passage) table -> (queries, rel, pool_id, pool_text); cached (same schema
    as `_build_pool`). The corpus is the passages themselves: every sampled query's relevant passage
    plus distractor passages, topped up by streaming `extra_split` when the eval split is small."""
    cache = _cache_path(cache_dir, key, n_queries, distractors, seed)
    hit = _cache_load(cache)
    if hit is not None:
        return hit

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
    return _cache_save(cache, queries, rel, pool_id, pool_text)


def _build_pool_inv(spec, key, n_queries, distractors, seed, cache_dir):
    """Inverted-relevance collection (each document lists the topics it is relevant to) -> reconstruct
    topics=queries + qrels + corpus, then pool = relevant docs + distractor docs (same schema as the
    other builders). The whole collection is the corpus, so distractors >= |collection| ranks every doc
    (true ad-hoc IR over the full collection)."""
    cache = _cache_path(cache_dir, key, n_queries, distractors, seed)
    hit = _cache_load(cache)
    if hit is not None:
        return hit

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
            if tno is None:
                continue
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
    return _cache_save(cache, queries, rel, pool_id, pool_text)


def _fetch_lines(url, timeout=60):
    import urllib.request
    return urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace").splitlines()


def _stream_jsonl(url, timeout=180):
    """Stream a (possibly large) JSONL over HTTP line-by-line without loading it all into memory."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:  # noqa: BLE001
                    continue


def _build_pool_clir(spec, our_lang, key, n_queries, distractors, seed, cache_dir):
    """Cross-lingual IR: English topics (TSV) + TREC qrels fetched over HTTP, scored over the corpus
    streamed as JSONL (relevant docs + distractors). Corpus field names and the stream cap come from
    the spec (`did`/`dtext`/`max_stream`; CIRAL uses docid/text, AfriCLIRMatrix id/contents). Same
    return schema as the other builders; returns None (graceful) if any source is unreachable or no
    relevant docs surface within the cap."""
    cache = _cache_path(cache_dir, key, n_queries, distractors, seed)
    hit = _cache_load(cache)
    if hit is not None:
        return hit

    name, code = spec["langs"][our_lang]
    did_k, dtext_k = spec.get("did", "id"), spec.get("dtext", "contents")
    max_stream = spec.get("max_stream", 400000)
    try:
        topics = _fetch_lines(spec["topics_url"].format(name=name, code=code))
        qrels = _fetch_lines(spec["qrels_url"].format(name=name, code=code))
    except Exception as e:  # noqa: BLE001
        print(f"  [qa] {key} topics/qrels fetch failed: {type(e).__name__}")
        return None

    qid2text = {}
    for ln in topics:
        if "\t" in ln:
            qid, txt = ln.split("\t", 1)
            qid2text[qid.strip()] = txt.strip()
    rel = {}
    for ln in qrels:                                          # TREC: qid Q0 docid rel (space-separated)
        p = ln.split()
        if len(p) >= 4 and p[3].lstrip("-").isdigit() and int(p[3]) > 0:
            rel.setdefault(p[0], set()).add(p[2])

    rng = np.random.default_rng(seed)
    qids = [q for q in qid2text if q in rel]
    rng.shuffle(qids)
    qids = qids[:n_queries]
    if not qids:
        return None
    needed = set().union(*(rel[q] for q in qids))

    id2text, distract, seen = {}, [], 0
    try:
        for d in _stream_jsonl(spec["corpus_url"].format(name=name)):
            seen += 1
            did = str(d.get(did_k))
            txt = (str(d.get("title") or "") + " " + str(d.get(dtext_k) or "")).strip()
            if did in needed and did not in id2text:
                id2text[did] = txt
            elif len(distract) < distractors:
                distract.append((did, txt))
            if (len(id2text) == len(needed) and len(distract) >= distractors) or seen >= max_stream:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  [qa] {key} corpus stream failed: {type(e).__name__}")
        return None

    found = set(id2text)
    queries = {q: qid2text[q] for q in qids if rel[q] & found}
    rel = {q: sorted(rel[q] & found) for q in queries}
    if not queries:
        print(f"  [qa] {key}: no relevant docs within {max_stream} streamed -> skip")
        return None
    pool_id = list(id2text) + [i for i, _ in distract]
    pool_text = list(id2text.values()) + [t for _, t in distract]
    return _cache_save(cache, queries, rel, pool_id, pool_text)


def _build_pool_afriqa(spec, our_lang, key, n_queries, distractors, seed, cache_dir):
    """AfriQA XOR: native African-language questions -> ENGLISH gold passages (JSONL from GitHub).
    Pool = all gold contexts across pool_splits (natural confusables) + English distractors streamed
    from the MIRACL en corpus. Same return schema as the other builders."""
    cache = _cache_path(cache_dir, key, n_queries, distractors, seed)
    hit = _cache_load(cache)
    if hit is not None:
        return hit

    code = spec["langs"][our_lang]
    ctx_id, id2text, rows = {}, {}, []
    try:
        for split in spec["pool_splits"]:
            try:
                lines = _fetch_lines(spec["url"].format(code=code, split=split))
            except Exception:  # noqa: BLE001 — a split may not exist for this language
                continue
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                txt = (str(r.get("title") or "") + " " + str(r.get("context") or "")).strip()
                pid = ctx_id.setdefault(txt, f"g{len(ctx_id)}")
                id2text[pid] = txt
                if split in spec["query_splits"]:
                    rows.append((f"{split}-{r.get('id')}", str(r.get("question_lang") or ""), pid))
    except Exception as e:  # noqa: BLE001
        print(f"  [qa] {key} fetch failed: {type(e).__name__}")
        return None
    if not rows:
        print(f"  [qa] {key}: no queries")
        return None

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:n_queries]
    queries = {qid: q for qid, q, _ in rows if q}
    rel = {qid: {pid} for qid, q, pid in rows if q}

    pool_id = list(id2text)                                   # every gold context (incl. non-sampled)
    pool_text = [id2text[i] for i in pool_id]
    path, cfg, prefer = spec["distractor_corpus"]             # top up with English distractors
    cs = _corpus_stream(path, cfg, prefer)
    if cs is not None:
        seen = 0
        for d in cs:
            seen += 1
            pool_id.append(f"d{seen}")
            pool_text.append((str(d.get("title") or "") + " " + str(d.get("text") or "")).strip())
            if seen >= distractors:
                break
    _cache_save(cache, queries, rel, pool_id, pool_text)
    return queries, rel, pool_id, pool_text


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
        if not r:                                            # no relevant doc in pool -> skip (no div/0)
            continue
        ranked = docid[np.argsort(-(Q[k] @ P.T))]
        rr = np.fromiter((1.0 if d in r else 0.0 for d in ranked), float, len(ranked))
        ndcgs.append(_ndcg_at_k(rr, 10))
        recalls.append(float(rr[:100].sum()) / len(r))
    if not ndcgs:
        return None
    return {"ndcg@10": round(float(np.mean(ndcgs)), 4), "recall@100": round(float(np.mean(recalls)), 4),
            "n_queries": len(ndcgs), "pool": len(pool_text)}


def eval_qa_retrieval(encode_fn,
                      benchmarks=("amharicpr", "ciral"),
                      n_queries=250, distractors=20000, seed=0, cache_dir="checkpoints"):
    """Per-benchmark, per-language nDCG@10 / recall@100 (+ benchmark means). The RAG-retrieval axis.
    Handles mteb-layout benchmarks (`QA_BENCH`), flat query->passage ones (`QA_FLAT`), inverted-relevance
    collections (`QA_INV`, e.g. the formal Amharic 2AIRTC), and cross-lingual IR (`QA_CLIR`, AfriCLIRMatrix
    — English query -> African passage, the only public deep-retrieval signal for Hausa). A failure in one
    (benchmark, language) degrades to None for that cell only — it never sinks the rest."""
    def safe(label, build):
        try:
            built = build()
            return _score_pool(encode_fn, built) if built else None
        except Exception as e:  # noqa: BLE001
            print(f"  [qa:{label}] failed: {type(e).__name__}: {e}")
            return None

    out = {}
    for bench in benchmarks:
        per = {}
        if bench in QA_BENCH:
            path, split, langmap = QA_BENCH[bench]
            for our_lang, cfg in langmap.items():
                per[our_lang] = safe(f"{bench}:{our_lang}", lambda c=cfg, l=our_lang: _build_pool(
                    path, c, split, f"{bench}_{l}", n_queries, distractors, seed, cache_dir))
        elif bench in QA_FLAT:
            spec = QA_FLAT[bench]
            for our_lang in spec["langs"]:
                per[our_lang] = safe(f"{bench}:{our_lang}", lambda l=our_lang: _build_pool_flat(
                    spec, f"{bench}_{l}", n_queries, distractors, seed, cache_dir))
        elif bench in QA_INV:
            spec = QA_INV[bench]
            for our_lang in spec["langs"]:
                per[our_lang] = safe(f"{bench}:{our_lang}", lambda l=our_lang: _build_pool_inv(
                    spec, f"{bench}_{l}", n_queries, distractors, seed, cache_dir))
        elif bench in QA_CLIR:
            spec = QA_CLIR[bench]
            for our_lang in spec["langs"]:
                per[our_lang] = safe(f"{bench}:{our_lang}", lambda l=our_lang: _build_pool_clir(
                    spec, l, f"{bench}_{l}", n_queries, distractors, seed, cache_dir))
        elif bench in QA_XOR:
            spec = QA_XOR[bench]
            for our_lang in spec["langs"]:
                per[our_lang] = safe(f"{bench}:{our_lang}", lambda l=our_lang: _build_pool_afriqa(
                    spec, l, f"{bench}_{l}", n_queries, distractors, seed, cache_dir))
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
