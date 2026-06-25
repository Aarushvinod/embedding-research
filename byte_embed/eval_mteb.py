"""Uniform MTEB eval battery for the low-resource byte-vs-subword study (run_lowresource.py).

Every one of the 9 study languages is scored on the SAME four task types, so cross-language
comparison is clean (no "flaky" per-language task gaps):

  - **Retrieval (uniform)** : Belebele — rank a question against the pool of unique passages
                              (nDCG@10 / recall@10). 900 questions/lang, all 9 langs.
  - **Bitext (parallel)**   : FLORES-1012 — find-the-translation P@1 (lang -> English) on the
                              1012 perfectly-parallel FLORES devtest sentences (`mteb/flores`,
                              one column per language). The common, comparable semantic axis.
  - **Classification**      : SIB-200 (reuse `common.eval.sib_probe`).
  - **STS**                 : Spearman of cosine vs. human score. Routed per language to the
                              dataset that covers it (SemRel24STS / IndicCrosslingualSTS /
                              C-MTEB STS-B); all three share `sentence1/sentence2/score`. STS is
                              zero-shot (no fitting) so we concatenate ALL available splits.

MIRACL deep retrieval (en/zh/ar/te only) stays in `byte_embed/miracl.py` and is called by the
orchestrator as a bonus axis. Every `encode_fn` here is `list[str] -> np.ndarray [n, d]`.

Exact sizes (measured): SIB 204 test/701 train · FLORES 1012 · Belebele 900 · STS all-splits
en 8350/ha 2551/mr 1746/te 1573/am 1258/rw 1102/ar 627/ta 256(en-ta)/zh 1361.
"""
from __future__ import annotations

import numpy as np

from byte_embed.config import FLORES_CODE
from common.eval import l2norm, sib_probe

# STS routing: lang -> (dataset_id, config, splits-to-concat). All share sentence1/sentence2/score.
# SemRel covers en/ar/am/ha/rw/mr/te; Tamil via cross-lingual Indic (en-ta); Chinese via C-MTEB.
STS_ROUTE = {
    "en": ("mteb/SemRel24STS", "eng", ("train", "dev", "test")),
    "ar": ("mteb/SemRel24STS", "arb", ("dev", "test")),          # arb has no train split
    "am": ("mteb/SemRel24STS", "amh", ("train", "dev", "test")),
    "ha": ("mteb/SemRel24STS", "hau", ("train", "dev", "test")),
    "rw": ("mteb/SemRel24STS", "kin", ("train", "dev", "test")),
    "mr": ("mteb/SemRel24STS", "mar", ("train", "dev", "test")),
    "te": ("mteb/SemRel24STS", "tel", ("train", "dev", "test")),
    "ta": ("mteb/IndicCrosslingualSTS", "en-ta", ("test",)),      # cross-lingual en<->ta
    "zh": ("C-MTEB/STSB", "default", ("test",)),
}

_FLORES_BITEXT = "mteb/flores"   # one row per parallel sentence; one column per FLORES language


# --------------------------------------------------------------------------- STS
def _concat_splits(ds_id, cfg, splits):
    """Load + concatenate the requested splits (skip any that are absent). Returns the HF Dataset
    rows as (sentence1, sentence2, score) lists. STS is zero-shot, so using all splits is leakage-free
    and just enlarges the (otherwise small) low-resource test sets."""
    from datasets import load_dataset

    s1, s2, sc = [], [], []
    for sp in splits:
        try:
            d = load_dataset(ds_id, cfg, split=sp)
        except Exception:  # noqa: BLE001 — split absent for this config
            continue
        s1 += list(d["sentence1"]); s2 += list(d["sentence2"]); sc += list(d["score"])
    return s1, s2, [float(x) for x in sc]


def eval_sts(encode_fn, lang):
    """Spearman(cosine(emb(s1), emb(s2)), human score). Routed per language. Returns None if the
    language has no STS dataset (shouldn't happen for the study 9)."""
    from scipy.stats import spearmanr

    route = STS_ROUTE.get(lang)
    if route is None:
        return None
    s1, s2, score = _concat_splits(*route)
    if len(s1) < 10:
        return None
    a = l2norm(encode_fn(s1))
    b = l2norm(encode_fn(s2))
    cos = (a * b).sum(1)
    return {"spearman": round(float(spearmanr(cos, score).correlation), 4),
            "n": len(s1), "source": route[0].split("/")[-1]}


# --------------------------------------------------------------------------- Belebele retrieval
def _ndcg_recall(rank0, k=10):
    """Single-relevant retrieval metrics from the 0-indexed rank of the gold doc."""
    ndcg = (1.0 / np.log2(rank0 + 2.0)) if rank0 < k else 0.0   # IDCG = 1 (one relevant)
    return ndcg, float(rank0 < k), float(rank0 == 0)            # nDCG@k, recall@k, recall@1


def eval_belebele(encode_fn, lang, n_queries=None, seed=0):
    """Monolingual passage retrieval: rank each question against the pool of UNIQUE passages,
    score nDCG@10 / recall@10 / recall@1. 900 questions/lang. Returns None if unavailable."""
    from datasets import load_dataset

    fc = FLORES_CODE.get(lang)
    if fc is None:
        return None
    try:
        d = load_dataset("facebook/belebele", fc, split="test")
    except Exception:  # noqa: BLE001
        return None

    passages = sorted(set(d["flores_passage"]))
    pid = {p: i for i, p in enumerate(passages)}
    questions = list(d["question"])
    gold = np.array([pid[p] for p in d["flores_passage"]])
    if n_queries and n_queries < len(questions):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(questions), n_queries, replace=False)
        questions = [questions[i] for i in idx]; gold = gold[idx]

    Q = l2norm(encode_fn(questions))
    P = l2norm(encode_fn(passages))
    sims = Q @ P.T                                   # [nq, n_passages]
    # rank of the gold passage per query = #passages scoring strictly higher
    ranks = (sims > sims[np.arange(len(gold)), gold][:, None]).sum(1)
    ndcg, rec10, rec1 = zip(*[_ndcg_recall(int(r)) for r in ranks])
    return {"ndcg@10": round(float(np.mean(ndcg)), 4), "recall@10": round(float(np.mean(rec10)), 4),
            "recall@1": round(float(np.mean(rec1)), 4), "n_queries": len(questions),
            "n_passages": len(passages)}


# --------------------------------------------------------------------------- FLORES bitext
def _flores_devtest():
    """The 1012 parallel FLORES devtest rows (one column per FLORES language). Cached by `datasets`."""
    from datasets import load_dataset

    return load_dataset(_FLORES_BITEXT, "default", split="devtest")


def eval_flores_bitext(encode_fn, langs, pivot="en", _cache={}):
    """Find-the-translation P@1 for each language against the pivot (English) on FLORES-1012.
    Parallel + identical sentences across all languages => the clean comparable semantic axis.
    Returns {lang: {p@1, n}} for every lang != pivot (the pivot has no self-score)."""
    d = _cache.get("d") or _cache.setdefault("d", _flores_devtest())
    pcol = FLORES_CODE[pivot]
    en = l2norm(encode_fn(list(d[pcol])))
    out = {}
    for lang in langs:
        if lang == pivot:
            continue
        col = FLORES_CODE.get(lang)
        if col is None or col not in d.column_names:
            out[lang] = None
            continue
        src = l2norm(encode_fn(list(d[col])))
        p1 = float((((src @ en.T).argmax(1)) == np.arange(len(src))).mean())
        out[lang] = {"p@1": round(p1, 4), "n": len(src)}
    return out


# --------------------------------------------------------------------------- battery
def eval_battery(encode_fn, langs):
    """Run SIB (classification) + Belebele (retrieval) + STS for every language, and FLORES bitext
    across languages. Returns a structured per-task dict with per-language means. MIRACL (deep
    retrieval, en/zh/ar/te) is run separately by the orchestrator via byte_embed.miracl."""
    res = {"sib": {}, "belebele": {}, "sts": {}}
    for lang in langs:
        res["sib"][lang] = sib_probe(encode_fn, lang)
        res["belebele"][lang] = eval_belebele(encode_fn, lang)
        res["sts"][lang] = eval_sts(encode_fn, lang)
        b = res["belebele"][lang]; s = res["sts"][lang]
        print(f"    {lang}: sib={res['sib'][lang]} "
              f"belebele_ndcg={b['ndcg@10'] if b else None} "
              f"sts={s['spearman'] if s else None}")
    res["flores_bitext"] = eval_flores_bitext(encode_fn, langs)

    def _mean(d, key=None):
        vals = [(v[key] if key else v) for v in d.values() if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    res["means"] = {
        "sib": _mean(res["sib"]),
        "belebele_ndcg@10": _mean(res["belebele"], "ndcg@10"),
        "flores_p@1": _mean(res["flores_bitext"], "p@1"),
        "sts_spearman": _mean(res["sts"], "spearman"),
    }
    return res
