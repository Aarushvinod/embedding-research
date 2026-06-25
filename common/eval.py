"""Evaluation: CSLS retrieval and Bilingual Lexicon Induction (BLI).

BLI is the standard, training-free probe of cross-lingual alignment quality: after
mapping source vectors into the target space, how often is the correct translation the
nearest neighbour? We also return per-source-word correctness so the reliability gate
can be CALIBRATED against true alignment error.
"""
from __future__ import annotations

import numpy as np


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


# FLORES-200 codes for SIB-200 (massively-multilingual topic classification; MMTEB family)
SIB_CODE = {
    "en": "eng_Latn", "ca": "cat_Latn", "de": "deu_Latn", "id": "ind_Latn",
    "pl": "pol_Latn", "vi": "vie_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl",
    "bg": "bul_Cyrl", "ar": "arb_Arab", "fa": "pes_Arab", "ur": "urd_Arab",
    "hi": "hin_Deva", "mr": "mar_Deva", "ne": "npi_Deva", "bn": "ben_Beng",
    "el": "ell_Grek", "he": "heb_Hebr", "tr": "tur_Latn", "fi": "fin_Latn",
    "am": "amh_Ethi", "km": "khm_Khmr", "si": "sin_Sinh", "ka": "kat_Geor",
    # low-resource-study languages (run_lowresource.py)
    "te": "tel_Telu", "ta": "tam_Taml", "ha": "hau_Latn", "rw": "kin_Latn", "zh": "zho_Hans",
}


def sib_probe(encode_fn, lang):
    """Monolingual TARGET-language topic classification (SIB-200): embed train/test target
    sentences, fit a logistic-regression probe, return test accuracy. Tests whether the
    representation captures target-language semantics on its own — not just retrieval to
    English. Returns None if the language is unavailable."""
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression

    code = SIB_CODE.get(lang)
    if code is None:
        return None
    try:
        tr = load_dataset("Davlan/sib200", code, split="train")
        te = load_dataset("Davlan/sib200", code, split="test")
    except (ValueError, FileNotFoundError):  # config genuinely absent -> None; let net/auth/OOM raise
        return None
    Xtr, Xte = encode_fn(list(tr["text"])), encode_fn(list(te["text"]))
    clf = LogisticRegression(max_iter=2000, C=10.0).fit(Xtr, list(tr["category"]))
    return round(float(clf.score(Xte, list(te["category"]))), 3)


# STS22 monolingual tracks (MTEB) — harder, FINE-GRAINED semantic similarity (vs coarse topics)
STS22_LANGS = {"de", "ru", "ar", "en", "es", "fr", "it", "pl", "tr", "zh"}


def sts_probe(encode_fn, lang):
    """Monolingual TARGET-language STS (STS22): Spearman correlation of cosine(emb1, emb2)
    with human similarity scores. Fine-grained — separates real semantic structure from
    'topic-words translated well enough'. Returns None if unavailable."""
    if lang not in STS22_LANGS:
        return None
    from datasets import load_dataset
    from scipy.stats import spearmanr

    try:
        ds = load_dataset("mteb/sts22-crosslingual-sts", lang, split="test")
    except (ValueError, FileNotFoundError):  # config genuinely absent -> None; let net/auth/OOM raise
        return None
    a = l2norm(encode_fn(list(ds["sentence1"])))
    b = l2norm(encode_fn(list(ds["sentence2"])))
    cos = (a * b).sum(1)
    return round(float(spearmanr(cos, list(ds["score"])).correlation), 3)


def _mean_topk_sim(query: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    """For each row of `query`, mean cosine to its k nearest neighbours in `bank`.
    Inputs assumed L2-normalized so dot == cosine."""
    sims = query @ bank.T
    # partial top-k per row
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    topk = np.take_along_axis(sims, idx, axis=1)
    return topk.mean(axis=1)


def csls_scores(src: np.ndarray, tgt: np.ndarray, k: int = 10) -> np.ndarray:
    """CSLS score matrix [n_src, n_tgt] (Conneau et al. 2018), mitigates hubness."""
    src, tgt = l2norm(src), l2norm(tgt)
    r_t = _mean_topk_sim(src, tgt, k)  # source rows vs target bank
    r_s = _mean_topk_sim(tgt, src, k)  # target rows vs source bank
    sims = src @ tgt.T
    return 2.0 * sims - r_t[:, None] - r_s[None, :]


def bli(
    src_words: list[str],
    src_emb: np.ndarray,
    tgt_words: list[str],
    tgt_emb: np.ndarray,
    dictionary: dict[str, set[str]],
    csls_k: int = 10,
) -> dict:
    """Bilingual lexicon induction over source words that appear in `dictionary`.

    Returns p@1, p@5, the number of evaluated words, and `per_word_correct`
    ({src_word: bool}) for gate calibration.
    """
    tgt_index = {w: i for i, w in enumerate(tgt_words)}
    eval_rows, gold = [], []
    for i, w in enumerate(src_words):
        golds = dictionary.get(w)
        if not golds:
            continue
        gold_ids = [tgt_index[g] for g in golds if g in tgt_index]
        if gold_ids:
            eval_rows.append(i)
            gold.append(set(gold_ids))
    if not eval_rows:
        return {"p_at_1": float("nan"), "p_at_5": float("nan"), "n": 0,
                "per_word_correct": {}}

    scores = csls_scores(src_emb[eval_rows], tgt_emb, k=csls_k)
    top5 = np.argpartition(-scores, kth=min(5, scores.shape[1] - 1), axis=1)[:, :5]
    # order the 5 by score
    order = np.argsort(-np.take_along_axis(scores, top5, axis=1), axis=1)
    top5 = np.take_along_axis(top5, order, axis=1)

    per_word, hit1, hit5 = {}, 0, 0
    for r, row_i in enumerate(eval_rows):
        preds = top5[r]
        c1 = preds[0] in gold[r]
        c5 = any(p in gold[r] for p in preds)
        hit1 += int(c1)
        hit5 += int(c5)
        per_word[src_words[row_i]] = bool(c1)
    n = len(eval_rows)
    return {"p_at_1": hit1 / n, "p_at_5": hit5 / n, "n": n,
            "per_word_correct": per_word}
