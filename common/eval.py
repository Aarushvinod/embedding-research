"""Shared evaluation helpers for the byte-vs-subword study.

`l2norm` is used everywhere (retrieval scoring assumes unit vectors so dot == cosine). `sib_probe`
and `sts_probe` are optional monolingual MTEB probes (off the default retrieval battery; available
via `eval_battery(tasks=...)`).
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
    "sw": "swh_Latn", "yo": "yor_Latn",
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
