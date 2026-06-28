"""STS training + eval data for the STS-focused study (8 languages, all native graded train).

Chosen by STS-data availability (real graded sentence1/sentence2/score; train + usable test):
  am/ha/rw/mr/te/arq/en  -> mteb/SemRel24STS   (train+dev for training, test for eval)
  zh                      -> C-MTEB/STSB        (train+validation for training, test for eval)

6 low-resource (Amharic, Hausa, Kinyarwanda, Telugu, Marathi, Algerian Arabic) + 2 anchors
(English, Chinese). Scores are normalized to [0,1] per source so cross-lingual batches share a
scale (SemRel is already 0-1; STSB is 0-5 -> /5). Eval is TEST-ONLY -> leakage-safe vs the
train+dev we train on. CoSENT only uses score *ordering*, so the absolute scale is harmless either way.
"""
from __future__ import annotations

# lang -> (dataset_id, config, score_max)
STS_SOURCE = {
    "am":  ("mteb/SemRel24STS", "amh", 1.0),
    "ha":  ("mteb/SemRel24STS", "hau", 1.0),
    "rw":  ("mteb/SemRel24STS", "kin", 1.0),
    "mr":  ("mteb/SemRel24STS", "mar", 1.0),
    "te":  ("mteb/SemRel24STS", "tel", 1.0),
    "arq": ("mteb/SemRel24STS", "arq", 1.0),
    "en":  ("mteb/SemRel24STS", "eng", 1.0),
    "zh":  ("C-MTEB/STSB",      "default", 5.0),
}
STS_LANGS = list(STS_SOURCE)
LANG_NAME = {"am": "Amharic", "ha": "Hausa", "rw": "Kinyarwanda", "mr": "Marathi", "te": "Telugu",
             "arq": "Algerian Arabic", "en": "English", "zh": "Chinese"}
_TRAIN_SPLITS = ("train", "dev", "validation")    # everything that isn't test; eval is test-only


def load_sts_split(lang, which="train"):
    """(s1, s2, score in [0,1]) for one language. which='train' -> train+dev/validation; 'test' -> test."""
    from datasets import load_dataset

    ds_id, cfg, smax = STS_SOURCE[lang]
    splits = _TRAIN_SPLITS if which == "train" else ("test",)
    s1, s2, sc = [], [], []
    for sp in splits:
        try:
            d = load_dataset(ds_id, cfg, split=sp)
        except Exception:  # noqa: BLE001 — split absent for this config
            continue
        s1 += list(d["sentence1"])
        s2 += list(d["sentence2"])
        sc += [float(x) / smax for x in d["score"]]
    return s1, s2, sc


def load_sts_train(langs=None):
    """Pooled, shuffled training triples across languages -> (s1, s2, score, lang). Logs per-lang counts."""
    import random

    langs = langs or STS_LANGS
    rows = []
    for lang in langs:
        s1, s2, sc = load_sts_split(lang, "train")
        rows += list(zip(s1, s2, sc, [lang] * len(s1)))
        print(f"  [sts] {lang} ({LANG_NAME[lang]}): {len(s1)} train pairs")
    random.Random(0).shuffle(rows)
    S1, S2, SC, LG = (list(c) for c in zip(*rows)) if rows else ([], [], [], [])
    print(f"  [sts] total train pairs: {len(S1)} across {len(langs)} languages")
    return S1, S2, SC, LG
