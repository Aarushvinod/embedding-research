"""Orthographic perturbations + a stability probe.

The hypothesis: a byte-level student is more robust to non-standard orthography
(romanization, diacritic loss, spelling noise) than a subword model, because there is no
tokenizer to mis-segment. We measure stability = cosine(emb(text), emb(perturbed(text)));
higher is more robust. Comparing the byte student vs the subword teacher tests H2.
"""
from __future__ import annotations

import random
import unicodedata


def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def romanize(s: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(s)
    except Exception:  # noqa: BLE001 — fallback if unidecode not installed
        return strip_diacritics(s).encode("ascii", "ignore").decode("ascii")


def spelling_noise(s: str, p: float = 0.08, seed: int | None = None) -> str:
    rng = random.Random(seed)
    chars = list(s)
    out = []
    for c in chars:
        if c.isspace():
            out.append(c)
            continue
        roll = rng.random()
        if roll < p / 2 and out:        # transpose with previous
            out[-1], c = c, out[-1]
            out.append(c)
        elif roll < p:                  # drop
            continue
        else:
            out.append(c)
    return "".join(out)


PERTURBATIONS = {
    "diacritics": strip_diacritics,
    "romanize": romanize,
    "spelling": lambda s: spelling_noise(s, p=0.08, seed=0),
}


def stability(encode_fn, sentences, perturbation):
    """Mean cosine between clean and perturbed embeddings. `encode_fn(list)->[n,d] (np)."""
    import numpy as np

    clean = encode_fn(sentences)
    pert = encode_fn([perturbation(s) for s in sentences])
    clean = clean / (np.linalg.norm(clean, axis=1, keepdims=True) + 1e-9)
    pert = pert / (np.linalg.norm(pert, axis=1, keepdims=True) + 1e-9)
    return float((clean * pert).sum(axis=1).mean())
