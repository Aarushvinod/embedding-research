"""Orthographic perturbations + a stability probe.

The hypothesis (H2): a byte-level student is more robust to non-standard orthography
(romanization, diacritic loss, spelling/typo noise, casing, punctuation) than a subword
model, because there is no tokenizer to mis-segment. We measure
  stability = cosine(emb(text), emb(perturbed(text)));
higher = more robust. Comparing the byte student vs the subword teacher tests H2.

The perturbation suite is deliberately broad so the H2 claim is not resting on one noise
type: diacritic loss and romanization stress non-Latin/accented scripts; keyboard/spelling/
swap/delete stress within-script typos; case/punct stress surface normalization.
"""
from __future__ import annotations

import random
import unicodedata

import numpy as np

# adjacency map for QWERTY keyboard typos (a representative subset)
_QWERTY = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij",
    "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz", "d": "serfc", "f": "drtgv",
    "g": "ftyhb", "h": "gyujn", "j": "huikm", "k": "jiol", "l": "kop", "z": "asx",
    "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


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
    """Drop / transpose characters with probability ~p (a generic typo model)."""
    rng = random.Random(seed)
    out: list[str] = []
    for c in s:
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


def keyboard_typo(s: str, p: float = 0.08, seed: int | None = None) -> str:
    """Replace characters with a QWERTY-adjacent key (Latin only; others pass through)."""
    rng = random.Random(seed)
    out = []
    for c in s:
        low = c.lower()
        if low in _QWERTY and rng.random() < p:
            repl = rng.choice(_QWERTY[low])
            out.append(repl.upper() if c.isupper() else repl)
        else:
            out.append(c)
    return "".join(out)


def char_swap(s: str, p: float = 0.08, seed: int | None = None) -> str:
    """Swap adjacent non-space characters with probability ~p (script-agnostic)."""
    rng = random.Random(seed)
    chars = list(s)
    i = 0
    while i < len(chars) - 1:
        if not chars[i].isspace() and not chars[i + 1].isspace() and rng.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    return "".join(chars)


def char_delete(s: str, p: float = 0.06, seed: int | None = None) -> str:
    rng = random.Random(seed)
    return "".join(c for c in s if c.isspace() or rng.random() >= p)


def case_noise(s: str, p: float = 0.20, seed: int | None = None) -> str:
    """Randomly flip the case of letters (stresses casing normalization)."""
    rng = random.Random(seed)
    return "".join(c.swapcase() if c.isalpha() and rng.random() < p else c for c in s)


def strip_punct(s: str) -> str:
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)


# The full suite. Each is deterministic (fixed seed) so student/teacher see identical noise.
PERTURBATIONS = {
    "diacritics": strip_diacritics,
    "romanize": romanize,
    "spelling": lambda s: spelling_noise(s, p=0.08, seed=0),
    "keyboard": lambda s: keyboard_typo(s, p=0.08, seed=0),
    "swap": lambda s: char_swap(s, p=0.08, seed=0),
    "delete": lambda s: char_delete(s, p=0.06, seed=0),
    "case": lambda s: case_noise(s, p=0.20, seed=0),
    "punct": strip_punct,
}

# a smaller core set for quick ablations
CORE_PERTURBATIONS = ("diacritics", "romanize", "spelling", "keyboard")


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def stability_vec(encode_fn, sentences, perturbation) -> np.ndarray:
    """Per-sentence cosine between clean and perturbed embeddings (the raw vector, so a
    caller can bootstrap a CI or run a paired test). encode_fn(list) -> [n, d] np.ndarray."""
    clean = _l2(encode_fn(sentences))
    pert = _l2(encode_fn([perturbation(s) for s in sentences]))
    return (clean * pert).sum(axis=1)


def stability(encode_fn, sentences, perturbation) -> float:
    """Mean per-sentence clean-vs-perturbed cosine (higher = more robust)."""
    return float(stability_vec(encode_fn, sentences, perturbation).mean())
