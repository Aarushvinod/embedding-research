"""Loaders for fastText word vectors and MUSE bilingual dictionaries.

Everything is downloaded on demand from the official public CDNs and cached under
``data/``. No parallel/bitext data is used for *alignment* — the MUSE dictionaries are
loaded for EVALUATION ONLY (bilingual lexicon induction), never to fit the map.
"""
from __future__ import annotations

import gzip
import io
import os
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz"
MUSE_DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.txt"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def load_fasttext_topk(
    lang: str, k: int = 2000, cache_dir: Path | None = None
) -> tuple[list[str], np.ndarray]:
    """Return (words, vectors[k, 300]) for the k most frequent words of `lang`.

    fastText .vec files are frequency-sorted, so the first k lines are the top-k words.
    Vectors are returned L2-normalized.
    """
    cache_dir = cache_dir or DATA_DIR / "fasttext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"cc.{lang}.top{k}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["words"].tolist(), d["X"]

    # Stream-decompress the .gz over HTTP and stop after k lines, so we download only the
    # prefix (~tens of MB) instead of the full multi-GB file.
    words: list[str] = []
    vecs: list[np.ndarray] = []
    with requests.get(FASTTEXT_URL.format(lang=lang), stream=True, timeout=120) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
            next(text)  # header: "<count> <dim>"
            for line in text:
                if len(words) >= k:
                    break
                parts = line.rstrip().split(" ")
                if len(parts) != 301:
                    continue
                words.append(parts[0])
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
    X = np.vstack(vecs)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    np.savez(cache, words=np.array(words, dtype=object), X=X)
    return words, X


def load_muse_dict(
    src: str, tgt: str = "en", cache_dir: Path | None = None
) -> dict[str, set[str]]:
    """Load the MUSE `src`->`tgt` dictionary as {src_word: {tgt_word, ...}} (EVAL ONLY).

    Returns an empty dict if the pair is unavailable (some low-resource pairs are missing).
    """
    cache_dir = cache_dir or DATA_DIR / "muse"
    dest = cache_dir / f"{src}-{tgt}.txt"
    try:
        _download(MUSE_DICT_URL.format(src=src, tgt=tgt), dest)
    except requests.HTTPError:
        return {}

    d: dict[str, set[str]] = {}
    with open(dest, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != 2:
                continue
            s, t = parts
            d.setdefault(s, set()).add(t)
    return d
