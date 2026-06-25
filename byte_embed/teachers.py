"""Teacher encoders for distillation + a precompute/cache layer.

**SONAR** (NLLB-200 encoder, 1024-d) is the primary teacher: it covers all 200 FLORES languages,
so every study language has a real target geometry (no teacher-ceiling — the constraint that
previously forced language choices). SONAR's encoder needs a `source_lang` (FLORES code) per
sentence, so we embed language-by-language.

**Robustness:** if the `sonar` package fails to import/load (`fairseq2` can be finicky on Colab),
we fall back to **LaBSE** (sentence-transformers, 768-d). The whole pipeline is teacher-agnostic
because the students train against CACHED target vectors (`precompute_targets`), not the live
teacher — so a one-time teacher pass is all that's needed and the student head just adapts to
`teacher.dim`.
"""
from __future__ import annotations

import json
import os

import numpy as np

from byte_embed.config import FLORES_CODE


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class SonarTeacher:
    """SONAR text encoder (1024-d). `encode(texts, source_lang=<FLORES code>)`."""

    name = "sonar"
    dim = 1024

    def __init__(self, device="cuda"):
        import torch
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        self._t2vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device))

    def encode(self, texts, source_lang, batch_size=128):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(texts), 2000):                       # coarse chunk for memory
            emb = self._t2vec.predict(texts[i:i + 2000], source_lang=source_lang,
                                      batch_size=batch_size)
            out.append(emb.detach().cpu().float().numpy())
        return _l2(np.concatenate(out, 0))


class LabseTeacher:
    """Fallback teacher: LaBSE (768-d) via sentence-transformers. `source_lang` ignored."""

    name = "labse"
    dim = 768

    def __init__(self, device="cuda"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer("sentence-transformers/LaBSE", device=device)
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts, source_lang=None, batch_size=128):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self._m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)


def load_teacher(name="sonar", device="cuda"):
    """Load the requested teacher; fall back to LaBSE if SONAR can't be imported/loaded."""
    if name == "sonar":
        try:
            t = SonarTeacher(device=device)
            print("  [teacher] SONAR text encoder loaded (1024-d)")
            return t
        except Exception as e:  # noqa: BLE001 — fairseq2/sonar install issues on some runtimes
            print(f"  [teacher] SONAR unavailable ({type(e).__name__}: {str(e)[:90]}); "
                  f"falling back to LaBSE")
    t = LabseTeacher(device=device)
    print(f"  [teacher] LaBSE loaded ({t.dim}-d)")
    return t


def precompute_targets(teacher, balanced, langs, cache_dir="checkpoints", tag=""):
    """Embed the balanced training sentences ONCE with the teacher (per-language `source_lang` for
    SONAR) and cache to disk. Both the byte and subword students then train against the SAME cached
    vectors → identical supervision + a single teacher pass.

    Returns (sentences, sent_langs, targets) where `targets` is an L2-normalized float32 array
    [N, teacher.dim] index-aligned with `sentences`. Cached as `<tag>.npy` + a `.json` index, so a
    re-run (or the second student) skips re-embedding."""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"teachertargets_{teacher.name}_{tag}")
    npy, idx = base + ".npy", base + ".json"
    if os.path.exists(npy) and os.path.exists(idx):
        meta = json.loads(open(idx, encoding="utf-8").read())
        print(f"  [teacher] reusing cached targets {npy} ({meta['n']} x {meta['dim']})")
        return meta["sentences"], meta["sent_langs"], np.load(npy)

    sentences, sent_langs, chunks = [], [], []
    for lang in langs:
        sents = balanced[lang]
        sl = FLORES_CODE[lang]
        print(f"  [teacher] embedding {len(sents)} {lang} ({sl}) sentences ...")
        chunks.append(teacher.encode(sents, source_lang=sl).astype(np.float32))
        sentences += sents
        sent_langs += [lang] * len(sents)
    targets = np.concatenate(chunks, 0)
    np.save(npy, targets)
    open(idx, "w", encoding="utf-8").write(json.dumps(
        {"sentences": sentences, "sent_langs": sent_langs,
         "n": len(sentences), "dim": int(targets.shape[1])}, ensure_ascii=False))
    print(f"  [teacher] cached targets {targets.shape} -> {npy}")
    return sentences, sent_langs, targets
