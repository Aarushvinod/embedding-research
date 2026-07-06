"""Teacher encoders for distillation + a precompute/cache layer.

**SONAR** (NLLB-200 encoder, 1024-d) is the primary teacher: it covers all 200 FLORES languages,
so every study language has a real target geometry (no teacher-ceiling — the constraint that
previously forced language choices). SONAR's encoder needs a `source_lang` (FLORES code) per
sentence, so we embed language-by-language.

**Three-tier loading** (`load_teacher`): the official `sonar-space`/fairseq2 stack (source of truth)
if installed → else the **fairseq2-free HF port** `cointegrated/SONAR_200_text_encoder` (1024-d,
validated to match official SONAR; the reliable default on Colab, where fairseq2 has no wheel for the
newest torch) → else LaBSE (768-d) as a last resort. The whole pipeline is teacher-agnostic because
the students train against CACHED target vectors (`precompute_targets`), not the live teacher — so a
one-time teacher pass is all that's needed and the student head just adapts to `teacher.dim`.
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


class SonarHFTeacher:
    """SONAR text encoder (1024-d) via the **fairseq2-free** HuggingFace port
    `cointegrated/SONAR_200_text_encoder` — an off-the-shelf transformers model (M2M100 encoder +
    the model card's documented masked mean-pool, which IS SONAR's pooling: SONAR has no learned
    pooling head). The port's author validates that its embeddings equal the official SONAR's; we
    additionally checked cross-lingual alignment (parallel en↔te ≈ 0.73–0.82 cos, non-parallel ≈ 0.15).

    This is the reliable default on Colab, where the official `sonar-space` stack needs a `fairseq2`
    wheel pinned to the exact torch build (none exists for Colab's newest torch). Same 202 NLLB
    languages; `encode(texts, source_lang=<FLORES code>)`."""

    name = "sonar"        # same name as the official path → the targets cache is interchangeable
    dim = 1024
    MODEL = "cointegrated/SONAR_200_text_encoder"

    def __init__(self, device="cuda"):
        from transformers import AutoTokenizer
        from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.enc = M2M100Encoder.from_pretrained(self.MODEL).to(device).eval()

    def encode(self, texts, source_lang, batch_size=64):
        import torch

        if not texts:
            return np.zeros((0, self.dim), np.float32)
        self.tok.src_lang = source_lang                      # NLLB/FLORES code, e.g. "tel_Telu"
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                h = self.enc(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1)
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1)   # masked mean-pool = SONAR pooling
            out.append(emb.float().cpu().numpy())
        return _l2(np.concatenate(out, 0))


class RetrievalTeacher:
    """Retrieval-trained teacher via sentence-transformers — the retrieval-study teachers.

    Unlike SONAR (a bitext/translation encoder), these models were TRAINED for dense retrieval, so
    distilling them hands the students a retrieval-shaped geometry (the deep-retrieval ceiling of the
    SONAR runs was the teacher, not the students). Language-agnostic encode: `source_lang` accepted
    and ignored. `prefix` is prepended to every sentence (mE5 wants its 'passage: ' role marker;
    BGE-M3 needs none). Both are 1024-d, so the student head is unchanged vs the SONAR runs."""

    def __init__(self, model_id, name, prefix="", device="cuda"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.prefix = prefix
        self._m = SentenceTransformer(model_id, device=device)
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts, source_lang=None, batch_size=64):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self._m.encode([self.prefix + t for t in texts], batch_size=batch_size,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)


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
    """Load the teacher. `sonar` (default): official `sonar-space`/fairseq2 stack if installed, else
    the fairseq2-free HF port, else LaBSE. `bge-m3` / `me5-large`: retrieval-trained teachers for the
    retrieval study. All expose the same `encode(texts, source_lang)` (retrieval teachers ignore
    source_lang), so the cached-targets pipeline is teacher-agnostic."""
    if name in ("bge-m3", "bgem3"):
        t = RetrievalTeacher("BAAI/bge-m3", "bge-m3", device=device)
        print(f"  [teacher] BGE-M3 (BAAI/bge-m3) loaded — retrieval-trained, {t.dim}-d")
        return t
    if name in ("me5-large", "me5"):
        t = RetrievalTeacher("intfloat/multilingual-e5-large", "me5-large",
                             prefix="passage: ", device=device)
        print(f"  [teacher] mE5-large loaded — retrieval-trained, {t.dim}-d ('passage: ' prefix)")
        return t
    if name == "sonar":
        try:
            t = SonarTeacher(device=device)
            print("  [teacher] official SONAR (sonar-space / fairseq2) loaded — 1024-d")
            return t
        except Exception as e:  # noqa: BLE001 — fairseq2 absent or torch/ABI mismatch (common on Colab)
            print(f"  [teacher] official sonar-space unavailable ({type(e).__name__}); "
                  f"using the fairseq2-free HF SONAR port instead")
        try:
            t = SonarHFTeacher(device=device)
            print("  [teacher] SONAR via HF port 'cointegrated/SONAR_200_text_encoder' loaded — 1024-d")
            return t
        except Exception as e:  # noqa: BLE001
            print(f"  [teacher] HF SONAR port unavailable ({type(e).__name__}: {str(e)[:80]}); "
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


def _targets_base(cache_dir, teacher_name, tag):
    return os.path.join(cache_dir, f"teachertargets_{teacher_name}_{tag}")


def targets_exist(cache_dir, teacher_name, tag):
    """True if precomputed targets for (teacher_name, tag) are cached → callers can SKIP loading the
    teacher entirely (crucial for parallel workers: no 6× SONAR loads)."""
    b = _targets_base(cache_dir, teacher_name, tag)
    return os.path.exists(b + ".npy") and os.path.exists(b + ".json")


def load_cached_targets(cache_dir, teacher_name, tag):
    """Load cached (sentences, sent_langs, targets) without touching the teacher."""
    b = _targets_base(cache_dir, teacher_name, tag)
    meta = json.loads(open(b + ".json", encoding="utf-8").read())
    print(f"  [teacher] reusing cached targets {b}.npy ({meta['n']} x {meta['dim']}) — no teacher load")
    return meta["sentences"], meta["sent_langs"], np.load(b + ".npy")
