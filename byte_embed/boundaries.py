"""Boundary-injection input transforms — the tokenization-mechanism probe (byte students only).

Three arms, all the SAME ByT5 architecture/params; only the input text differs:
  A  raw bytes                — the normal student (no transform)
  B  'teacher' boundaries     — a 1-byte marker (U+001E) inserted at every position where the
                                TEACHER's tokenizer (BGE-M3 = XLM-R sentencepiece) would split.
                                Grants the byte model subword's segmentation INFORMATION with none
                                of its vocabulary table.
  C  'random' boundaries      — the placebo: the SAME NUMBER of markers per sentence at random
                                character positions (deterministic per sentence via crc32, so fake
                                boundaries are as stable across epochs as real ones). Controls for
                                sequence-length / delimiter / register-token artifacts: if B > A but
                                B ~ C, markers helped for non-linguistic reasons.

Insertions are at CHARACTER positions, so multi-byte UTF-8 sequences are never split. The teacher
always sees CLEAN text (targets are cached before the transform); the transform applies to the
student's training inputs AND to all eval inputs (each arm is evaluated with its own transform).

  python -m byte_embed.boundaries --selftest
"""
from __future__ import annotations

import random
import zlib

MARKER = "\x1e"                      # record separator: 1 UTF-8 byte, ~absent from web corpora
_TOK = {}


def _teacher_tok(model_id="BAAI/bge-m3"):
    if model_id not in _TOK:
        from transformers import AutoTokenizer
        _TOK[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOK[model_id]


def boundary_cuts(text, tok):
    """Character positions (0 < pos < len) where `tok` places token boundaries."""
    offs = tok(text, return_offsets_mapping=True, add_special_tokens=False,
               truncation=True, max_length=512)["offset_mapping"]
    return sorted({e for _, e in offs if 0 < e < len(text)})


def _insert(text, cuts):
    if not cuts:
        return text
    parts, prev = [], 0
    for c in cuts:
        parts.append(text[prev:c])
        prev = c
    parts.append(text[prev:])
    return MARKER.join(parts)


def mark_teacher(text, model_id="BAAI/bge-m3"):
    """Arm B: markers at the teacher tokenizer's boundaries."""
    return _insert(text, boundary_cuts(text, _teacher_tok(model_id)))


def mark_random(text, model_id="BAAI/bge-m3"):
    """Arm C: the SAME marker count as arm B, at random character positions. Deterministic per
    sentence (crc32 seed) so the fake boundaries are stable across epochs/processes."""
    k = len(boundary_cuts(text, _teacher_tok(model_id)))
    if k == 0 or len(text) < 2:
        return text
    rng = random.Random(zlib.crc32(text.encode("utf-8")))
    k = min(k, len(text) - 1)
    cuts = sorted(rng.sample(range(1, len(text)), k))
    return _insert(text, cuts)


def make_transform(kind, model_id="BAAI/bge-m3"):
    """None | 'teacher' | 'random' -> a text->text callable (or None for the raw arm)."""
    if not kind or kind == "none":
        return None
    if kind == "teacher":
        return lambda t: mark_teacher(t, model_id)
    if kind == "random":
        return lambda t: mark_random(t, model_id)
    raise ValueError(f"unknown boundary kind {kind!r} (use None|'teacher'|'random')")


def _selftest():
    texts = ["The quick brown fox jumps over the lazy dog.",
             "የኢትዮጵያ ፕሪምየር ሊግ ጨዋታ ተጀመረ",
             "తెలుగు వికీపీడియా ఒక స్వేచ్ఛా విజ్ఞాన సర్వస్వము"]
    for t in texts:
        b, c = mark_teacher(t), mark_random(t)
        assert b.count(MARKER) == c.count(MARKER), "arm C must match arm B's marker count"
        assert c == mark_random(t), "arm C must be deterministic per sentence"
        assert b.replace(MARKER, "") == t and c.replace(MARKER, "") == t, "markers only, no edits"
        assert b.encode("utf-8").decode("utf-8") == b, "no broken UTF-8"
        print(f"  ok  markers={b.count(MARKER):>2}  len {len(t)} -> {len(b)}  | {t[:30]}...")
    print("boundaries selftest OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
