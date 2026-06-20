"""Training + evaluation data for ByteEmbed.

Training sentences come from Wikipedia (monolingual, multilingual mix). Cross-lingual
retrieval eval uses FLORES-200 (parallel, EVAL-ONLY). Both are streamed from the HF Hub.
Swap `load_sentences` for any text source you like.
"""
from __future__ import annotations

import random


def load_sentences(langs, n_per_lang=20000, min_len=20, max_len=300):
    """Monolingual training sentences from Wikipedia, falling back to FLORES dev."""
    from datasets import load_dataset

    out = []
    for lang in langs:
        got = 0
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                              split="train", streaming=True)
            for ex in ds:
                for para in ex.get("text", "").split("\n"):
                    para = para.strip()
                    if min_len <= len(para) <= max_len:
                        out.append(para)
                        got += 1
                        if got >= n_per_lang:
                            break
                if got >= n_per_lang:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"  [data] wikipedia unavailable for {lang} ({e}); using FLORES dev")
        if got == 0:
            out.extend(_flores(lang, split="dev"))
    random.shuffle(out)
    return out


def _flores(lang, split="dev"):
    from datasets import load_dataset

    from byte_embed.config import FLORES_CODE

    code = FLORES_CODE.get(lang, lang)
    try:
        ds = load_dataset("openlanguagedata/flores_plus", code, split=split)
        return [r["text"] for r in ds]
    except Exception as e:  # noqa: BLE001
        print(f"  [data] FLORES unavailable for {lang}/{code}: {e}")
        return []


def load_flores_parallel(langs, n=400):
    """Return {lang: [sentences]} aligned by index across languages (EVAL ONLY)."""
    par = {}
    for lang in langs:
        sents = _flores(lang, split="dev")[:n]
        if sents:
            par[lang] = sents
    # keep only the common aligned length
    if par:
        m = min(len(v) for v in par.values())
        par = {k: v[:m] for k, v in par.items()}
    return par


def load_parallel(tgt, n=400):
    """Aligned {tgt: [...], 'en': [...]} parallel sentences from OPUS-100 (public,
    non-gated). Used for cross-lingual sentence-retrieval eval (EVAL ONLY)."""
    from datasets import load_dataset

    cfg = "-".join(sorted([tgt, "en"]))
    ds = None
    for split in ("test", "validation"):  # some low-resource pairs lack a test split
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", cfg, split=split)
            break
        except Exception:  # noqa: BLE001
            ds = None
    if ds is None:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train", streaming=True)
        except Exception:  # noqa: BLE001
            return {}
    out = {tgt: [], "en": []}
    for ex in ds:  # works for both map-style and streaming datasets
        tr = ex["translation"]
        s_t, s_e = tr.get(tgt), tr.get("en")
        if s_t and s_e and len(s_e) > 15:
            out[tgt].append(s_t)
            out["en"].append(s_e)
            if len(out["en"]) >= n:
                break
    return out
