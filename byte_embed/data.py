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
