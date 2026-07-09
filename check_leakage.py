"""Train/eval leakage check: does any training sentence appear in the eval text?

Training data is Wikipedia (`load_balanced_sentences`). Checks verbatim (NFKC-normalized) overlap
against the ACTUAL eval datasets the study scores on:
  - FLORES bitext  : mteb/flores devtest (one column per FLORES language)   [Wikipedia-derived]
  - Belebele       : facebook/belebele (questions exact + passages substring) [Wikipedia-derived]
  - SIB-200        : Davlan/sib200 (text)                                      [Wikipedia-derived]
  - STS            : SemRel / IndicCrosslingual / C-MTEB sentence pairs        [low risk]
  - MIRACL         : cached pools if present                                   [Wikipedia corpus]

Run where the cached training set lives (rebuild locally with load_balanced_sentences if needed):
  python check_leakage.py [path/to/wiki_balanced_*.json]
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

LANGS = ["te", "ta", "mr", "am", "ha", "rw", "en", "zh", "ar"]
TRAIN = (sys.argv[1] if len(sys.argv) > 1
         else "checkpoints/wiki_balanced_te-ta-mr-am-ha-rw-en-zh-ar_42000.json")


def norm(s) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower())


train = json.loads(Path(TRAIN).read_text(encoding="utf-8"))
tset = {l: {norm(x) for x in train.get(l, [])} for l in LANGS}
print("unique training sentences/lang:", {l: len(tset[l]) for l in LANGS})

from datasets import load_dataset  # noqa: E402

from byte_embed.config import FLORES_CODE  # noqa: E402


def exact(l, sents):
    ev = {norm(s) for s in sents if s}
    return len(ev & tset[l]), len(ev)


def substr(l, passages):
    """# training sentences that appear as a substring of some eval passage (catches long passages)."""
    blob = norm(" \n ".join(str(p) for p in passages))
    return sum(1 for s in tset[l] if len(s) >= 20 and s in blob)


def line(l, hit, n, extra=""):
    print(f"  {l}: {hit:4d}/{n:5d} verbatim ({100 * hit / max(1, n):5.2f}%){extra}")


print("\n=== FLORES bitext  (mteb/flores devtest) ===")
try:
    fl = load_dataset("mteb/flores", "default", split="devtest")
    for l in LANGS:
        col = FLORES_CODE[l]
        if col in fl.column_names:
            line(l, *exact(l, list(fl[col])))
        else:
            print(f"  {l}: column {col} absent")
except Exception as e:  # noqa: BLE001
    print(f"  mteb/flores load failed: {type(e).__name__}: {e}")

print("\n=== Belebele  (facebook/belebele: questions exact + passages substring) ===")
for l in LANGS:
    try:
        d = load_dataset("facebook/belebele", FLORES_CODE[l], split="test")
        qh, qn = exact(l, list(d["question"]))
        ps = substr(l, set(d["flores_passage"]))
        line(l, qh, qn, f"  | {ps} train sents are substrings of a passage")
    except Exception as e:  # noqa: BLE001
        print(f"  {l}: {type(e).__name__}")

print("\n=== SIB-200  (Davlan/sib200, all splits) ===")
for l in LANGS:
    sents = []
    for sp in ("train", "validation", "test"):
        try:
            sents += list(load_dataset("Davlan/sib200", FLORES_CODE[l], split=sp)["text"])
        except Exception:  # noqa: BLE001
            pass
    if sents:
        line(l, *exact(l, sents))
    else:
        print(f"  {l}: unavailable")

print("\n=== STS  (sentence1 + sentence2, all routed splits) ===")
from byte_embed.eval_mteb import STS_ROUTE, _concat_splits  # noqa: E402
for l in LANGS:
    s1, s2, _ = _concat_splits(*STS_ROUTE[l])
    if s1:
        line(l, *exact(l, s1 + s2))
    else:
        print(f"  {l}: unavailable")

print("\n=== MIRACL  (cached pools, if present) ===")
for l in ["en", "zh", "ar", "te", "bn", "sw", "yo"]:      # every trained + MIRACL-evaluated language
    caches = sorted(Path("checkpoints").glob(f"miracl_{l}_*.json"))
    if not caches:
        print(f"  {l}: no cached pool (skip)")
        continue
    for cache in caches:                                  # check EVERY pool variant, not just caches[0]
        d = json.loads(cache.read_text(encoding="utf-8"))
        qh, qn = exact(l, list(d["queries"].values()))
        ps = substr(l, d["pool_text"])
        print(f"  {l} [{cache.name}]: queries {qh}/{qn} verbatim | "
              f"{ps} train sents are substrings of a pool passage")
