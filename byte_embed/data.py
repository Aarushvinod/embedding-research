"""Training + evaluation data for ByteEmbed.

Training sentences come from Wikipedia (monolingual, multilingual mix). Cross-lingual
retrieval eval uses OPUS-100 (`load_parallel`, parallel, EVAL-ONLY); `load_flores_parallel`
is an alternate FLORES-200 path, currently unused by the eval. Both stream from the HF Hub.
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


def _norm(s):
    """Whitespace-collapsed, case-folded verbatim key for leakage matching."""
    return " ".join(str(s).split()).casefold()


def _eval_exclusion(langs, cache_dir):
    """Normalized set of eval passages/queries to keep OUT of the training data (leakage). Best-effort:
    the GOLD passages + queries of any cached deep-eval pool (the exact items the model is scored on)
    plus the Belebele passages (the shallow-eval retrieval targets). The MIRACL/QA CORPORA are
    Wikipedia at large and cannot be fully excluded — reservoir sampling (below) reduces that residual
    overlap and check_leakage.py measures it."""
    import glob
    import json
    import os

    excl = set()
    if cache_dir:
        for f in (glob.glob(os.path.join(cache_dir, "miracl_*.json"))
                  + glob.glob(os.path.join(cache_dir, "qa_*.json"))):
            try:
                d = json.loads(open(f, encoding="utf-8").read())
                rel = d.get("rel") or {}
                gold_ids = set().union(*[set(v) for v in rel.values()]) if rel else set()
                id2txt = dict(zip(d.get("pool_id", []), d.get("pool_text", [])))   # gold only (bounded)
                excl.update(_norm(id2txt[g]) for g in gold_ids if g in id2txt)
                excl.update(_norm(t) for t in (d.get("queries") or {}).values())
            except Exception:  # noqa: BLE001
                continue
    try:
        from datasets import load_dataset

        from byte_embed.eval_mteb import FLORES_CODE
        for lang in langs:
            fc = FLORES_CODE.get(lang)
            if not fc:
                continue
            try:
                d = load_dataset("facebook/belebele", fc, split="test")
                excl.update(_norm(p) for p in set(d["flores_passage"]))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    if excl:
        print(f"  [data] leakage-exclusion set: {len(excl)} normalized eval passages/queries")
    return excl


def load_balanced_sentences(langs, n_per_lang=42000, min_len=20, max_len=300, seed=0,
                            cache_dir=None):
    """EQUAL (max-min balanced) monolingual training sentences per language, so no language
    dominates the distillation mix (high-resource en/zh/ar would otherwise swamp Kinyarwanda).

    Streams Wikipedia per language collecting up to `n_per_lang` usable paragraphs in
    [min_len, max_len]; if any language falls short, ALL languages are truncated to the realized
    floor (true max-min). Returns `{lang: [sentences]}` with EQUAL counts and logs realized counts.
    Measured floor for the study-9 is ~42,621 (Kinyarwanda), so the default 42k clears every language.
    Grouping by language is required so the SONAR teacher can pass the right `source_lang`."""
    import json
    import os

    cp = None
    if cache_dir:
        # cache key now depends on seed (the sample is random) + a version tag (reservoir + dedup) so
        # the old head-of-corpus caches are never silently reused.
        cp = os.path.join(cache_dir, f"wiki_balanced_{'-'.join(langs)}_{n_per_lang}_s{seed}_rsv2.json")
        if os.path.exists(cp):
            print(f"  [data] reusing balanced cache {cp}")
            return json.loads(open(cp, encoding="utf-8").read())

    from datasets import load_dataset

    rng = random.Random(seed)
    exclude = _eval_exclusion(langs, cache_dir)      # eval passages kept OUT of training (leakage)
    scan_cap = max(n_per_lang * 8, 500_000)          # how far to scan each stream for the reservoir
    per = {}
    for lang in langs:
        # RESERVOIR-sample up to n_per_lang paragraphs UNIFORMLY over the scanned corpus rather than
        # taking the first-N: head-of-corpus over-samples prominent/early articles -> a biased slice
        # that maximizes FLORES/MIRACL overlap. Eval passages are dropped inline (leakage).
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                              split="train", streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [data] wikipedia unavailable for {lang} ({e})")
            per[lang] = []
            continue
        res, seen = [], 0
        for ex in ds:
            for para in ex.get("text", "").split("\n"):
                p = para.strip()
                if not (min_len <= len(p) <= max_len) or _norm(p) in exclude:
                    continue
                seen += 1
                if len(res) < n_per_lang:            # fill the reservoir...
                    res.append(p)
                else:                                # ...then Algorithm R: keep w/ prob n_per_lang/seen
                    j = rng.randrange(seen)
                    if j < n_per_lang:
                        res[j] = p
            if seen >= scan_cap:
                break
        per[lang] = res
        print(f"  [data] {lang}: reservoir {len(res)} from {seen} scanned")

    floor = min(len(v) for v in per.values())        # balanced floor — DATA-DRIVEN (min over langs)
    out = {}
    for lang, sents in per.items():
        out[lang] = rng.sample(sents, floor)         # RANDOM-select the floor (better distribution than
                                                     # a prefix slice); collect-then-truncate
    print(f"  [data] balanced floor = {floor}/lang -> {floor * len(langs)} total sentences")

    if cp:
        os.makedirs(cache_dir, exist_ok=True)
        open(cp, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False))
    return out


def _flores(lang, split="dev"):
    """Robust FLORES-200 load via the PUBLIC Muennighoff/flores200 mirror (same eng_Latn
    codes; the canonical openlanguagedata/flores_plus is GATED -> 401). Retries, and tries
    both with and without trust_remote_code so it survives whatever `datasets` version the
    runtime ships (a script-based config on a newer datasets needs trust_remote_code=True;
    an older one rejects the kwarg). Errors are PRINTED, never silently swallowed."""
    from datasets import load_dataset

    from byte_embed.config import FLORES_CODE

    code = FLORES_CODE.get(lang, lang)
    last = None
    for _ in range(3):
        for kwargs in ({}, {"trust_remote_code": True}):
            try:
                ds = load_dataset("Muennighoff/flores200", code, split=split, **kwargs)
                return [r["sentence"] for r in ds]
            except TypeError:  # this datasets version doesn't accept trust_remote_code
                continue
            except Exception as e:  # noqa: BLE001
                last = e
    print(f"  [data] FLORES unavailable for {lang}/{code}: {type(last).__name__}: {last}")
    return []


def load_mono(lang, n=200, min_len=20, max_len=300):
    """n monolingual sentences for ONE language (Wikipedia stream -> FLORES dev fallback).

    The orthographic-robustness probe (H2) and teacher-alignment need only target-language
    text, NOT parallel data, so this is the ROBUST CORE of the eval: it runs even when every
    parallel corpus is unavailable. Wikipedia loaded fine on Colab (80k training sentences),
    so this path is reliable where the FLORES-parallel path was not."""
    from datasets import load_dataset

    out: list[str] = []
    try:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                          split="train", streaming=True)
        for ex in ds:
            for para in ex.get("text", "").split("\n"):
                para = para.strip()
                if min_len <= len(para) <= max_len:
                    out.append(para)
                    if len(out) >= n:
                        return out
    except Exception as e:  # noqa: BLE001
        print(f"  [data] wikipedia unavailable for {lang} ({type(e).__name__}); trying FLORES dev")
    if len(out) < n:
        out.extend(_flores(lang, split="dev"))
    return out[:n]


def load_eval_parallel(lang, n=400):
    """Index-aligned {'tgt': [...], 'en': [...], '_src': str} for cross-lingual P@1.

    Tries FLORES first (lang and en share sentence indices -> naturally parallel), then falls
    back to OPUS-100 (bilingual; lacks some langs e.g. Swahili). Returns {} if neither works,
    so the caller can skip xP@1 while still reporting robustness + alignment from load_mono."""
    tgt = _flores(lang, split="dev")[:n]
    en = _flores("en", split="dev")[:n]
    if tgt and en:
        m = min(len(tgt), len(en))
        return {"tgt": tgt[:m], "en": en[:m], "_src": "flores"}
    par = load_parallel(lang, n=n)
    if par and lang in par and par.get("en"):
        return {"tgt": par[lang], "en": par["en"], "_src": f"opus-{par.get('_split')}"}
    return {}


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
    """Aligned {tgt: [...], 'en': [...], '_split': str} parallel sentences from OPUS-100
    (public, non-gated). Cross-lingual sentence-retrieval eval (EVAL ONLY).

    Prefers a held-out split: test > validation > (train fallback for pairs that ship no
    eval split). `_split` records which was used so a train fallback is never silently read
    as "held-out". The English bank is deduplicated (keeping target alignment) because
    OPUS-100 repeats English sentences, and index-based P@1 would otherwise count a correct
    retrieval of a duplicate as a miss (a downward bias on the reported score).
    """
    from datasets import load_dataset

    cfg = "-".join(sorted([tgt, "en"]))
    ds, used_split = None, None
    for split in ("test", "validation", "train"):  # some low-resource pairs lack test/validation
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", cfg, split=split,
                              streaming=(split == "train"))
            used_split = split
            break
        except (ValueError, FileNotFoundError):  # unknown config/split -> try next (NOT net/auth)
            ds = None
    if ds is None:
        return {}
    out = {tgt: [], "en": [], "_split": used_split}
    seen = set()
    for ex in ds:  # works for both map-style and streaming datasets
        tr = ex["translation"]
        s_t, s_e = tr.get(tgt), tr.get("en")
        if s_t and s_e and len(s_e) > 15 and s_e not in seen:  # dedup English bank
            seen.add(s_e)
            out[tgt].append(s_t)
            out["en"].append(s_e)
            if len(out["en"]) >= n:
                break
    if not out["en"]:  # nothing passed the filter -> clean skip, not a downstream crash
        return {}
    return out
