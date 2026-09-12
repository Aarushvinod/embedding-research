"""Exp 4 — native script vs romanized script.

Claim under test: byte's advantage is largest on native non-Latin scripts, and may partly reverse
when those languages are romanized.

Measurements (te, bn, am, ar, zh; both architectures, all sizes):
  1. Native-script advantage by script (merge, from the stored native results): is byte's advantage
     over subword larger on the non-Latin cells than on the Latin cells? Bootstrapped over cells.
  2. Three conditions on the SAME query ids of the cached 20k pools (MIRACL te/bn/ar/zh, Amharic-PR
     am) and Belebele:  NN native queries + native passages (the stored results), RR both romanized,
     RN romanized queries against native passages (the realistic script-mismatch case: Arabizi,
     pinyin, romanized Telugu queries against native corpora). Romanization = uroman, cached per
     text; a second scheme for Arabic (Buckwalter, lossless 1:1) and Chinese (tone-marked pinyin,
     pypinyin) controls for the transliteration choice.
  3. Representation level: per-sentence cosine between native and romanized FLORES encodings, and
     each language's centroid distance to the Latin cluster, native vs romanized.
Statistics: paired bootstrap over queries for RR−NN and RN−NN; the difference-in-differences
(byte's drop minus subword's drop) with its own bootstrap interval is the statistic for the
trade-off. Only the DIFFERENCE is interpretable: romanization destroys information for everyone.

Reading (pre-registered): larger byte advantage on non-Latin cells plus a larger byte drop under RR
or RN -> byte is better for native scripts and pays for it on romanized input; no differential drop
-> the native-script advantage comes free.

Requires `uroman` (pip install uroman) — hard failure if absent, never a silent fallback.

  python -m byte_embed.interp_script --only byte-small
  python -m byte_embed.interp_script --only byte-small --skip-battery      # shift only
  python -m byte_embed.interp_script --merge
  python -m byte_embed.interp_script --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from byte_embed.interp_common import (LATIN, MAIN_MODELS, SCRIPT, flores_parallel, load_student,
                                      make_encode, merge_parts, models_in, part_path, read_json,
                                      stored_results, utf8_stdout, write_json)

ANALYSIS = "script"
SCHEMA = 2                         # bumped when the part-file contents change meaning
ROMAN_LANGS = ["te", "bn", "am", "ar", "zh"]
ISO3 = {"te": "tel", "bn": "ben", "am": "amh", "ar": "ara", "zh": "zho"}
MIRACL_ROMAN = ["te", "bn", "zh", "ar"]
ALT_SCHEME = {"ar": "buckwalter", "zh": "pinyin"}
CONDITIONS = ("RR", "RN")

# Buckwalter transliteration (lossless, 1:1) — the standard table plus Arabic-Indic digits and
# Arabic punctuation; every other character passes through unchanged.
BUCKWALTER = {
    "ء": "'", "آ": "|", "أ": ">", "ؤ": "&", "إ": "<", "ئ": "}", "ا": "A", "ب": "b", "ة": "p", "ت": "t",
    "ث": "v", "ج": "j", "ح": "H", "خ": "x", "د": "d", "ذ": "*", "ر": "r", "ز": "z", "س": "s", "ش": "$",
    "ص": "S", "ض": "D", "ط": "T", "ظ": "Z", "ع": "E", "غ": "g", "ـ": "_", "ف": "f", "ق": "q", "ك": "k",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ى": "Y", "ي": "y", "ً": "F", "ٌ": "N", "ٍ": "K",
    "َ": "a", "ُ": "u", "ِ": "i", "ّ": "~", "ْ": "o", "ٰ": "`", "پ": "P", "چ": "J", "ڤ": "V", "گ": "G",
    "،": ",", "؛": ";", "؟": "?", "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6",
    "٧": "7", "٨": "8", "٩": "9"}

# Unicode blocks of each source script: a "romanized" text that still contains any of these letters
# was NOT romanized and must not be scored.
SCRIPT_BLOCKS = {"te": [(0x0C00, 0x0C7F)], "bn": [(0x0980, 0x09FF)],
                 "am": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)],
                 "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF),
                        (0xFE70, 0xFEFF)],
                 "zh": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)]}


def buckwalter(text):
    return "".join(BUCKWALTER.get(c, c) for c in text)


class Romanizer:
    """A transliteration scheme with an on-disk cache per (scheme, language)
    (`checkpoints/translit_{scheme}_{lang}.json`) so the 20k-passage pools are romanized once; used
    as the text transform inside the eval battery, where it only romanizes cache misses."""

    def __init__(self, lang, cache_dir="checkpoints", scheme="uroman", engine=None):
        self.lang, self.scheme = lang, scheme
        self.cp = Path(cache_dir) / f"translit_{scheme}_{lang}.json"
        self.cache = read_json(self.cp) or {}
        self._engine, self.dirty = engine, 0

    def engine(self):
        if self._engine is None:
            if self.scheme == "uroman":
                try:
                    from uroman import Uroman
                except ImportError as e:  # never degrade silently
                    raise SystemExit("uroman is required for the transliteration probe: pip install uroman") from e
                ur = Uroman()
                self._engine = lambda t: ur.romanize_string(t, lcode=ISO3[self.lang])
            elif self.scheme == "buckwalter":
                self._engine = buckwalter
            elif self.scheme == "pinyin":
                try:
                    from pypinyin import Style, lazy_pinyin
                except ImportError as e:
                    raise SystemExit("pypinyin is required for the pinyin control: pip install pypinyin") from e
                self._engine = lambda t: "".join(lazy_pinyin(t, style=Style.TONE))
            else:
                raise ValueError(self.scheme)
        return self._engine

    def __call__(self, text):
        r = self.cache.get(text)
        if r is None:
            r = self.engine()(text)
            self.cache[text] = r
            self.dirty += 1
            if self.dirty >= 5000:
                self.flush()
        return r

    def many(self, texts):
        out = [self(t) for t in texts]
        self.flush()
        check_romanized(out, self.lang)
        return out

    def flush(self):
        if self.dirty:
            write_json(self.cp, self.cache)
            self.dirty = 0


def _source_script_left(text, lang):
    blocks = SCRIPT_BLOCKS.get(lang, [])
    return any(lo <= ord(c) <= hi for c in text for lo, hi in blocks if c.isalpha())


def check_romanized(rom, lang, min_frac=0.99):
    """Refuse to score garbage: >= min_frac of the outputs must contain an ASCII letter AND no letter
    of the source script may survive."""
    ok = sum(1 for r in rom if r and any(c.isascii() and c.isalpha() for c in r)
             and not _source_script_left(r, lang))
    frac = ok / max(len(rom), 1)
    if frac < min_frac:
        raise SystemExit(f"romanization for {lang} produced unusable output for {1 - frac:.1%} "
                         f"of texts — refusing to score garbage")
    return frac


def _centroid_dist(A, B):
    return round(float(np.linalg.norm(A.mean(0) - B.mean(0))), 4)


def belebele_texts(lang):
    """(questions, unique passages) exactly as eval_mteb.eval_belebele builds them."""
    from datasets import load_dataset

    from byte_embed.config import FLORES_CODE
    d = load_dataset("facebook/belebele", FLORES_CODE[lang], split="test")
    return list(d["question"]), sorted(set(d["flores_passage"]))


def pool_texts(lang, ckpt_dir):
    """(queries, passages) of the cached 20k pools this language is scored on."""
    q, p = [], []
    if lang in MIRACL_ROMAN:
        pool = read_json(Path(ckpt_dir) / f"miracl_{lang}_250q_20000d_0.json")
        if pool:
            q += list(pool["queries"].values()); p += list(pool["pool_text"])
    if lang == "am":
        for cp in sorted(Path(ckpt_dir).glob("qa_amharicpr_am_250q_20000d_0*.json")):
            pool = read_json(cp)
            if pool:
                q += list(pool["queries"].values()); p += list(pool["pool_text"])
    return q, p


def schemes_for(lang):
    return ["uroman"] + ([ALT_SCHEME[lang]] if lang in ALT_SCHEME else [])


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, seed=0, skip_battery=False):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:          # first-round part files (different keys) are discarded
        res = {"schema": SCHEMA}
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    bs = 32 if bm.get("kind") == "byte" else 128
    enc = make_encode(student, device, batch_size=bs)
    res.update(model=name, kind=bm.get("kind"), steps_run=bm.get("steps_run"))
    par = flores_parallel(cache_dir=ckpt_dir)
    langs = list(par)

    # ---- (3) representation shift on FLORES, per scheme
    res.setdefault("shift", {})
    F = None
    for lang in ROMAN_LANGS:
        for scheme in schemes_for(lang):
            key = f"{scheme}:{lang}"
            if key in res["shift"]:
                continue
            if F is None:
                F = {l: enc(par[l]) for l in langs}
                latin_c = np.mean([F[l].mean(0) for l in LATIN], 0, keepdims=True)
            try:
                R = enc(Romanizer(lang, ckpt_dir, scheme).many(par[lang]))
            except SystemExit as e:
                if scheme == "uroman":
                    raise
                print(f"  [script] {key}: {e} -> alternative scheme skipped")
                res["shift"][key] = None
                continue
            cos = (F[lang] * R).sum(1)
            res["shift"][key] = {"cos_mean": round(float(cos.mean()), 4), "cos_std": round(float(cos.std()), 4),
                                 "centroid_shift": _centroid_dist(F[lang], R),
                                 "dist_to_latin_native": _centroid_dist(F[lang], latin_c),
                                 "dist_to_latin_roman": _centroid_dist(R, latin_c)}
            write_json(outp, res)
    print("  shift: " + "  ".join(f"{k}:cos={v['cos_mean']}" for k, v in res["shift"].items() if v))
    if skip_battery:
        return

    # ---- (2) RR and RN conditions, per language and scheme, saved as they complete
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    res.setdefault("cond", {})
    for lang in ROMAN_LANGS:
        for scheme in schemes_for(lang):
            todo = [c for c in CONDITIONS if lang not in (res["cond"].get(f"{scheme}:{c}") or {}).get("langs", [])]
            if not todo:
                continue
            rz = Romanizer(lang, ckpt_dir, scheme)
            pq, pp = pool_texts(lang, ckpt_dir)
            bq, bp = belebele_texts(lang)
            try:
                rz.many(pq + bq)                              # queries (both conditions)
                rz.many(pp + bp)                              # passages (RR only), checked as well
            except SystemExit as e:
                if scheme == "uroman":
                    raise
                print(f"  [script] {scheme}:{lang}: {e} -> alternative scheme skipped")
                continue
            query_set = set(pq) | set(bq)
            for cond in todo:
                transform = rz if cond == "RR" else (lambda t, _r=rz, _q=query_set: _r(t) if t in _q else t)
                enc_c = make_encode(student, device, transform=transform, batch_size=bs)
                block = res["cond"].setdefault(f"{scheme}:{cond}", {"langs": [], "belebele": {}, "miracl": {"per_lang": {}},
                                                                    "qa_retrieval": {"amharicpr": {"per_lang": {}}}})
                print(f"=== {name}: {scheme} {cond} {lang} ===")
                if lang in MIRACL_ROMAN:
                    m = eval_miracl_langs(enc_c, [lang], n_queries=250, distractors=20000, cache_dir=ckpt_dir)
                    block["miracl"]["per_lang"][lang] = m["per_lang"].get(lang)
                if lang == "am":
                    q = eval_qa_retrieval(enc_c, benchmarks=("amharicpr",), n_queries=250, distractors=20000,
                                          cache_dir=ckpt_dir)
                    block["qa_retrieval"]["amharicpr"]["per_lang"]["am"] = q["amharicpr"]["per_lang"].get("am")
                block["belebele"][lang] = eval_battery(enc_c, [lang])["belebele"].get(lang)
                block["langs"].append(lang)
                rz.flush()
                write_json(outp, res)
    print(f"  saved -> {outp}")


# ----------------------------------------------------------------------------------------------
def cell_script(cell):
    return SCRIPT.get(cell[1], "?")


def native_advantage_by_script(byte_bm, sub_bm, n_boot=10000, seed=0):
    """byte − subword nDCG@10 per cell from the stored native results, grouped Latin vs non-Latin;
    the contrast (non-Latin mean − Latin mean) bootstrapped over cells within each group."""
    from byte_embed.stats import iter_cells
    b, s = dict(iter_cells(byte_bm)), dict(iter_cells(sub_bm))
    deltas = {c: b[c]["ndcg@10"] - s[c]["ndcg@10"] for c in b if c in s and b[c] and s[c]}
    non = np.array([v for c, v in deltas.items() if cell_script(c) != "Latn"])
    lat = np.array([v for c, v in deltas.items() if cell_script(c) == "Latn"])
    if not len(non) or not len(lat):
        return None
    rng = np.random.default_rng(seed)
    boots = (non[rng.integers(0, len(non), (n_boot, len(non)))].mean(1)
             - lat[rng.integers(0, len(lat), (n_boot, len(lat)))].mean(1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"non_latin_mean": round(float(non.mean()), 4), "latin_mean": round(float(lat.mean()), 4),
            "contrast": round(float(non.mean() - lat.mean()), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "n_non": int(len(non)), "n_latin": int(len(lat)),
            "significant": bool(lo > 0 or hi < 0)}


def diff_in_diff(a_rom, a_nat, b_rom, b_nat, n_boot=10000, seed=0):
    """Per cell: (a_rom − a_nat) − (b_rom − b_nat) over the shared query ids, bootstrapped."""
    from byte_embed.stats import iter_perquery
    A_r, A_n, B_r, B_n = (dict(iter_perquery(x)) for x in (a_rom, a_nat, b_rom, b_nat))
    rows = []
    rng = np.random.default_rng(seed)
    for cell in sorted(set(A_r) & set(A_n) & set(B_r) & set(B_n)):
        keys = sorted(set(A_r[cell]) & set(A_n[cell]) & set(B_r[cell]) & set(B_n[cell]))
        if not keys:
            continue
        d = np.array([(A_r[cell][k] - A_n[cell][k]) - (B_r[cell][k] - B_n[cell][k]) for k in keys])
        boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append((cell, {"delta": round(float(d.mean()), 4), "ci_low": round(float(lo), 4),
                            "ci_high": round(float(hi), 4), "n_paired": len(d),
                            "significant": bool(lo > 0 or hi < 0)}))
    return rows


def merge(results="results/retrieval_bgem3.json", n_boot=10000):
    from byte_embed.stats import compare
    d = merge_parts(ANALYSIS)
    M = d["models"]
    models = models_in(results)[0]
    print("\nEXP 4 — NATIVE SCRIPT vs ROMANIZED SCRIPT")
    print("  (1) native-script advantage: byte − subword nDCG@10 per cell (stored 20k-pool results), "
          "non-Latin vs Latin cells; contrast bootstrapped over cells")
    for size in ("small", "base", "large"):
        b, s = models.get(f"byte-{size}"), models.get(f"subword-{size}")
        if not (b and s):
            continue
        r = native_advantage_by_script(b, s, n_boot)
        if r:
            print(f"  {size:6} non-Latin cells {r['non_latin_mean']:+.4f} (n={r['n_non']})   Latin cells "
                  f"{r['latin_mean']:+.4f} (n={r['n_latin']})   contrast {r['contrast']:+.4f} "
                  f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{' *' if r['significant'] else ''}")
    print("\n  (2) romanized − native, nDCG@10 (paired bootstrap over queries; RR = both sides romanized, "
          "RN = romanized queries vs native passages)")
    per = {}
    for n in MAIN_MODELS:
        r = M.get(n)
        base = stored_results(n, results)
        if not (r and r.get("cond") and base):
            continue
        for key, block in sorted(r["cond"].items()):
            rows = compare(block, base, n_boot)
            per[(n, key)] = rows
            cells = "  ".join(f"{c[0][:6]}-{c[1]}:{x['delta']:+.3f}{'*' if x['significant'] else ''}" for c, x in rows)
            print(f"  {n:15}{key:14}{cells}")
    print("\n  (3) difference-in-differences per size: byte's drop − subword's drop (negative = byte pays more "
          "for romanization)")
    for size in ("small", "base", "large"):
        rb, rs = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        bb, bs_ = stored_results(f"byte-{size}", results), stored_results(f"subword-{size}", results)
        if not (rb and rs and bb and bs_):
            continue
        for key in sorted(set(rb.get("cond", {})) & set(rs.get("cond", {}))):
            rows = diff_in_diff(rb["cond"][key], bb, rs["cond"][key], bs_, n_boot)
            cells = "  ".join(f"{c[0][:6]}-{c[1]}:{x['delta']:+.3f}{'*' if x['significant'] else ''}" for c, x in rows)
            print(f"  {size:6} {key:14}{cells}")
    print("\n  (4) representation shift: cos(native, romanized) per sentence; centroid distance to the Latin "
          "cluster native → romanized")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r or not r.get("shift"):
            continue
        print(f"  {n:15}" + "  ".join(f"{k}:{v['cos_mean']:.3f} L{v['dist_to_latin_native']:.2f}→{v['dist_to_latin_roman']:.2f}"
                                     for k, v in r["shift"].items() if v))
    print("  reading: larger byte advantage on non-Latin cells + a larger byte drop under RR/RN -> better for "
          "native scripts, paid for on romanized input; no differential drop -> the advantage comes free.")


def _selftest():
    assert buckwalter("العربية") == "AlErbyp" and buckwalter("٢٠٢٣ ok") == "2023 ok"
    assert check_romanized(["telugu vakyam", "inko"], "te") == 1.0
    for bad in (["", "", "ok"], ["ok తెలుగు", "ok"]):
        try:
            check_romanized(bad, "te")
            raise AssertionError("should have refused")
        except SystemExit:
            pass
    assert check_romanized(["AlErbyp hy"], "ar") == 1.0
    class Fake:
        def __call__(self, t):
            return "rom:" + t
    rz = Romanizer("te", cache_dir="results/_selftest_tmp", engine=Fake())
    assert rz.many(["a", "b"]) == ["rom:a", "rom:b"] and rz.cache["a"] == "rom:a"
    rz.cp.unlink(missing_ok=True)
    q = {f"q{i}": 0.5 for i in range(50)}
    mk = lambda off: {"miracl": {"per_lang": {"te": {"per_query": {k: v + off for k, v in q.items()}}}}}  # noqa: E731
    rows = diff_in_diff(mk(-0.2), mk(0.0), mk(-0.05), mk(0.0), n_boot=200)
    assert rows[0][0] == ("MIRACL", "te") and abs(rows[0][1]["delta"] + 0.15) < 1e-6, rows
    cells = {"belebele": {l: {"ndcg@10": 0.8 + (0.1 if SCRIPT[l] != "Latn" else 0.0), "per_query": {}} for l in SCRIPT}}
    base = {"belebele": {l: {"ndcg@10": 0.8, "per_query": {}} for l in SCRIPT}}
    r = native_advantage_by_script(cells, base, n_boot=200)
    assert abs(r["contrast"] - 0.1) < 1e-6 and r["n_non"] == 5 and r["n_latin"] == 5, r
    assert schemes_for("ar") == ["uroman", "buckwalter"] and schemes_for("te") == ["uroman"]
    print("selftest OK: Buckwalter, romanization guard, cached romanizer, diff-in-diff, script contrast")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-battery", action="store_true", help="representation shift only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge(a.results)
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.seed, a.skip_battery)


if __name__ == "__main__":
    main()
