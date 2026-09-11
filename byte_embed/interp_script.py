"""Exp 4 — script vs language (transliteration probe): is byte's cross-lingual advantage
script-mediated?

Both of the study's cross-lingual wins (AfriQA rw->en, CIRAL en->ha) are Latin<->Latin pairs, and a
byte model sees raw script bytes where a subword model sees vocabulary ids. Prior work on decoder
LLMs found unit-level specialization is conditioned on SCRIPT: romanized and native-script
representations overlap < 0.3 (Verma et al. 2026) and non-Latin scripts carry 2-20x more
language-specific neurons (Gurgurov et al. 2025) — never tested on byte models. Three
measurements, byte vs subword: (1) SHIFT — encode FLORES natively and uroman-romanized for the five
non-Latin languages; per-sentence cosine, centroid shift, and whether romanization pulls a language
toward the Latin cluster / its English translations; (2) CELLS — FLORES translation-pair alignment
and retrieval grouped into same-script (Latin-Latin) vs cross-script pairs, plus the built-in design
cells sw-rw (same family, same script), ha-am (same family, DIFFERENT script), yo-Latin (unrelated,
same script); (3) ROBUSTNESS — the 20k-pool monolingual battery of those languages re-run with
romanized queries AND passages, paired-bootstrapped against the native scores. Prediction on
record: byte shows the larger same-script alignment advantage and the larger romanization
sensitivity on non-Latin languages => script-mediated.

Requires `uroman` (pip install uroman; pinned in requirements-cloud.txt) — hard failure if absent,
never a silent fallback (the repo's unidecode path strips these scripts to nothing).

  python -m byte_embed.interp_script --only byte-small
  python -m byte_embed.interp_script --merge
  python -m byte_embed.interp_script --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from byte_embed.interp_common import (LATIN, MAIN_MODELS, SCRIPT, alignment, flores_parallel,
                                      flores_xling, load_student, make_encode, merge_parts,
                                      models_in, part_path, read_json, utf8_stdout, write_json)

ANALYSIS = "script"
ROMAN_LANGS = ["te", "bn", "am", "ar", "zh"]
ISO3 = {"te": "tel", "bn": "ben", "am": "amh", "ar": "ara", "zh": "zho"}
MIRACL_ROMAN = ["te", "bn", "zh", "ar"]
DESIGN_CELLS = {"sw_rw": [("sw", "rw")], "ha_am": [("ha", "am")],
                "yo_latin": [("yo", "sw"), ("yo", "rw"), ("yo", "ha")]}


class Romanizer:
    """uroman with an on-disk cache per language (`checkpoints/translit_uroman_{lang}.json`) so the
    20k-passage pools are romanized once; the transform used inside the eval battery reads the
    in-memory dict and only romanizes cache misses (Belebele texts on first use)."""

    def __init__(self, lang, cache_dir="checkpoints", engine=None):
        self.lang, self.cp = lang, Path(cache_dir) / f"translit_uroman_{lang}.json"
        self.cache = read_json(self.cp) or {}
        self._engine, self.dirty = engine, 0

    def engine(self):
        if self._engine is None:
            try:
                from uroman import Uroman
            except ImportError as e:  # never degrade silently
                raise SystemExit("uroman is required for the transliteration probe: "
                                 "pip install uroman") from e
            self._engine = Uroman()
        return self._engine

    def __call__(self, text):
        r = self.cache.get(text)
        if r is None:
            r = self.engine().romanize_string(text, lcode=ISO3[self.lang])
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


# Unicode blocks of each source script: a "romanized" text that still contains any of these letters
# was NOT romanized (uroman regression / unsupported character) and must not be scored.
SCRIPT_BLOCKS = {"te": [(0x0C00, 0x0C7F)], "bn": [(0x0980, 0x09FF)],
                 "am": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)],
                 "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF),
                        (0xFE70, 0xFEFF)],
                 "zh": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)]}


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


def script_cells(matrix, langs=None):
    """Group a {a: {b: value}} pair matrix (directional) into same-script vs cross-script means
    (each unordered pair averaged over both directions) plus the design cells."""
    langs = list(langs or matrix)
    def val(a, b):
        return float(np.mean([matrix[a][b], matrix[b][a]]))
    same = [val(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]
            if SCRIPT[a] == SCRIPT[b] and a in matrix and b in matrix]
    cross = [val(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]
             if SCRIPT[a] != SCRIPT[b] and a in matrix and b in matrix]
    out = {"same_script": round(float(np.mean(same)), 4) if same else None,
           "cross_script": round(float(np.mean(cross)), 4) if cross else None,
           "n_same": len(same), "n_cross": len(cross)}
    for name, pairs in DESIGN_CELLS.items():
        vals = [val(a, b) for a, b in pairs if a in matrix and b in matrix]
        out[name] = round(float(np.mean(vals)), 4) if vals else None
    return out


def _centroid_dist(A, B):
    return round(float(np.linalg.norm(A.mean(0) - B.mean(0))), 4)


def run_one(name, results, ckpt_dir, device, seed=0, skip_battery=False):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("roman_battery") or (skip_battery and res.get("shift")):
        print(f"=== {ANALYSIS}/{name}: already done -> skip ===")
        return
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    enc = make_encode(student, device, batch_size=(32 if bm.get("kind") == "byte" else 128))
    res.update(model=name, kind=bm.get("kind"), steps_run=bm.get("steps_run"))
    par = flores_parallel(cache_dir=ckpt_dir)
    langs = list(par)
    if res.get("shift") and res.get("cells"):
        print(f"  {name}: shift + cells already saved -> straight to the robustness battery")
    else:
        _shift_and_cells(name, res, enc, par, langs, outp)
    if skip_battery:
        return
    _robustness_battery(name, res, student, bm, device, ckpt_dir, outp)


def _shift_and_cells(name, res, enc, par, langs, outp):
    F = {l: enc(par[l]) for l in langs}
    latin_c = np.mean([F[l].mean(0) for l in LATIN], 0, keepdims=True)

    # (1) romanization shift
    res["shift"] = {}
    for lang in ROMAN_LANGS:
        R = enc(Romanizer(lang, ckpt_dir).many(par[lang]))
        cos = (F[lang] * R).sum(1)
        res["shift"][lang] = {
            "cos_mean": round(float(cos.mean()), 4), "cos_std": round(float(cos.std()), 4),
            "centroid_shift": _centroid_dist(F[lang], R),
            "dist_to_latin_native": _centroid_dist(F[lang], latin_c),
            "dist_to_latin_roman": _centroid_dist(R, latin_c),
            "align_en_native": alignment(F[lang], F["en"]),
            "align_en_roman": alignment(R, F["en"])}

    # (2) same-script vs cross-script cells (alignment: lower = closer; P@1: higher = better)
    align_m = {a: {b: alignment(F[a], F[b]) for b in langs if b != a} for a in langs}
    p1_m = {a: {b: r["p@1"] for b, r in row.items()} for a, row in flores_xling(F, langs).items()}
    res["cells"] = {"alignment": script_cells(align_m, langs), "p@1": script_cells(p1_m, langs)}
    write_json(outp, res)
    print(f"  shift: " + "  ".join(f"{l}:cos={res['shift'][l]['cos_mean']}" for l in ROMAN_LANGS))
    print(f"  cells P@1 same/cross = {res['cells']['p@1']['same_script']}/{res['cells']['p@1']['cross_script']}")


def _robustness_battery(name, res, student, bm, device, ckpt_dir, outp):
    # (3) robustness: monolingual battery with romanized queries AND passages, per language
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    part = res.setdefault("roman_partial", {})           # per-language save -> preemption-safe
    for lang in ROMAN_LANGS:
        if lang in part:
            print(f"  {name}/{lang}: romanized battery already saved -> skip")
            continue
        rz = Romanizer(lang, ckpt_dir)
        enc_r = make_encode(student, device, transform=rz,
                            batch_size=(32 if bm.get("kind") == "byte" else 128))
        # pre-romanize the cached pool texts in one checked pass (hoisted out of the 8192-chunked
        # scorer loop; also runs the >=99%-usable guard on the pools, not only on FLORES)
        pools = ([Path(ckpt_dir) / f"miracl_{lang}_250q_20000d_0.json"] if lang in MIRACL_ROMAN else []) + \
                (sorted(Path(ckpt_dir).glob("qa_amharicpr_am_250q_20000d_0*.json")) if lang == "am" else [])
        for cp in pools:
            pool = read_json(cp)
            if pool:
                rz.many(list(pool["queries"].values()) + pool["pool_text"])
        cell = {}
        if lang in MIRACL_ROMAN:
            m = eval_miracl_langs(enc_r, [lang], n_queries=250, distractors=20000, cache_dir=ckpt_dir)
            cell["miracl"] = m["per_lang"].get(lang)
        if lang == "am":
            q = eval_qa_retrieval(enc_r, benchmarks=("amharicpr",), n_queries=250,
                                  distractors=20000, cache_dir=ckpt_dir)
            cell["amharicpr"] = q["amharicpr"]["per_lang"].get("am")
        cell["belebele"] = eval_battery(enc_r, [lang])["belebele"].get(lang)
        rz.flush()
        part[lang] = cell
        write_json(outp, res)
    # assemble the iter_perquery-compatible battery (ONE miracl block; qa nested by benchmark)
    res["roman_battery"] = {
        "belebele": {l: c.get("belebele") for l, c in part.items()},
        "miracl": {"per_lang": {l: c["miracl"] for l, c in part.items() if "miracl" in c}},
        "qa_retrieval": {"amharicpr": {"per_lang": {"am": part["am"]["amharicpr"]}}} if "am" in part else {}}
    write_json(outp, res)
    print(f"  saved -> {outp}")


def _baseline_for(name, results):
    """Native (un-romanized) per-query scores: exp 3's `none` variant if present, else the stored
    training-time results entry (same protocol, same query ids)."""
    lg = read_json(part_path("langgeom", name))
    if lg and "none" in lg.get("variants", {}):
        return lg["variants"]["none"]
    models, _, _ = models_in(results)
    return models.get(name)


def merge(results="results/retrieval_bgem3.json", n_boot=10000):
    from byte_embed.stats import compare
    d = merge_parts(ANALYSIS)
    M = d["models"]
    print("\nEXP 4 — SCRIPT vs LANGUAGE")
    print("  (1) romanization shift (cos native↔roman; centroid dist to the Latin cluster native→roman;"
          " alignment to English translations native→roman)")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r:
            continue
        cells = "  ".join(f"{l}:{s['cos_mean']:.3f} L{s['dist_to_latin_native']:.2f}→{s['dist_to_latin_roman']:.2f}"
                          f" en{s['align_en_native']:.3f}→{s['align_en_roman']:.3f}"
                          for l, s in r["shift"].items())
        print(f"  {n:15}{cells}")
    print("\n  (2) same-script vs cross-script cells  [alignment: lower=closer | P@1: higher=better]")
    print(f"  {'model':15}{'same':>7}{'cross':>7}{'sw-rw':>7}{'ha-am':>7}{'yo-Lat':>7}   |"
          f"{'same':>7}{'cross':>7}{'sw-rw':>7}{'ha-am':>7}{'yo-Lat':>7}")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r:
            continue
        a, p = r["cells"]["alignment"], r["cells"]["p@1"]
        row = "".join(f"{(c[k] if c[k] is not None else float('nan')):>7.3f}"
                      for c in (a, p) for k in ("same_script", "cross_script", "sw_rw", "ha_am", "yo_latin"))
        print(f"  {n:15}{row[:35]}   |{row[35:]}")
    for size in ("small", "base", "large"):
        cells = [((M.get(f"{k}-{size}") or {}).get("cells") or {}).get("p@1") or {}
                 for k in ("byte", "subword")]
        if not all(c.get("same_script") is not None and c.get("cross_script") is not None for c in cells):
            continue
        gb = cells[0]["same_script"] - cells[0]["cross_script"]
        gs = cells[1]["same_script"] - cells[1]["cross_script"]
        print(f"  {size:6} same−cross P@1 gap: byte {gb:+.3f}  subword {gs:+.3f}  "
              f"(byte−subword {gb - gs:+.3f}; positive = byte more script-bound)")
    print("\n  (3) romanization robustness: Δ nDCG@10 romanized − native (paired bootstrap)")
    from byte_embed.stats import iter_perquery
    for n in MAIN_MODELS:
        r = M.get(n)
        base = _baseline_for(n, results)
        if not (r and r.get("roman_battery") and base):
            continue
        if not dict(iter_perquery(base)):
            print(f"  {n:15}native baseline has no per-query scores -> skipped")
            continue
        rows = compare(r["roman_battery"], base, n_boot)
        cells = "  ".join(f"{c[0][:6]}-{c[1]}:{x['delta']:+.3f}{'*' if x['significant'] else ''}"
                          for c, x in rows)
        print(f"  {n:15}{cells}")
    print("  reading: larger same-script gap AND larger romanization drop for byte -> script-mediated.")


def _selftest():
    langs = ["sw", "rw", "yo", "ha", "en", "te", "bn", "am", "zh", "ar"]
    m = {a: {b: (0.9 if SCRIPT[a] == SCRIPT[b] else 0.3) for b in langs if b != a} for a in langs}
    c = script_cells(m, langs)
    assert c["same_script"] == 0.9 and c["cross_script"] == 0.3 and c["n_same"] == 10 and c["n_cross"] == 35, c
    assert c["sw_rw"] == 0.9 and c["ha_am"] == 0.3 and c["yo_latin"] == 0.9
    assert check_romanized(["telugu vakyam", "inko"], "te") == 1.0
    for bad in (["", "", "ok"], ["ok తెలుగు", "ok"]):           # empty / source script survived
        try:
            check_romanized(bad, "te")
            raise AssertionError("should have refused")
        except SystemExit:
            pass
    assert check_romanized(["zhonghua 1ge"], "zh") == 1.0 and not _source_script_left("café", "zh")
    class Fake:
        def romanize_string(self, t, lcode):
            return "rom:" + t
    rz = Romanizer("te", cache_dir="results/_selftest_tmp", engine=Fake())
    assert rz.many(["a", "b"]) == ["rom:a", "rom:b"] and rz.cache["a"] == "rom:a"
    rz.cp.unlink(missing_ok=True)
    print("selftest OK: script cells, romanization guard, cached romanizer")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-battery", action="store_true", help="shift + cells only")
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
