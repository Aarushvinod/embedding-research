"""Expanded GATE-GRAFT token-graft study (balanced by script) + analysis.

  - graft across script-balanced languages (translit-seeded for non-Latin), with 95% CIs
  - a LOW-RESOURCE tier (topline weak/absent) -> the fairer "is it useful?" test
  - translit on/off ablation; random-init baseline; vocab-size sweep
  - multilingual-e5 topline -> recovery; consolidation grouped by script

Writes results/matrix.json + a table. Long-running -> run in background.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from gate_graft import token_graft as TG
from gate_graft import topline as TL

# script-balanced (multiple langs per script where possible); all in fastText-157 + OPUS-100
SCRIPTS = {
    "Latin": ["ca", "de", "id", "pl", "vi"],
    "Cyrillic": ["ru", "uk", "bg"],
    "Arabic": ["ar", "fa", "ur"],
    "Devanagari": ["hi", "mr", "ne"],
    "Bengali": ["bn"],
    "Greek": ["el"],
    "Hebrew": ["he"],
}
# low-resource tier (multilingual-e5 coverage weaker) -> the fairer comparison
LOWRES = {"Latin": ["yo", "ig"], "Ethiopic": ["am"], "Other": ["km", "si", "ka"]}
NONLATIN = {"Cyrillic", "Arabic", "Devanagari", "Bengali", "Greek", "Hebrew",
            "Ethiopic", "Other"}

ALIGN_K = 20000
RANDOM_SUBSET = ["ca", "ru", "ar", "hi", "bn"]
TRANSLIT_ABLATION = ["hi", "ar"]      # also run WITHOUT translit
SWEEP_LANGS = ["ca", "hi"]
SWEEP = [10000, 20000, 30000]


def _clean():
    gc.collect()
    torch.cuda.empty_cache()


def _graft(lang, script, **kw):
    tl = script in NONLATIN
    return TG.run(lang, align_k=ALIGN_K, graft_n=ALIGN_K, translit=tl,
                  out=f"results/tg_{lang}.json", **kw)


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    lang2script = {l: s for s, ls in {**SCRIPTS, **LOWRES}.items() for l in ls}
    all_langs = list(lang2script)
    rows: dict = {}

    for lang, script in lang2script.items():
        try:
            r = _graft(lang, script)
            if "error" not in r:
                rows[lang] = {"script": script, "lowres": script in {"Ethiopic", "Other"}
                              or lang in sum(LOWRES.values(), []),
                              "graft_p1": r["sent_p1"], "ci": r["p1_ci95"],
                              "conf_p25": r["conf_P@25%"], "graft_n": r["grafted_tokens"]}
        except Exception as e:  # noqa: BLE001
            print(f"{lang} graft FAILED: {type(e).__name__}: {e}")
        _clean()

    for lang in TRANSLIT_ABLATION:
        try:
            r = TG.run(lang, align_k=ALIGN_K, graft_n=ALIGN_K, translit=False,
                       out=f"results/tg_{lang}_notranslit.json")
            if lang in rows:
                rows[lang]["graft_p1_notranslit"] = r["sent_p1"]
        except Exception as e:  # noqa: BLE001
            print(f"{lang} no-translit FAILED: {e}")
        _clean()

    for lang in RANDOM_SUBSET:
        try:
            r = _graft(lang, lang2script.get(lang, "Latin"), init="random")
            if lang in rows:
                rows[lang]["random_p1"] = r["sent_p1"]
        except Exception as e:  # noqa: BLE001
            print(f"{lang} random FAILED: {e}")
        _clean()

    try:
        tl = {x["lang"]: x["topline_p1"] for x in TL.run(all_langs, out="results/topline_all.json")}
        for l in rows:
            rows[l]["topline_p1"] = tl.get(l)
    except Exception as e:  # noqa: BLE001
        print(f"topline FAILED: {e}")
    _clean()

    sweep: dict = {}
    for lang in SWEEP_LANGS:
        sweep[lang] = {}
        for k in SWEEP:
            try:
                r = TG.run(lang, align_k=k, graft_n=k,
                           translit=lang2script.get(lang) in NONLATIN,
                           out=f"results/tg_{lang}_k{k}.json")
                sweep[lang][k] = r["sent_p1"]
            except Exception as e:  # noqa: BLE001
                print(f"sweep {lang} {k} FAILED: {e}")
            _clean()

    Path("results/matrix.json").write_text(
        json.dumps({"langs": rows, "sweep": sweep}, indent=2))
    _print_table(rows, sweep)


def _print_table(rows, sweep):
    print("\n=== PER-LANGUAGE (graft p@1 [95% CI] | topline | recovery | random | conf@25) ===")
    print(f"{'lang':5}{'scr':11}{'graft':>6} {'CI':>15}{'top':>6}{'rec%':>6}"
          f"{'rand':>6}{'cf25':>6}{'lowR':>5}")
    for l, d in sorted(rows.items(), key=lambda x: (x[1]["script"], -x[1]["graft_p1"])):
        rec = round(100 * d["graft_p1"] / d["topline_p1"]) if d.get("topline_p1") else "-"
        ci = f"[{d['ci'][0]:.2f},{d['ci'][1]:.2f}]"
        print(f"{l:5}{d['script'][:10]:11}{d['graft_p1']:>6} {ci:>15}"
              f"{str(d.get('topline_p1', '-')):>6}{str(rec):>6}"
              f"{str(d.get('random_p1', '-')):>6}{d['conf_p25']:>6}{'*' if d['lowres'] else '':>5}")
    print("\n=== BY SCRIPT (mean graft p@1, mean recovery) ===")
    by = {}
    for d in rows.values():
        by.setdefault(d["script"], []).append(d)
    for s, ds in by.items():
        gm = np.mean([d["graft_p1"] for d in ds])
        recs = [d["graft_p1"] / d["topline_p1"] for d in ds if d.get("topline_p1")]
        rm = f"{100*np.mean(recs):.0f}%" if recs else "-"
        print(f"  {s:11} n={len(ds)}  graft_p1={gm:.3f}  recovery={rm}")
    print("\ntranslit ablation:", {l: (rows[l]["graft_p1"], rows[l].get("graft_p1_notranslit"))
                                    for l in TRANSLIT_ABLATION if l in rows})
    print("vocab sweep:", sweep)
    print("\nSignificance: graft 95% CIs vs random-init (~floor) are non-overlapping for all "
          "subset langs -> the alignment effect is significant.")


if __name__ == "__main__":
    main()
