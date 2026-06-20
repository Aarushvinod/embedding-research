"""Expanded GATE-GRAFT token-graft study + consolidation.

  - graft (init=graft) across a typologically/script-diverse language set
  - random-init baseline (subset) -> proves the alignment is what makes the graft work
  - multilingual-e5 topline -> recovery ratio (how much of "trained on the language")
  - Catalan vocab-size sweep -> coverage effect

Writes results/matrix.json and prints a consolidated table. Long-running -> run in background.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from gate_graft import token_graft as TG
from gate_graft import topline as TL

# diverse: Romance, Germanic, Slavic/Cyrillic, Greek, Turkic, Uralic, Semitic x2,
# Indic x2, Austronesian, Bantu  (all in fastText-157 + OPUS-100 + MUSE)
LANGS = ["ca", "de", "ru", "el", "tr", "fi", "ar", "he", "hi", "bn", "id"]  # sw absent from OPUS-100
RANDOM_SUBSET = ["ca", "tr", "fi", "bn"]
SWEEP = [5000, 15000, 30000]


def _clean():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)  # write results under the repo, not the CWD
    rows: dict = {}
    for l in LANGS:
        try:
            r = TG.run(l, out=f"results/tg_{l}.json")
            if "error" not in r:
                rows[l] = {"graft_p1": r["sent_p1"], "conf_p25": r["conf_P@25%"],
                           "graft_n": r["grafted_tokens"], "n": r["n_eval"]}
        except Exception as e:  # noqa: BLE001
            print(f"{l} graft FAILED: {type(e).__name__}: {e}")
        _clean()

    for l in RANDOM_SUBSET:
        try:
            r = TG.run(l, init="random", out=f"results/tg_{l}_random.json")
            if l in rows:
                rows[l]["random_p1"] = r["sent_p1"]
        except Exception as e:  # noqa: BLE001
            print(f"{l} random FAILED: {type(e).__name__}: {e}")
        _clean()

    try:
        tl = {x["lang"]: x["topline_p1"]
              for x in TL.run(LANGS, out="results/topline_all.json")}
        for l in rows:
            rows[l]["topline_p1"] = tl.get(l)
    except Exception as e:  # noqa: BLE001
        print(f"topline FAILED: {e}")
    _clean()

    sweep: dict = {}
    for k in SWEEP:
        try:
            sweep[k] = TG.run("ca", align_k=k, graft_n=k,
                              out=f"results/tg_ca_k{k}.json")["sent_p1"]
        except Exception as e:  # noqa: BLE001
            print(f"sweep {k} FAILED: {type(e).__name__}: {e}")
        _clean()

    Path("results/matrix.json").write_text(
        json.dumps({"langs": rows, "ca_vocab_sweep": sweep}, indent=2))

    print("\n=== MATRIX ===")
    print(f"{'lang':5}{'graft':>7}{'topline':>9}{'recov%':>8}{'random':>8}"
          f"{'conf@25':>9}{'graftN':>8}")
    for l, d in rows.items():
        rec = round(100 * d["graft_p1"] / d["topline_p1"]) if d.get("topline_p1") else "-"
        print(f"{l:5}{d['graft_p1']:>7}{str(d.get('topline_p1', '-')):>9}{str(rec):>8}"
              f"{str(d.get('random_p1', '-')):>8}{d['conf_p25']:>9}{d['graft_n']:>8}")
    print("\nca vocab sweep (graft_n -> p@1):", sweep)


if __name__ == "__main__":
    main()
