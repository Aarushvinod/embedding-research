"""Exp 2 — alignment / uniformity decomposition (objective): which InfoNCE term each gap lives in.

InfoNCE asymptotically optimizes exactly two properties of an embedding space (Wang & Isola 2020):
ALIGNMENT — positive pairs sit close (E||f(x)-f(x+)||^2) — and UNIFORMITY — embeddings spread
evenly on the hypersphere (log E exp(-2||f(x)-f(y)||^2) over random pairs). Restating every
byte-vs-subword gap in those two terms says whether a gap is a "positives closer" effect or a
"space better spread" effect. Positive pairs come from three sources the study already has:
(a) student(sentence) vs its cached BGE-M3 teacher target — alignment-to-teacher per language;
(b) FLORES translation pairs — cross-lingual alignment for every language pair; (c) query vs gold
passage from the cached 20k retrieval pools — task alignment. Uniformity is reported per language
and per pool. Reading rule (pre-registered): a term that moves byte-vs-subword in step with the
retrieval gap is the term the gap lives in.

  python -m byte_embed.interp_alignuni --only byte-small
  python -m byte_embed.interp_alignuni --merge
  python -m byte_embed.interp_alignuni --selftest
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from byte_embed.interp_common import (MAIN_MODELS, alignment, flores_parallel, flores_xling,
                                      load_student, make_encode, merge_parts, models_in, part_path,
                                      read_json, sample_training_sentences, uniformity, utf8_stdout,
                                      write_json)

ANALYSIS = "alignuni"
POOL_TAG = "250q_20000d"          # the training-time 20k-pool caches


def gold_pairs(cache, cap=2000, seed=0):
    """(query text, gold passage text) pairs from a pool-cache dict; capped, seeded order."""
    id2text = dict(zip(cache["pool_id"], cache["pool_text"]))
    pairs = [(cache["queries"][q], id2text[r]) for q in cache["queries"]
             for r in cache["rel"].get(q, []) if r in id2text]
    rng = np.random.default_rng(seed)
    if len(pairs) > cap:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=cap, replace=False)]
    return pairs


def _pool_key(path):
    stem = Path(path).stem                     # miracl_te_250q_20000d_0 | qa_afriqa_rw_250q_20000d_0
    return stem.split(f"_{POOL_TAG}")[0]


def run_one(name, results, ckpt_dir, device, n_train=1000, n_pool=2000, seed=0):
    outp = part_path(ANALYSIS, name)
    if outp.exists():
        print(f"=== {ANALYSIS}/{name}: already done -> skip ===")
        return
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    enc = make_encode(student, device, batch_size=(32 if bm.get("kind") == "byte" else 128))
    res = {"model": name, "kind": bm.get("kind"), "steps_run": bm.get("steps_run")}

    # (a) alignment to the teacher target + uniformity, per language
    try:
        sents, sl, T = sample_training_sentences(ckpt_dir, per_lang=n_train, seed=seed, results=results)
        E, sl = enc(sents), np.asarray(sl)
        res["teacher_align"] = {l: alignment(E[sl == l], T[sl == l]) for l in sorted(set(sl.tolist()))}
        res["uniformity"] = {l: uniformity(E[sl == l], seed=seed) for l in sorted(set(sl.tolist()))}
        res["uniformity"]["all"] = uniformity(E, seed=seed)
        res["teacher_align"]["mean"] = round(float(np.mean(list(res["teacher_align"].values()))), 4)
    except SystemExit as e:
        print(f"  [alignuni] {e} -> teacher-alignment section skipped")

    # (b) FLORES cross-lingual alignment (translation pairs) + cross-lingual retrieval P@1
    par = flores_parallel(cache_dir=ckpt_dir)
    F = {l: enc(par[l]) for l in par}
    langs = list(par)
    res["xling_align"] = {a: {b: alignment(F[a], F[b]) for b in langs if b != a} for a in langs}
    res["flores_uniformity"] = {l: uniformity(F[l], seed=seed) for l in langs}
    xl = flores_xling(F, langs)
    res["flores_xling"] = {a: {b: {k: v for k, v in r.items() if k != "per_query"}
                               for b, r in row.items()} for a, row in xl.items()}
    res["xling_align_mean"] = round(float(np.mean([v for row in res["xling_align"].values()
                                                   for v in row.values()])), 4)

    # (c) query <-> gold-passage alignment + pool uniformity from the cached 20k pools
    res["pool_align"], res["pool_uniformity"] = {}, {}
    caches = sorted(glob.glob(os.path.join(ckpt_dir, f"miracl_*_{POOL_TAG}_0.json"))
                    + glob.glob(os.path.join(ckpt_dir, f"qa_*_{POOL_TAG}_0*.json")))
    rng = np.random.default_rng(seed)
    for cp in caches:
        d = read_json(cp)
        pairs = gold_pairs(d, cap=n_pool, seed=seed)
        if not pairs:
            continue
        key = _pool_key(cp)
        Eq, Ed = enc([p[0] for p in pairs]), enc([p[1] for p in pairs])
        res["pool_align"][key] = alignment(Eq, Ed)
        samp = rng.choice(len(d["pool_text"]), size=min(n_pool, len(d["pool_text"])), replace=False)
        res["pool_uniformity"][key] = uniformity(enc([d["pool_text"][i] for i in samp]), seed=seed)
    write_json(outp, res)
    print(f"  saved {name}: teacher_align={res.get('teacher_align', {}).get('mean')}  "
          f"xling_align={res['xling_align_mean']}  pools={len(res['pool_align'])} -> {outp}")


def _fmt(x):
    return f"{x:>10.4f}" if isinstance(x, (int, float)) else f"{'—':>10}"


def merge():
    d = merge_parts(ANALYSIS)
    M = d["models"]
    print("\nEXP 2 — ALIGNMENT (lower = positives closer) / UNIFORMITY (more negative = better spread)")
    print(f"  {'model':15}{'align→teacher':>14}{'unif(all)':>10}{'align xling':>12}"
          f"{'align pool':>11}{'unif pool':>10}")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r:
            continue
        pa = np.mean(list(r["pool_align"].values())) if r.get("pool_align") else None
        pu = np.mean(list(r["pool_uniformity"].values())) if r.get("pool_uniformity") else None
        print(f"  {n:15}{_fmt(r.get('teacher_align', {}).get('mean')):>14}"
              f"{_fmt(r.get('uniformity', {}).get('all'))}{_fmt(r.get('xling_align_mean')):>12}"
              f"{_fmt(pa):>11}{_fmt(pu)}")
    print("\n  byte − subword per size (negative alignment Δ = byte's positives closer; "
          "negative uniformity Δ = byte better spread)")
    for size in ("small", "base", "large"):
        b, s = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        if not (b and s):
            continue
        row = []
        for label, get in (("teacher", lambda r: r.get("teacher_align", {}).get("mean")),
                           ("unif", lambda r: r.get("uniformity", {}).get("all")),
                           ("xling", lambda r: r.get("xling_align_mean")),
                           ("pool", lambda r: (np.mean(list(r["pool_align"].values()))
                                              if r.get("pool_align") else None))):
            x, y = get(b), get(s)
            row.append(f"{label} {x - y:+.4f}" if x is not None and y is not None else f"{label} —")
        print(f"  {size:6}  " + "   ".join(row))
    report_centricity(M)


def hubness(A, X):
    """English-centricity of a cross-lingual space from the FLORES pair matrices.
    A[a][b] = translation-pair alignment (lower = closer); X[a][b]["p@1"] = retrieval a->b.
    Returns means over ordered pairs that involve English vs pairs between two non-English
    languages, the mean rank of English among each non-English language's partners by alignment
    (1 = English is its nearest language; 5 = no hub), and how many languages have English nearest."""
    langs = list(A)
    p1 = {a: {b: X[a][b]["p@1"] for b in X[a]} for a in X}
    pairs = [(a, b) for a in langs for b in langs if a != b]
    en = [(a, b) for a, b in pairs if "en" in (a, b)]
    non = [(a, b) for a, b in pairs if "en" not in (a, b)]
    ranks = []
    for l in langs:
        if l == "en":
            continue
        order = [b for _, b in sorted((A[l][b], b) for b in langs if b != l)]
        ranks.append(order.index("en") + 1)
    return {"p1_en": float(np.mean([p1[a][b] for a, b in en])),
            "p1_non": float(np.mean([p1[a][b] for a, b in non])),
            "align_en": float(np.mean([A[a][b] for a, b in en])),
            "align_non": float(np.mean([A[a][b] for a, b in non])),
            "en_hub_rank": float(np.mean(ranks)), "en_nearest_for": int(sum(r == 1 for r in ranks)),
            "n_non_en": len(ranks), "p1": p1}


def report_centricity(M):
    from byte_embed.interp_script import script_cells
    print("\n  ENGLISH-CENTRICITY (FLORES 10-way): pairs involving English vs pairs between two non-English "
          "languages; hub rank = rank of English among each language's 9 partners by alignment (1 = nearest)")
    print(f"  {'model':15}{'P@1 en':>8}{'P@1 non':>9}{'align en':>10}{'align non':>11}{'hub rank':>10}{'en nearest':>12}"
          f"   |{'same-scr':>9}{'cross':>7}{'sw-rw':>7}{'ha-am':>7}{'yo-Lat':>7}  (P@1: same=Latin-Latin pairs)")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r or not r.get("xling_align") or not r.get("flores_xling"):
            continue
        h = hubness(r["xling_align"], r["flores_xling"])
        sc = script_cells(h["p1"], list(r["xling_align"]))
        nan = float("nan")
        print(f"  {n:15}{h['p1_en']:>8.3f}{h['p1_non']:>9.3f}{h['align_en']:>10.3f}{h['align_non']:>11.3f}"
              f"{h['en_hub_rank']:>10.2f}{h['en_nearest_for']:>8}/{h['n_non_en']:<3}"
              f"   |{sc['same_script'] if sc['same_script'] is not None else nan:>9.3f}"
              f"{sc['cross_script'] if sc['cross_script'] is not None else nan:>7.3f}"
              + "".join(f"{(sc[k] if sc[k] is not None else nan):>7.3f}" for k in ("sw_rw", "ha_am", "yo_latin")))
    print("  reading: P@1 en >> non and hub rank near 1 -> English-centric; same-script >> cross-script -> "
          "script-bound (exp 4's cell analysis, computed from exp 2's matrices).")


def _selftest():
    cache = {"queries": {"q1": "hello", "q2": "world"}, "rel": {"q1": ["d1", "d9"], "q2": ["d2"]},
             "pool_id": ["d1", "d2", "d3"], "pool_text": ["A", "B", "C"]}
    pairs = gold_pairs(cache)
    assert sorted(pairs) == [("hello", "A"), ("world", "B")], pairs     # missing d9 dropped
    assert _pool_key("checkpoints/miracl_te_250q_20000d_0.json") == "miracl_te"
    assert _pool_key("checkpoints/qa_afriqa_rw_250q_20000d_0.json") == "qa_afriqa_rw"
    assert _pool_key("checkpoints/qa_ciral_ha_250q_20000d_0_s800000.json") == "qa_ciral_ha"
    print("selftest OK: gold-pair extraction + pool keys")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-train", type=int, default=1000, help="cached training sentences per language")
    ap.add_argument("--n-pool", type=int, default=2000, help="max gold pairs / pool sample per cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge()
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.n_train, a.n_pool, a.seed)


if __name__ == "__main__":
    main()
