"""Exp 3 — language-identity geometry -> erasure (language-neutrality): is byte's cross-lingual edge
a more language-neutral space?

Multilingual encoders represent language identity, to first order, as a per-language mean offset
in a shared subspace (Chang, Tu & Bergen 2022; Libovicky et al. 2020). On the FLORES 10-way parallel
sentences (same content in every language) we measure, per layer: language centroids, the share of
variance they explain, linear language-ID probe accuracy, and how much the languages share a
subspace after centering. Then we ERASE the language component from the final embeddings — plain
centroid-difference projection and LEACE (Belrose et al. 2023, linear guardedness) — and re-run the
20k-pool retrieval battery with the erased encoder. Cross-lingual cells (CIRAL en->ha, AfriQA
X->en) are the TREATMENT: erasing language identity should move them. Monolingual cells (MIRACL,
Amharic-PR, Belebele) are the NEGATIVE CONTROL: query and candidates share a language, so the
language component is common-mode and rankings should barely change; if they DO drop, the erased
subspace was entangled with content (flag, don't over-read). Reading rule (pre-registered): byte's
cross-lingual margin surviving erasure better than subword's => byte's space is more
language-neutral. The un-erased `none` variant doubles as the end-to-end check that this module's
loading + encoding reproduces the stored results (max |dnDCG@10| <= 0.005).

  python -m byte_embed.interp_langgeom --only byte-small
  python -m byte_embed.interp_langgeom --only subword-small --variants none   # loader check only
  python -m byte_embed.interp_langgeom --merge
  python -m byte_embed.interp_langgeom --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from byte_embed.interp_common import (MAIN_MODELS, apply_projection, fit_leace, fit_mean_diff,
                                      flores_parallel, flores_xling, lang_probe, layer_pooled,
                                      load_student, make_encode, merge_parts, models_in, part_path,
                                      read_json, write_json)

ANALYSIS = "langgeom"
VARIANTS = ("none", "mean_diff", "leace")
MIRACL_LANGS = ["en", "zh", "ar", "te", "bn", "sw", "yo"]
QA_BENCH = ("amharicpr", "ciral", "afriqa")
TREATMENT = {("ciral", "ha"), ("afriqa", "rw"), ("afriqa", "ha"), ("afriqa", "sw"), ("afriqa", "yo")}


def is_treatment(cell):
    return tuple(cell) in TREATMENT


def fits_path(name):
    return part_path(ANALYSIS, name).with_suffix(".fits.npz")


# ----------------------------------------------------------------------------------------------
# geometry (pure numpy)
# ----------------------------------------------------------------------------------------------
def language_geometry(X, labels, seed=0, k=16, standardize=True):
    """Per-layer stats: share of variance explained by language centroids, linear language-ID probe
    accuracy, and mean pairwise subspace overlap of the per-language clouds after centering
    (overlap = ||B_a^T B_b||_F^2 / k over top-k principal directions; 1 = identical subspaces)."""
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    labs = sorted(set(labels.tolist()))
    mu = X.mean(0)
    C = np.stack([X[labels == l].mean(0) for l in labs], 0)
    w = np.array([(labels == l).mean() for l in labs])
    between = float((w * ((C - mu) ** 2).sum(1)).sum())
    total = float(((X - mu) ** 2).sum(1).mean())
    bases = []
    for j, l in enumerate(labs):
        Xc = X[labels == l] - C[j]
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        bases.append(Vt[:k].T)
    ov = [float(((bases[a].T @ bases[b]) ** 2).sum()) / k
          for a in range(len(labs)) for b in range(a + 1, len(labs))]
    return {"centroid_var_ratio": round(between / max(total, 1e-12), 4),
            "lang_probe_acc": lang_probe(X, labels, seed=seed),
            "subspace_overlap": round(float(np.mean(ov)), 4) if ov else None}


def erasure_fits(F, labels):
    """{variant: (P, mu)} on the FINAL embeddings (FLORES; held out from every eval pool)."""
    return {"mean_diff": fit_mean_diff(F, labels), "leace": fit_leace(F, labels)}


def _xling_summary(xl):
    return {a: {b: {"p@1": r["p@1"], "ndcg@10": r["ndcg@10"]} for b, r in row.items()}
            for a, row in xl.items()}


def flores_variants(F_by_lang, fits):
    """FLORES cross-lingual retrieval (P@1 / nDCG@10 per language pair) for: raw, centered (subtract
    each side's OWN centroid — legal here because the language of every sentence is known), and
    each erasure projection. Also the post-erasure language-probe accuracy on the final embeddings."""
    langs = list(F_by_lang)
    X = np.concatenate([F_by_lang[l] for l in langs], 0)
    labels = np.concatenate([[l] * len(F_by_lang[l]) for l in langs])
    out = {"raw": _xling_summary(flores_xling(F_by_lang, langs))}
    cen = {l: F_by_lang[l] - F_by_lang[l].mean(0) for l in langs}
    out["centered"] = _xling_summary(flores_xling({l: _l2(v) for l, v in cen.items()}, langs))
    probes = {"raw": lang_probe(X, labels)}
    for v, (P, mu) in fits.items():
        Xe = apply_projection(X, P, mu)
        probes[v] = lang_probe(Xe, labels)
        E = {l: _l2(apply_projection(F_by_lang[l], P, mu)) for l in langs}
        out[v] = _xling_summary(flores_xling(E, langs))
    return out, probes


def _l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def xling_mean(m):
    return round(float(np.mean([r["p@1"] for row in m.values() for r in row.values()])), 4)


# ----------------------------------------------------------------------------------------------
# battery (20k-pool training-time protocol) + loader identity check
# ----------------------------------------------------------------------------------------------
def battery(enc, ckpt_dir):
    from byte_embed.config import STUDY_LANGS
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    return {"belebele": eval_battery(enc, STUDY_LANGS).get("belebele"),
            "miracl": eval_miracl_langs(enc, MIRACL_LANGS, n_queries=250, distractors=20000,
                                        cache_dir=ckpt_dir),
            "qa_retrieval": eval_qa_retrieval(enc, benchmarks=QA_BENCH, n_queries=250,
                                              distractors=20000, cache_dir=ckpt_dir)}


def identity_check(mine, stored, tol=0.005):
    """max |nDCG@10 difference| over shared (benchmark, lang) cells between a freshly scored `none`
    battery and the stored training-time results — proves loader + tokenization replication."""
    from byte_embed.stats import iter_perquery
    a, b = dict(iter_perquery(mine)), dict(iter_perquery(stored))
    diffs = {}
    for cell in set(a) & set(b):
        qa, qb = a[cell], b[cell]
        ks = sorted(set(qa) & set(qb))
        if ks:
            diffs[cell] = abs(float(np.mean([qa[k] for k in ks])) - float(np.mean([qb[k] for k in ks])))
    worst = max(diffs.values()) if diffs else None
    return {"max_abs_diff": (round(worst, 4) if worst is not None else None), "cells": len(diffs),
            "pass": bool(worst is not None and worst <= tol)}


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, variants=VARIANTS, seed=0, n_probe=400,
            skip_battery=False):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("variants") and all(v in res["variants"] for v in variants) and "layers" in res:
        print(f"=== {ANALYSIS}/{name}: already done -> skip ===")
        return
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    res.update(model=name, kind=bm.get("kind"), steps_run=bm.get("steps_run"))
    par = flores_parallel(cache_dir=ckpt_dir)
    langs = list(par)
    n = len(par[langs[0]])
    texts = [par[l][i] for l in langs for i in range(n)]
    labels = np.array([l for l in langs for _ in range(n)])

    if "layers" not in res:
        states, layer_ids, Ffinal = layer_pooled(student, texts, device=device,
                                                 batch_size=(4 if "large" in name else 8))
        rng = np.random.default_rng(seed)
        sub = np.concatenate([rng.choice(np.flatnonzero(labels == l), size=min(n_probe, n),
                                         replace=False) for l in langs])
        res["layers"] = [{"layer": int(l), **language_geometry(states[j][sub], labels[sub], seed)}
                         for j, l in enumerate(layer_ids)]
        res["final_geometry"] = language_geometry(Ffinal, labels, seed, standardize=False)
        F_by_lang = {l: Ffinal[labels == l] for l in langs}
        fits = erasure_fits(Ffinal, labels)
        res["flores_xling"], res["probe_after_erasure"] = flores_variants(F_by_lang, fits)
        res["flores_xling_mean"] = {v: xling_mean(m) for v, m in res["flores_xling"].items()}
        np.savez(fits_path(name), **{f"{v}_P": P for v, (P, mu) in fits.items()},
                 **{f"{v}_mu": mu for v, (P, mu) in fits.items()})
        write_json(outp, res)
        del states
        print(f"  geometry: final probe={res['final_geometry']['lang_probe_acc']} "
              f"after erasure={res['probe_after_erasure']}  FLORES P@1={res['flores_xling_mean']}")
    fz = np.load(fits_path(name))
    fits = {v: (fz[f"{v}_P"], fz[f"{v}_mu"]) for v in ("mean_diff", "leace")}

    if skip_battery:
        return
    res.setdefault("variants", {})
    for v in variants:
        if v in res["variants"]:
            continue
        if v == "none":
            enc = make_encode(student, device)
        else:
            P, mu = fits[v]
            enc = make_encode(student, device, post=lambda E, P=P, mu=mu: apply_projection(E, P, mu))
        print(f"=== {name}: battery, variant={v} ===")
        res["variants"][v] = battery(enc, ckpt_dir)
        if v == "none":
            res["identity_check"] = identity_check(res["variants"]["none"], bm)
            ic = res["identity_check"]
            print(f"  [loader check] max|dnDCG@10| = {ic['max_abs_diff']} over {ic['cells']} cells -> "
                  f"{'PASS' if ic['pass'] else 'WARN: does not reproduce stored results'}")
        write_json(outp, res)
    print(f"  saved -> {outp}")


# ----------------------------------------------------------------------------------------------
def merge(n_boot=10000):
    from byte_embed.stats import compare
    d = merge_parts(ANALYSIS)
    M = d["models"]
    print("\nEXP 3 — LANGUAGE GEOMETRY (final embeddings) + ERASURE")
    print(f"  {'model':15}{'probe raw':>10}{'probe md':>9}{'probe leace':>12}{'centroid var':>13}"
          f"{'overlap':>8}{'FLORES P@1 raw/cen/md/leace':>30}{'loader':>8}")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r or "final_geometry" not in r:
            continue
        g, pa, fx = r["final_geometry"], r.get("probe_after_erasure", {}), r.get("flores_xling_mean", {})
        ic = r.get("identity_check", {})
        nan = float("nan")
        fx_str = "/".join(f"{fx.get(k, nan):.3f}" for k in ("raw", "centered", "mean_diff", "leace"))
        loader = "PASS" if ic.get("pass") else ("WARN" if ic else "—")
        print(f"  {n:15}{g['lang_probe_acc']:>10.3f}{pa.get('mean_diff', nan):>9.3f}"
              f"{pa.get('leace', nan):>12.3f}{g['centroid_var_ratio']:>13.4f}"
              f"{(g['subspace_overlap'] or nan):>8.3f}{fx_str:>30}{loader:>8}")

    print("\n  ERASURE Δ nDCG@10 vs `none` (paired bootstrap; T = cross-lingual TREATMENT cells, "
          "C = monolingual CONTROL cells)")
    summary = {}
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r or "none" not in r.get("variants", {}):
            continue
        for v in ("mean_diff", "leace"):
            if v not in r["variants"]:
                continue
            rows = compare(r["variants"][v], r["variants"]["none"], n_boot)
            t = [x["delta"] for c, x in rows if is_treatment(c)]
            c = [x["delta"] for c_, x in rows if not is_treatment(c_)]
            summary[(n, v)] = (np.mean(t) if t else None, np.mean(c) if c else None)
            print(f"  {n:15}{v:10} T mean Δ={_f(summary[(n, v)][0])}  C mean Δ={_f(summary[(n, v)][1])}")
            for cell, x in rows:
                if is_treatment(cell) or x["significant"]:
                    star = "*" if x["significant"] else ""
                    print(f"      {'T' if is_treatment(cell) else 'C'} {cell[0]:10}{cell[1]:4}"
                          f"{x['delta']:>+8.4f} [{x['ci_low']:+.3f},{x['ci_high']:+.3f}] {star}")
    print("\n  byte − subword per size: how much MORE of byte's cross-lingual score survives erasure "
          "(positive = byte more language-neutral)")
    for v in ("mean_diff", "leace"):
        for size in ("small", "base", "large"):
            b, s = summary.get((f"byte-{size}", v)), summary.get((f"subword-{size}", v))
            if b and s and b[0] is not None and s[0] is not None:
                print(f"  {v:10}{size:6} T Δ(byte)−Δ(subword) = {b[0] - s[0]:+.4f}")
    print("  reading: erasure moves T but not C -> mechanism real; C also drops -> erased subspace "
          "entangled with content (flag).")


def _f(x):
    return "—" if x is None else f"{x:+.4f}"


def _selftest():
    rng = np.random.default_rng(0)
    d, K, n = 48, 10, 120
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    labels = np.repeat(np.arange(K), n).astype(str)
    X = rng.standard_normal((K * n, d)) * 0.4 + 2.5 * Q[:, :K][:, np.repeat(np.arange(K), n)].T
    g = language_geometry(X, labels)
    g0 = language_geometry(X, rng.permutation(labels))        # shuffled labels = null control
    assert g["lang_probe_acc"] > 0.95 and g["centroid_var_ratio"] > 0.3, g
    assert g0["lang_probe_acc"] < 0.25 and g0["centroid_var_ratio"] < 0.05, g0
    fits = erasure_fits(X, labels)
    F = {str(k): _l2(X[labels == str(k)]) for k in range(K)}
    xl, probes = flores_variants(F, fits)
    assert probes["raw"] > 0.95 and probes["leace"] < 0.2 and probes["mean_diff"] < 0.2, probes
    assert set(xl) == {"raw", "centered", "mean_diff", "leace"}
    assert is_treatment(("afriqa", "rw")) and not is_treatment(("miracl", "te"))
    mine = {"miracl": {"per_lang": {"te": {"per_query": {"q1": 0.5, "q2": 0.7}}}}}
    stored = {"miracl": {"per_lang": {"te": {"per_query": {"q1": 0.5, "q2": 0.704}}}}}
    ic = identity_check(mine, stored)
    assert ic["pass"] and ic["cells"] == 1, ic
    print("selftest OK: geometry stats, erasure -> chance probes, treatment cells, identity check")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-probe", type=int, default=400, help="FLORES sentences/lang for per-layer probes")
    ap.add_argument("--skip-battery", action="store_true", help="geometry + erasure fits only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge()
    variants = tuple(v for v in a.variants.split(",") if v)
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, variants, a.seed, a.n_probe, a.skip_battery)


if __name__ == "__main__":
    main()
