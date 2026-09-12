"""Exp 3 — is English load-bearing for the other languages? A causal intervention INSIDE the network.

Claim under test: an English-centric model represents non-English inputs partly through
English-associated features in its middle layers, so removing the English direction inside the
network should hurt every other language — and more so in the subword model than in the byte model.

Procedure (per model; directions fitted on FLORES, effects measured on Belebele + the 20k pools):
  1. At four depths (encoder blocks at 25/50/75/100% of the stack) collect residual-stream states at
     random positions of every FLORES sentence, labelled with the sentence's language, and fit a
     rank-1 LEACE eraser for the binary concept "English vs not" per depth — plus one eraser per
     other language and one random direction in the same whitened metric (the controls).
  2. Latent-English curve: an English-vs-rest logistic probe per depth, trained on half of the
     sentence ids and scored on the other half; per-language mean P(English). It shows whether
     non-English states look English-like mid-network and selects the intervention block
     (pre-registered rule: the block where non-English positions are most English-like).
  3. Intervene: a forward hook on the chosen block applies the eraser to every position of every
     input; the remaining blocks, the final LayerNorm, the trained pooler and the projection run on
     the edited activations. Re-score.
  4. Controls: an UNEDITED pass through the same loader (`none`) is the baseline every delta is
     measured against — so no number depends on the stored training-time file, and comparing `none`
     to that file is the loader identity check the study's 0.005-nDCG claim rests on. Plus the
     random direction and every other language's direction -> a 10x10 Belebele matrix of "effect on
     L of erasing M" at the chosen block (English/random at all four depths); `none`, English and
     random on the full 20k battery.
  5. Embedding-space picture: FLORES final embeddings, unedited and with English erased, saved for
     the t-SNE / PCA figure (gen_interp_figures.py), plus the per-language cosine between the edited
     and original embedding of every sentence.

Every reported delta is a PAIRED bootstrap over the Belebele queries (the cells carry per-query
nDCG@10), and the treatment and control means are taken over the SAME population — languages other
than English and other than the erased one — so the English excess is not a bookkeeping artifact.

Pre-registered readings: the English column hurts the other languages beyond the random and
own-direction controls, more for subword than byte -> English is load-bearing and byte depends on it
less. It hurts only English itself in both -> the English-centric story is wrong for both
architectures. Hurts both equally -> inherited from the teacher, tokenization irrelevant.
Stated limits: one linear direction at one depth (a distributed dependence would survive it); each
model's block is chosen by its own latent curve, so the cross-architecture contrast is ALSO reported
at matched depth fractions. Belebele's passages are drawn from FLORES, so the Belebele matrix is not
text-disjoint from the fitting set — but the erasers are fitted on 10-way PARALLEL sentences, which
holds content constant across the classes, so the class-mean difference cannot be passage content;
and the 20k-pool battery (MIRACL / Amharic-PR / CIRAL / AfriQA passages) IS disjoint from FLORES and
is reported as the out-of-sample confirmation of the same effect. `--flores-split dev` switches the
fitting set if full text-disjointness is wanted.

  python -m byte_embed.interp_english --only byte-small
  python -m byte_embed.interp_english --only subword-small --skip-battery   # erasers + latent + shift
  python -m byte_embed.interp_english --merge
  python -m byte_embed.interp_english --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from byte_embed.interp_common import (CROSS_CELLS, MAIN_MODELS, block_states, depth_blocks,
                                      flores_flat, flores_parallel, leace_direction, load_student,
                                      make_encode, merge_parts, models_in, n_blocks, part_path,
                                      rank1_eraser, rank1_torch_fn, read_json, stored_results,
                                      utf8_stdout, whitener, write_json)

ANALYSIS = "english"
SCHEMA = 1                         # bumped when the part-file contents change meaning
PER_TEXT = 10                      # sampled positions per FLORES sentence per depth
MIRACL_LANGS = ["en", "zh", "ar", "te", "bn", "sw", "yo"]
QA_BENCH = ("amharicpr", "ciral", "afriqa")
NONE = "none"                      # the unedited pass: baseline + loader identity check


def fits_path(name):
    return part_path(ANALYSIS, name).with_suffix(".erasers.npz")


def embeds_path(name):
    return part_path(ANALYSIS, name).with_suffix(".embeds.npz")


# ----------------------------------------------------------------------------------------------
# erasers + latent-English probe (numpy / sklearn)
# ----------------------------------------------------------------------------------------------
def fit_erasers(X, lang_rows, langs, seed=0, block=0):
    """One whitener per depth, then a rank-1 LEACE eraser per language ("L vs rest") and one random
    direction in the same whitened metric -> {name: (a, b, mu)}. The random control is drawn per
    BLOCK (seed, block), so the four depths are genuinely independent null draws."""
    W, Wp, mu = whitener(X)
    out = {l: rank1_eraser(W, Wp, mu, leace_direction(W, X, lang_rows == l)) for l in langs}
    rng = np.random.default_rng([seed, int(block)])
    out["random"] = rank1_eraser(W, Wp, mu, rng.standard_normal(X.shape[1]))
    return out


def latent_probe(X, lang_rows, fit_mask, seed=0):
    """English-vs-rest logistic probe on standardized per-position states, trained on the fit half
    of the sentence ids, scored on the other half -> balanced accuracy and per-language mean
    P(English) of the held-out positions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(lang_rows) == "en"
    tr, te = np.asarray(fit_mask, bool), ~np.asarray(fit_mask, bool)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    out = {"bacc": round(float(balanced_accuracy_score(y[te], p > 0.5)), 3), "p_en": {}}
    for l in sorted(set(np.asarray(lang_rows).tolist())):
        m = np.asarray(lang_rows)[te] == l
        out["p_en"][l] = round(float(p[m].mean()), 4) if m.any() else None
    return out


def non_en_mean(p_en):
    vals = [v for l, v in p_en.items() if l != "en" and v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def choose_block(latent, blocks):
    """Pre-registered: the depth where non-English positions are most English-like (the first max
    wins). All-NaN (no non-English held-out positions) falls back to the deepest block."""
    scores = [non_en_mean(latent[str(b)]["p_en"]) for b in blocks]
    if not np.isfinite(scores).any():
        return int(blocks[-1])
    return int(blocks[int(np.nanargmax(scores))])


def save_fits(path, fits):
    np.savez(path, **{f"{b}:{k}:{i}": v for b, er in fits.items()
                      for k, abm in er.items() for i, v in zip(("a", "b", "mu"), abm)})


def load_fits(path):
    z = np.load(path)
    fits = {}
    for key in z.files:
        b, k, i = key.split(":")
        fits.setdefault(int(b), {}).setdefault(k, {})[i] = z[key]
    return {b: {k: (v["a"], v["b"], v["mu"]) for k, v in er.items()} for b, er in fits.items()}


def belebele_plan(blocks, cb, langs):
    """(key, block, eraser) in priority order: the unedited baseline, the headline English/random
    pair at the chosen block, the nine control columns, then English/random at the other depths.
    `none` is block-independent (no hook), so it is computed exactly once."""
    plan = [(NONE, cb, NONE), (f"{cb}:en", cb, "en"), (f"{cb}:random", cb, "random")]
    plan += [(f"{cb}:{l}", cb, l) for l in langs if l != "en"]
    plan += [(f"{b}:{e}", b, e) for b in blocks if b != cb for e in ("en", "random")]
    return plan


def n_belebele_cells(n_langs=10, n_depths=4):
    """How many Belebele cells a complete run writes (status.sh reads this)."""
    return 1 + 2 + (n_langs - 1) + 2 * (n_depths - 1)


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, seed=0, skip_battery=False, flores_split="devtest"):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:            # part files from an older contract are discarded
        if res:
            print(f"  [english] {name}: part file schema {res.get('schema')} != {SCHEMA} -> recomputing")
        res = {"schema": SCHEMA}
    if res.get("battery") and all(e in res["battery"] for e in (NONE, "en", "random")):
        print(f"=== {ANALYSIS}/{name}: already done -> skip ===")
        return
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    res.update(model=name, kind=bm.get("kind"), steps_run=bm.get("steps_run"))
    n = n_blocks(student)
    blocks = depth_blocks(n)
    res["n_blocks"], res["blocks"] = n, blocks
    par = flores_parallel(cache_dir=ckpt_dir, split=flores_split)
    langs = list(par)
    texts, lang, sid = flores_flat(par)
    nsent = len(par[langs[0]])
    rng = np.random.default_rng(seed)
    fit_ids = np.zeros(nsent, dtype=bool)
    fit_ids[rng.choice(nsent, size=nsent // 2, replace=False)] = True
    big = "large" in name
    bs_enc = 32 if bm.get("kind") == "byte" else 128

    # ---- stage A: states at four depths -> erasers + latent-English probe -> chosen block
    if "latent" not in res or not fits_path(name).exists():
        states, index = block_states(student, texts, blocks, per_text=PER_TEXT, device=device,
                                     batch_size=(4 if big else 8), seed=seed)
        rows_text = np.array([t for t, _ in index])
        lang_rows, fit_mask = lang[rows_text], fit_ids[sid[rows_text]]
        fits, res["latent"] = {}, {}
        for b in blocks:
            X = states[b]
            fits[b] = fit_erasers(X[fit_mask], lang_rows[fit_mask], langs, seed, block=b)
            res["latent"][str(b)] = latent_probe(X, lang_rows, fit_mask, seed)
            print(f"  block {b:>2}: English probe bacc={res['latent'][str(b)]['bacc']}  "
                  f"mean P(en) of non-English positions={non_en_mean(res['latent'][str(b)]['p_en']):.3f}")
        save_fits(fits_path(name), fits)
        del states
        res["chosen_block"] = choose_block(res["latent"], blocks)
        write_json(outp, res)
        print(f"  chosen block: {res['chosen_block']} of {n}")
    fits = load_fits(fits_path(name))
    cb = res["chosen_block"]

    def encoder(block, eraser):
        if eraser == NONE:
            return make_encode(student, device, batch_size=bs_enc)
        return make_encode(student, device, batch_size=bs_enc,
                           hook=(block, rank1_torch_fn(*fits[block][eraser], device)))

    # ---- stage B: FLORES embedding shift under erasure + embeddings for the t-SNE figure
    if "shift" not in res:
        E0 = encoder(cb, NONE)(texts)
        embeds, res["shift"] = {"raw": E0.astype(np.float16), "lang": lang}, {}
        for er in ("en", "random"):
            E1 = encoder(cb, er)(texts)
            cos = (E0 * E1).sum(1)
            res["shift"][er] = {l: round(float(cos[lang == l].mean()), 4) for l in langs}
            if er == "en":
                embeds["erased_en"] = E1.astype(np.float16)
        np.savez(embeds_path(name), **embeds)
        write_json(outp, res)
        print("  shift (cos edited vs original) under English erasure: " +
              " ".join(f"{l}:{v:.3f}" for l, v in res["shift"]["en"].items()))
    if skip_battery:
        return

    # ---- stage C: Belebele matrix (unedited baseline first, then the erasures)
    from byte_embed.config import STUDY_LANGS
    from byte_embed.eval_mteb import eval_battery
    res.setdefault("belebele", {})
    for key, b, e in belebele_plan(blocks, cb, langs):
        if key in res["belebele"]:
            continue
        print(f"=== {name}: Belebele, {'UNEDITED baseline' if e == NONE else f'{e!r} erased at block {b}'} ===")
        cells = eval_battery(encoder(b, e), STUDY_LANGS)["belebele"]
        if not any(cells.values()):     # eval_belebele swallows dataset errors into None per language
            raise SystemExit(f"[english] every Belebele language returned None for {key} — the dataset "
                             "could not be loaded (offline node / HF error). Refusing to record a "
                             "matrix of nulls; fix the environment and rerun.")
        res["belebele"][key] = cells
        write_json(outp, res)
        if key == NONE:
            res["identity_check"] = identity_check(cells, (stored_results(name, results) or {}).get("belebele"))
            ic = res["identity_check"]
            print(f"  [loader check] unedited vs stored: max|dnDCG@10| = {ic['max_abs_diff']} over "
                  f"{ic['cells']} languages -> {ic['verdict']}")
            write_json(outp, res)

    # ---- stage D: the full 20k-pool battery, unedited + English/random erased at the chosen block
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    res.setdefault("battery", {})
    for e in (NONE, "en", "random"):
        if e in res["battery"]:
            continue
        print(f"=== {name}: full battery, {'UNEDITED' if e == NONE else f'{e!r} erased at block {cb}'} ===")
        enc = encoder(cb, e)
        qa = eval_qa_retrieval(enc, benchmarks=QA_BENCH, n_queries=250, distractors=20000,
                               cache_dir=ckpt_dir)
        if not any((bd or {}).get("per_lang") for bd in qa.values()):
            # eval_qa_retrieval catches every exception per benchmark; an all-None QA axis means the
            # hook or the pools failed, not that the intervention had no effect.
            raise SystemExit(f"[english] every QA benchmark returned None for variant {e!r} — refusing "
                             "to record an empty axis.")
        res["battery"][e] = {"belebele": res["belebele"][NONE if e == NONE else f"{cb}:{e}"],
                             "miracl": eval_miracl_langs(enc, MIRACL_LANGS, n_queries=250,
                                                         distractors=20000, cache_dir=ckpt_dir),
                             "qa_retrieval": qa}
        write_json(outp, res)
    print(f"  saved -> {outp}")


def identity_check(mine, stored, tol=0.005):
    """Unedited Belebele vs the stored training-time cells: the loader/tokenization reproduction the
    whole study's 0.005-nDCG claim rests on. `verdict` is explicit about 'nothing to compare'."""
    stored = stored or {}
    diffs = {l: abs(c["ndcg@10"] - stored[l]["ndcg@10"]) for l, c in mine.items()
             if c and stored.get(l)}
    worst = max(diffs.values()) if diffs else None
    return {"max_abs_diff": round(worst, 4) if worst is not None else None, "cells": len(diffs),
            "verdict": ("no stored cells to compare" if worst is None else
                        "PASS" if worst <= tol else "WARN: does not reproduce the stored results")}


# ----------------------------------------------------------------------------------------------
# summary (paired bootstrap over the Belebele queries)
# ----------------------------------------------------------------------------------------------
def _pq(r, key, lang):
    c = (r.get("belebele", {}).get(key) or {}).get(lang)
    return (c or {}).get("per_query")


def _boot(vecs, n_boot=10000, seed=0):
    """Mean and 95% CI of a per-language list of paired delta vectors, resampled WITHIN language
    (stratified) so unequal query counts do not reweight the languages."""
    vecs = [np.asarray(v, dtype=float) for v in vecs if len(v)]
    if not vecs:
        return None
    rng = np.random.default_rng(seed)
    per = np.array([v.mean() for v in vecs])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in vecs])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": round(float(per.mean()), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "n_lang": len(vecs),
            "significant": bool(lo > 0 or hi < 0)}


def matrix_summary(r, n_boot=2000, seed=0):
    """Belebele matrix at the chosen block, all measured against the UNEDITED pass and over the same
    population (languages other than English and other than the erased one):
      en_col     effect on those languages of erasing English        <- treatment
      other_cols effect on them of erasing some OTHER language       <- control (matched)
      random     effect on them of erasing a random direction        <- null
      own        effect on a language of erasing ITS OWN direction   <- upper reference
      en_self    effect on English of erasing English
      excess     paired (en_col - other_cols) per query              <- the headline
    Returns None when the unedited baseline or the English column is missing."""
    cb = r.get("chosen_block")
    if cb is None or NONE not in r.get("belebele", {}):
        return None
    langs = [l for l, c in r["belebele"][NONE].items() if c]
    others = [m for m in langs if m != "en" and f"{cb}:{m}" in r["belebele"]]
    en_v, oth_v, rnd_v, own_v, exc_v = [], [], [], [], []
    for l in langs:
        base = _pq(r, NONE, l)
        if not base:
            continue
        keys = sorted(base)

        def dv(key):
            got = _pq(r, key, l)
            return np.array([got[k] - base[k] for k in keys if k in got]) if got else None
        if l == "en":
            continue
        d_en, d_rnd = dv(f"{cb}:en"), dv(f"{cb}:random")
        cols = [dv(f"{cb}:{m}") for m in others if m != l]
        cols = [c for c in cols if c is not None and len(c)]
        if d_en is not None and len(d_en):
            en_v.append(d_en)
        if d_rnd is not None and len(d_rnd):
            rnd_v.append(d_rnd)
        if cols:
            d_oth = np.mean(np.stack(cols, 0), 0)
            oth_v.append(d_oth)
            if d_en is not None and len(d_en) == len(d_oth):
                exc_v.append(d_en - d_oth)                        # paired: same queries
        own = dv(f"{cb}:{l}")
        if own is not None and len(own):
            own_v.append(own)
    en_self = _pq(r, f"{cb}:en", "en"), _pq(r, NONE, "en")
    self_v = ([np.array([en_self[0][k] - en_self[1][k] for k in sorted(en_self[1]) if k in en_self[0]])]
              if en_self[0] and en_self[1] else [])
    return {"en_col": _boot(en_v, n_boot, seed), "other_cols": _boot(oth_v, n_boot, seed),
            "random": _boot(rnd_v, n_boot, seed), "own_direction": _boot(own_v, n_boot, seed),
            "en_self": _boot(self_v, n_boot, seed), "excess": _boot(exc_v, n_boot, seed),
            "n_control_cols": len(others)}


def en_by_depth(r, n_boot=2000, seed=0):
    """{block: bootstrap of the English column on non-English languages} — the depth curve, and the
    only cross-architecture comparison that is matched by depth fraction rather than by each model's
    own data-chosen block."""
    out = {}
    for b in r.get("blocks", []):
        vecs = []
        for l, c in (r["belebele"].get(NONE) or {}).items():
            if l == "en" or not c:
                continue
            base, got = _pq(r, NONE, l), _pq(r, f"{b}:en", l)
            if base and got:
                vecs.append(np.array([got[k] - base[k] for k in sorted(base) if k in got]))
        out[b] = _boot(vecs, n_boot, seed)
    return out


def _f(x, w=9, p=4):
    if isinstance(x, dict):
        x = x.get("mean")
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) and x == x else f"{'-':>{w}}"


def _ci(x):
    return f"[{x['ci_low']:+.3f},{x['ci_high']:+.3f}]{'*' if x['significant'] else ''}" if x else ""


def merge(results="results/retrieval_bgem3.json", n_boot=2000, seed=0):
    from byte_embed.stats import compare
    d = merge_parts(ANALYSIS, schema=SCHEMA)
    M = d["models"]
    print("\nEXP 3 — IS ENGLISH LOAD-BEARING? (rank-1 LEACE erasure of the English direction inside the "
          "encoder; every Δ = erased − the model's own UNEDITED pass, nDCG@10, paired bootstrap)")
    summary, depths = {}, {}
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r or "latent" not in r:
            continue
        blocks, cb = r["blocks"], r.get("chosen_block")
        print(f"\n  {n}  ({r['n_blocks']} blocks; intervention at block {cb} = "
              f"{(cb + 1) / r['n_blocks']:.0%} depth)")
        print("    latent English (mean P(en) of non-English / of English / probe bacc) by block: "
              + "  ".join(f"b{b}: {non_en_mean(r['latent'][str(b)]['p_en']):.3f}/"
                          f"{(r['latent'][str(b)]['p_en'].get('en') or float('nan')):.3f}/"
                          f"{r['latent'][str(b)]['bacc']}" for b in blocks))
        ic = r.get("identity_check")
        if ic:
            print(f"    loader check (unedited vs stored): max|dnDCG@10|={ic['max_abs_diff']} "
                  f"over {ic['cells']} languages -> {ic['verdict']}")
        if r.get("shift"):
            print("    cos(edited, original) under English erasure: "
                  + " ".join(f"{l}:{v:.3f}" for l, v in r["shift"]["en"].items())
                  + f"   | random: {np.mean(list(r['shift']['random'].values())):.3f}")
        s = matrix_summary(r, n_boot, seed)
        if not s:
            continue
        summary[n] = s
        depths[n] = en_by_depth(r, n_boot, seed)
        print(f"    Belebele matrix @ block {cb} (effects on languages other than English and other "
              f"than the erased one; {s['n_control_cols']} control columns)")
        for lab, key in (("English erased      ", "en_col"), ("other language erased", "other_cols"),
                         ("random direction    ", "random"), ("own direction (diag)", "own_direction"),
                         ("English on itself   ", "en_self")):
            print(f"      {lab} {_f(s[key])}  {_ci(s[key])}")
        print(f"      -> English EXCESS over the matched control {_f(s['excess'])}  {_ci(s['excess'])}")
        print("      English column on non-English by depth: "
              + "  ".join(f"b{b}: {_f(v, 7, 4).strip()}{'*' if v and v['significant'] else ''}"
                          for b, v in depths[n].items()))
        if r.get("battery") and NONE in r["battery"]:
            for e in ("en", "random"):
                if e not in r["battery"]:
                    continue
                rows = compare(r["battery"][e], r["battery"][NONE], n_boot)
                mono = [(c, x) for c, x in rows if tuple(c) not in CROSS_CELLS]
                cross = [(c, x) for c, x in rows if tuple(c) in CROSS_CELLS]
                s[f"battery_{e}_mono"] = float(np.mean([x["delta"] for _, x in mono])) if mono else None
                print(f"    20k battery, '{e}' erased: monolingual mean Δ {_f(s[f'battery_{e}_mono'])} "
                      f"({sum(x['significant'] for _, x in mono)}/{len(mono)} cells significant); "
                      f"cross-lingual mean Δ "
                      f"{_f(float(np.mean([x['delta'] for _, x in cross])) if cross else None)} "
                      f"({len(cross)} cells)")
                if e == "en":
                    print("      per cell: " + "  ".join(
                        f"{c[0][:6]}-{c[1]}:{x['delta']:+.3f}{'*' if x['significant'] else ''}"
                        for c, x in rows))
    print("\n  byte − subword per size (POSITIVE = erasing English hurts byte LESS than subword, "
          "i.e. byte depends on English less)")
    for size in ("small", "base", "large"):
        b, s = summary.get(f"byte-{size}"), summary.get(f"subword-{size}")
        if not (b and s):
            continue

        def dd(k):
            x, y = b.get(k), s.get(k)
            x = x.get("mean") if isinstance(x, dict) else x
            y = y.get("mean") if isinstance(y, dict) else y
            return f"{x - y:+.4f}" if x is not None and y is not None else "-"
        print(f"  {size:6} at each model's own block — English column: {dd('en_col')}   "
              f"English excess: {dd('excess')}   20k battery: {dd('battery_en_mono')}")
        db, ds = depths.get(f"byte-{size}") or {}, depths.get(f"subword-{size}") or {}
        matched = []
        for (bb, vb), (sb, vs) in zip(sorted(db.items()), sorted(ds.items())):
            if vb and vs:
                matched.append(f"{(bb + 1) / M[f'byte-{size}']['n_blocks']:.0%}: "
                               f"{vb['mean'] - vs['mean']:+.4f}")
        if matched:
            print(f"  {'':6} at MATCHED depth fractions — English column: " + "  ".join(matched))
    print("  reading: the English column below the matched control AND below random, more negative for "
          "subword -> English is load-bearing and byte depends on it less; only English itself hurt in "
          "both -> the story is wrong for both; equal -> inherited from the teacher.")


def _selftest():
    rng = np.random.default_rng(0)
    d, n = 32, 2000
    langs = ["en", "te", "sw"]
    lang_rows = np.array([langs[i % 3] for i in range(n)])
    sid = np.array([i // 3 for i in range(n)])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = rng.standard_normal((n, d)) * 0.5
    for j, l in enumerate(langs):
        X[lang_rows == l] += 2.0 * Q[:, j]
    fit_mask = (sid % 2 == 0)
    fits = fit_erasers(X[fit_mask], lang_rows[fit_mask], langs, block=2)
    assert set(fits) == {"en", "te", "sw", "random"}
    f2 = fit_erasers(X[fit_mask], lang_rows[fit_mask], langs, block=5)
    assert not np.allclose(fits["random"][0], f2["random"][0])     # independent null per depth
    assert np.allclose(fits["en"][0], f2["en"][0])                 # the fitted directions are not
    lat = latent_probe(X, lang_rows, fit_mask)
    assert lat["bacc"] > 0.95 and lat["p_en"]["en"] > 0.9 and non_en_mean(lat["p_en"]) < 0.1, lat
    from byte_embed.interp_common import apply_rank1
    a, b, mu = fits["en"]
    assert latent_probe(apply_rank1(X, a, b, mu), lang_rows, fit_mask)["bacc"] < 0.7
    latent = {"2": {"p_en": {"en": 0.9, "te": 0.2, "sw": 0.3}}, "5": {"p_en": {"en": 0.9, "te": 0.4, "sw": 0.5}},
              "8": {"p_en": {"en": 0.9, "te": 0.1, "sw": 0.1}}}
    assert choose_block(latent, [2, 5, 8]) == 5
    assert choose_block({str(b): {"p_en": {"en": 0.9}} for b in (2, 5, 8)}, [2, 5, 8]) == 8
    import tempfile
    p = tempfile.mktemp(suffix=".npz")
    save_fits(p, {2: fits})
    assert np.allclose(load_fits(p)[2]["en"][0], a) and set(load_fits(p)[2]) == set(fits)
    plan = belebele_plan([2, 5, 8, 11], 5, ["en", "te", "sw"])
    assert plan[0] == (NONE, 5, NONE) and len(plan) == 1 + 2 + 2 + 6, plan
    assert n_belebele_cells(10, 4) == 18 and len(belebele_plan([2, 5, 8, 11], 5, [f"l{i}" for i in range(9)] + ["en"])) == 18

    # matrix summary: English erasure hurts the others by -0.09, other columns by -0.01 on the same
    # queries, random by 0 -> excess must be -0.08 and the control must EXCLUDE the English row.
    qs = [f"q{i}" for i in range(120)]
    L = ["en", "te", "sw", "yo"]

    def cells(fn):
        return {l: {"ndcg@10": 0.8, "per_query": {q: fn(l, q) for q in qs}} for l in L}
    bel = {NONE: cells(lambda l, q: 0.80)}
    bel["5:en"] = cells(lambda l, q: 0.80 + (-0.20 if l == "en" else -0.09))
    bel["5:random"] = cells(lambda l, q: 0.80)
    for m in ("te", "sw", "yo"):
        bel[f"5:{m}"] = cells(lambda l, q, _m=m: 0.80 + (-0.30 if l == _m else -0.50 if l == "en" else -0.01))
    r = {"chosen_block": 5, "blocks": [5], "belebele": bel}
    s = matrix_summary(r, n_boot=200)
    assert abs(s["en_col"]["mean"] + 0.09) < 1e-9, s["en_col"]
    assert abs(s["other_cols"]["mean"] + 0.01) < 1e-9, s["other_cols"]       # the -0.50 en rows excluded
    assert abs(s["own_direction"]["mean"] + 0.30) < 1e-9 and abs(s["en_self"]["mean"] + 0.20) < 1e-9
    assert abs(s["excess"]["mean"] + 0.08) < 1e-9 and s["excess"]["significant"], s["excess"]
    assert abs(s["random"]["mean"]) < 1e-9 and not s["random"]["significant"]
    assert abs(en_by_depth(r, n_boot=200)[5]["mean"] + 0.09) < 1e-9
    ic = identity_check({"te": {"ndcg@10": 0.700}}, {"te": {"ndcg@10": 0.702}})
    assert ic["verdict"] == "PASS" and ic["cells"] == 1
    assert identity_check({"te": {"ndcg@10": 0.7}}, None)["verdict"].startswith("no stored")
    assert identity_check({"te": {"ndcg@10": 0.7}}, {"te": {"ndcg@10": 0.9}})["verdict"].startswith("WARN")
    print("selftest OK: per-depth erasers (independent random control), latent probe, block choice, "
          "plan, fits round-trip, matched-population matrix summary with paired CIs, identity check")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-battery", action="store_true", help="erasers, latent curve and shift only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples in --merge")
    ap.add_argument("--flores-split", default="devtest", choices=["devtest", "dev"],
                    help="FLORES split the erasers are fitted on (dev = text-disjoint from Belebele)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge(a.results, a.n_boot, a.seed)
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.seed, a.skip_battery, a.flores_split)


if __name__ == "__main__":
    main()
