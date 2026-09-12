"""Exp 3 — is English load-bearing for the other languages? A causal intervention INSIDE the network.

Claim under test: an English-centric model represents non-English inputs partly through
English-associated features in its middle layers, so removing the English direction inside the
network should hurt every other language — and more so in the subword model than in the byte model.

Procedure (per model; directions fitted on FLORES, effects measured on held-out eval pools):
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
  4. Controls: erase the random direction and every other language's direction in turn -> a 10x10
     matrix on Belebele of "effect on language L of erasing language M" (English/random also at the
     other three depths); the English column and the random control on the full 20k-pool battery.
  5. Embedding-space picture: FLORES final embeddings, unedited and with English erased, saved for
     the t-SNE / PCA figure (gen_interp_figures.py), plus the per-language cosine between the edited
     and original embedding of every sentence.
Baseline ("none") = the stored training-time results, which this loader reproduces within 0.005.

Pre-registered readings: the English column hurts the other languages beyond the random and
own-direction controls, more for subword than byte -> English is load-bearing and byte depends on it
less. It hurts only English itself in both -> the English-centric story is wrong for both
architectures. Hurts both equally -> inherited from the teacher, tokenization irrelevant.
Stated limit: one linear direction at one depth; a distributed dependence would survive it.

  python -m byte_embed.interp_english --only byte-small
  python -m byte_embed.interp_english --only subword-small --skip-battery   # erasers + latent + shift only
  python -m byte_embed.interp_english --merge
  python -m byte_embed.interp_english --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from byte_embed.interp_common import (MAIN_MODELS, block_states, depth_blocks, flores_flat,
                                      flores_parallel, leace_direction, load_student, make_encode,
                                      merge_parts, models_in, n_blocks, part_path, rank1_eraser,
                                      rank1_torch_fn, read_json, stored_results, utf8_stdout,
                                      whitener, write_json)

ANALYSIS = "english"
PER_TEXT = 10                      # sampled positions per FLORES sentence per depth
MIRACL_LANGS = ["en", "zh", "ar", "te", "bn", "sw", "yo"]
QA_BENCH = ("amharicpr", "ciral", "afriqa")
CROSS_CELLS = {("ciral", "ha"), ("afriqa", "rw"), ("afriqa", "ha"), ("afriqa", "sw"), ("afriqa", "yo")}


def fits_path(name):
    return part_path(ANALYSIS, name).with_suffix(".erasers.npz")


def embeds_path(name):
    return part_path(ANALYSIS, name).with_suffix(".embeds.npz")


# ----------------------------------------------------------------------------------------------
# erasers + latent-English probe (numpy / sklearn)
# ----------------------------------------------------------------------------------------------
def fit_erasers(X, lang_rows, langs, seed=0):
    """One whitener per depth, then a rank-1 LEACE eraser per language ("L vs rest") and one random
    direction in the same whitened metric -> {name: (a, b, mu)}."""
    W, Wp, mu = whitener(X)
    out = {l: rank1_eraser(W, Wp, mu, leace_direction(W, X, lang_rows == l)) for l in langs}
    rng = np.random.default_rng(seed)
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
    """Pre-registered: the depth where non-English positions are most English-like (ties -> the
    deeper of the tied blocks is NOT preferred; the first max wins)."""
    scores = [non_en_mean(latent[str(b)]["p_en"]) for b in blocks]
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


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, seed=0, skip_battery=False):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("battery") and all(e in res["battery"] for e in ("en", "random")):
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
    par = flores_parallel(cache_dir=ckpt_dir)
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
            fits[b] = fit_erasers(X[fit_mask], lang_rows[fit_mask], langs, seed)
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

    # ---- stage B: FLORES embedding shift under erasure + embeddings for the t-SNE figure
    if "shift" not in res:
        E0 = make_encode(student, device, batch_size=bs_enc)(texts)
        embeds, res["shift"] = {"raw": E0.astype(np.float16), "lang": lang}, {}
        for er in ("en", "random"):
            E1 = make_encode(student, device, batch_size=bs_enc,
                             hook=(cb, rank1_torch_fn(*fits[cb][er], device)))(texts)
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

    # ---- stage C: Belebele matrix at the chosen block (+ English/random at the other depths)
    from byte_embed.config import STUDY_LANGS
    from byte_embed.eval_mteb import eval_battery
    res.setdefault("belebele", {})
    plan = [(cb, e) for e in ["en", "random"] + [l for l in langs if l != "en"]]
    plan += [(b, e) for b in blocks if b != cb for e in ("en", "random")]
    for b, e in plan:
        key = f"{b}:{e}"
        if key in res["belebele"]:
            continue
        print(f"=== {name}: Belebele with '{e}' erased at block {b} ===")
        enc = make_encode(student, device, batch_size=bs_enc, hook=(b, rank1_torch_fn(*fits[b][e], device)))
        res["belebele"][key] = eval_battery(enc, STUDY_LANGS)["belebele"]
        write_json(outp, res)

    # ---- stage D: the full 20k-pool battery, English and random erased at the chosen block
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    res.setdefault("battery", {})
    for e in ("en", "random"):
        if e in res["battery"]:
            continue
        print(f"=== {name}: full battery with '{e}' erased at block {cb} ===")
        enc = make_encode(student, device, batch_size=bs_enc, hook=(cb, rank1_torch_fn(*fits[cb][e], device)))
        res["battery"][e] = {
            "belebele": res["belebele"][f"{cb}:{e}"],
            "miracl": eval_miracl_langs(enc, MIRACL_LANGS, n_queries=250, distractors=20000, cache_dir=ckpt_dir),
            "qa_retrieval": eval_qa_retrieval(enc, benchmarks=QA_BENCH, n_queries=250,
                                              distractors=20000, cache_dir=ckpt_dir)}
        write_json(outp, res)
    print(f"  saved -> {outp}")


# ----------------------------------------------------------------------------------------------
def belebele_deltas(r, base, key):
    """{lang: nDCG@10(erased) - nDCG@10(none)} for one 'block:eraser' key of the Belebele matrix."""
    none = {l: c["ndcg@10"] for l, c in (base.get("belebele") or {}).items() if c}
    got = r.get("belebele", {}).get(key) or {}
    return {l: round(c["ndcg@10"] - none[l], 4) for l, c in got.items() if c and l in none}


def matrix_summary(r, base):
    """Belebele-matrix summary at the chosen block: English column on non-English languages, English
    on itself, every other language's column on the languages other than itself, the diagonal, the
    random control, and the English excess over the other columns."""
    cb = r["chosen_block"]
    langs = [l for l in (base.get("belebele") or {})]
    en = belebele_deltas(r, base, f"{cb}:en")
    rnd = belebele_deltas(r, base, f"{cb}:random")
    others_off, diag = [], []
    for m in langs:
        if m == "en":
            continue
        dm = belebele_deltas(r, base, f"{cb}:{m}")
        if not dm:
            continue
        others_off.extend(v for l, v in dm.items() if l != m)
        if m in dm:
            diag.append(dm[m])
    en_non = [v for l, v in en.items() if l != "en"]
    s = {"en_col_non_en": float(np.mean(en_non)) if en_non else None,
         "en_self": en.get("en"),
         "other_cols_off_diag": float(np.mean(others_off)) if others_off else None,
         "diag_non_en": float(np.mean(diag)) if diag else None,
         "random": float(np.mean(list(rnd.values()))) if rnd else None,
         "per_lang_en_col": en}
    s["en_excess"] = (s["en_col_non_en"] - s["other_cols_off_diag"]
                      if s["en_col_non_en"] is not None and s["other_cols_off_diag"] is not None else None)
    return s


def _f(x, w=8, p=4):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) and x == x else f"{'—':>{w}}"


def merge(results="results/retrieval_bgem3.json", n_boot=10000):
    from byte_embed.stats import compare
    d = merge_parts(ANALYSIS)
    M = d["models"]
    print("\nEXP 3 — IS ENGLISH LOAD-BEARING? (rank-1 LEACE erasure of the English direction inside the "
          "encoder; Δ = erased − none, nDCG@10)")
    summary = {}
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r or "latent" not in r:
            continue
        base = stored_results(n, results) or {}
        blocks, cb = r["blocks"], r.get("chosen_block")
        print(f"\n  {n}  ({r['n_blocks']} blocks; intervention at block {cb})")
        print("    latent English (mean P(en) of non-English positions / English positions / probe bacc) by block: "
              + "  ".join(f"b{b}: {non_en_mean(r['latent'][str(b)]['p_en']):.3f}/"
                          f"{(r['latent'][str(b)]['p_en'].get('en') or float('nan')):.3f}/{r['latent'][str(b)]['bacc']}"
                          for b in blocks))
        if r.get("shift"):
            print("    cos(edited, original) under English erasure: "
                  + " ".join(f"{l}:{v:.3f}" for l, v in r["shift"]["en"].items())
                  + f"   | random: {np.mean(list(r['shift']['random'].values())):.3f}")
        if not (base.get("belebele") and r.get("belebele")):
            continue
        s = matrix_summary(r, base)
        summary[n] = s
        print(f"    Belebele matrix @ block {cb}:  English column on non-English {_f(s['en_col_non_en'])}"
              f"   English on itself {_f(s['en_self'])}   other columns off-diagonal {_f(s['other_cols_off_diag'])}"
              f"   own direction (diag) {_f(s['diag_non_en'])}   random {_f(s['random'])}"
              f"   -> English excess {_f(s['en_excess'])}")
        print("      English column per language: " + " ".join(f"{l}:{v:+.3f}" for l, v in s["per_lang_en_col"].items()))
        depth = []
        for b in blocks:
            en_d = belebele_deltas(r, base, f"{b}:en")
            vals = [v for l, v in en_d.items() if l != "en"]
            depth.append(f"b{b}: {np.mean(vals):+.4f}" if vals else f"b{b}: —")
        print("      English column on non-English by depth: " + "  ".join(depth))
        if r.get("battery"):
            for e in ("en", "random"):
                if e not in r["battery"]:
                    continue
                rows = compare(r["battery"][e], base, n_boot)
                mono = [(c, x) for c, x in rows if tuple(c) not in CROSS_CELLS]
                cross = [(c, x) for c, x in rows if tuple(c) in CROSS_CELLS]
                mono_d = [x["delta"] for _, x in mono]
                cross_d = [x["delta"] for _, x in cross]
                s[f"battery_{e}_mono"] = float(np.mean(mono_d)) if mono_d else None
                print(f"    battery, '{e}' erased: monolingual mean Δ {_f(s[f'battery_{e}_mono'])} "
                      f"({sum(x['significant'] for _, x in mono)}/{len(mono)} cells significant, "
                      f"{sum(x['delta'] < 0 for _, x in mono)} negative); cross-lingual mean Δ "
                      f"{_f(float(np.mean(cross_d)) if cross_d else None)} ({len(cross)} cells)")
                if e == "en":
                    print("      per cell: " + "  ".join(f"{c[0][:6]}-{c[1]}:{x['delta']:+.3f}{'*' if x['significant'] else ''}"
                                                       for c, x in rows))
    print("\n  byte − subword per size (negative = erasing English hurts byte LESS than subword)")
    for size in ("small", "base", "large"):
        b, s = summary.get(f"byte-{size}"), summary.get(f"subword-{size}")
        if not (b and s):
            continue
        def dd(k):
            return f"{b[k] - s[k]:+.4f}" if b.get(k) is not None and s.get(k) is not None else "—"
        print(f"  {size:6} English column on non-English: {dd('en_col_non_en')}   English excess: {dd('en_excess')}"
              f"   battery monolingual (English erased): {dd('battery_en_mono')}")
    print("  reading: English column << 0 beyond random/own-direction and more negative for subword -> English is "
          "load-bearing, byte depends on it less; only English itself hurt in both -> story wrong for both; "
          "equal -> inherited from the teacher.")


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
    fits = fit_erasers(X[fit_mask], lang_rows[fit_mask], langs)
    assert set(fits) == {"en", "te", "sw", "random"}
    lat = latent_probe(X, lang_rows, fit_mask)
    assert lat["bacc"] > 0.95 and lat["p_en"]["en"] > 0.9 and non_en_mean(lat["p_en"]) < 0.1, lat
    from byte_embed.interp_common import apply_rank1
    a, b, mu = fits["en"]
    lat2 = latent_probe(apply_rank1(X, a, b, mu), lang_rows, fit_mask)
    assert lat2["bacc"] < 0.7, lat2                                   # English erased
    latent = {"2": {"p_en": {"en": 0.9, "te": 0.2, "sw": 0.3}}, "5": {"p_en": {"en": 0.9, "te": 0.4, "sw": 0.5}},
              "8": {"p_en": {"en": 0.9, "te": 0.1, "sw": 0.1}}}
    assert choose_block(latent, [2, 5, 8]) == 5
    import tempfile
    p = tempfile.mktemp(suffix=".npz")
    save_fits(p, {2: fits})
    f2 = load_fits(p)
    assert np.allclose(f2[2]["en"][0], a) and set(f2[2]) == set(fits)
    r = {"chosen_block": 2, "belebele": {"2:en": {"en": {"ndcg@10": 0.80}, "te": {"ndcg@10": 0.60}, "sw": {"ndcg@10": 0.62}},
                                        "2:random": {"en": {"ndcg@10": 0.89}, "te": {"ndcg@10": 0.70}, "sw": {"ndcg@10": 0.71}},
                                        "2:te": {"en": {"ndcg@10": 0.89}, "te": {"ndcg@10": 0.60}, "sw": {"ndcg@10": 0.70}},
                                        "2:sw": {"en": {"ndcg@10": 0.89}, "te": {"ndcg@10": 0.70}, "sw": {"ndcg@10": 0.60}}}}
    base = {"belebele": {"en": {"ndcg@10": 0.90}, "te": {"ndcg@10": 0.70}, "sw": {"ndcg@10": 0.70}}}
    s = matrix_summary(r, base)
    assert abs(s["en_col_non_en"] + 0.09) < 1e-6 and abs(s["en_self"] + 0.10) < 1e-6
    assert abs(s["other_cols_off_diag"] + 0.005) < 1e-6 and abs(s["diag_non_en"] + 0.10) < 1e-6
    assert abs(s["random"]) < 1e-6 and abs(s["en_excess"] + 0.085) < 1e-6, s   # random: -0.01, 0, +0.01
    print("selftest OK: per-depth erasers + latent-English probe, English erasure removes the concept, "
          "block choice, fits round-trip, matrix summary")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-battery", action="store_true", help="erasers, latent curve and shift only")
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
