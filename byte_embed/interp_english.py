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
is reported as the out-of-sample confirmation of the same effect. `--flores-split` moves the fitting
set: `dev`, or `dev+devtest` to fit on dev and probe on devtest (~2x the positions on each side, and a
fit/probe boundary that is a real split boundary rather than a random half of one pool). Do NOT assume
either public split is disjoint from Belebele -- the Belebele paper says only that its 488 passages
exclude the HIDDEN FLORES test set, never which public split they came from. `--flores-overlap`
measures it.

  python -m byte_embed.interp_english --only byte-small
  python -m byte_embed.interp_english --only subword-small --skip-battery   # erasers + latent + shift
  python -m byte_embed.interp_english --merge
  python -m byte_embed.interp_english --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from byte_embed.interp_common import (CROSS_CELLS, MAIN_MODELS, apply_rank1, block_states, depth_blocks,
                                      flores_splits,
                                      flores_flat, flores_parallel, leace_direction, load_student,
                                      make_encode, merge_parts, models_in, n_blocks, part_path,
                                      rank1_eraser, rank1_torch_fn, read_json, stored_results,
                                      utf8_stdout, whitener, write_json)

ANALYSIS = "english"
SCHEMA = 1                         # bumped when the part-file contents change meaning
PER_TEXT = 10                      # sampled positions per FLORES sentence per depth
MIRACL_LANGS = ["en", "zh", "ar", "te", "bn", "sw", "yo"]
QA_BENCH = ("amharicpr", "ciral", "afriqa")
NONE = "none"
ALL_EN = "all:en"                  # English erased at EVERY encoder block simultaneously
ALL_RND = "all:random"             # the same count of random rank-1 edits at the same blocks
FIT_TAG = "sequential-per-arm"     # all_depth_fit marker; bumped when the fitting contract changes                      # the unedited pass: baseline + loader identity check


def _split_suffix(kind, split):
    """Eraser sidecars are keyed by FLORES split, because the split IS the fitting set: a
    `--flores-split dev` run must not reuse directions fitted on devtest. devtest keeps the original
    filename so finished runs stay cached."""
    return f".{kind}.npz" if split == "devtest" else f".{kind}-{split}.npz"


def flores_overlap(cache_dir="checkpoints", lang="en"):
    """How much of Belebele's passage text comes from each FLORES split.

    Belebele reconstructs paragraphs from consecutive FLORES sentences, and its paper says only that
    the 488 passages exclude the HIDDEN test set -- it never says which PUBLIC split they come from.
    That matters because the erasers are fitted on FLORES and the Belebele cells are scored on
    Belebele: whichever split the passages came from is not held-out text for this experiment. So
    measure it rather than assume it."""
    from byte_embed.interp_common import flores_parallel
    from byte_embed.interp_script import belebele_texts
    _, passages = belebele_texts(lang)
    joined = "\n".join(passages)
    print(f"\n  BELEBELE PASSAGE PROVENANCE ({lang}, {len(passages)} distinct passages)")
    print(f"  {'split':10}{'sentences':>11}{'found in a passage':>20}{'share':>8}")
    seen = {}
    for sp in ("dev", "devtest"):
        sents = [s for s in flores_parallel([lang], cache_dir, sp)[lang] if len(s) > 25]
        hit = [s for s in sents if s in joined]
        seen[sp] = set(hit)
        print(f"  {sp:10}{len(sents):>11}{len(hit):>20}{len(hit) / max(len(sents), 1):>8.1%}")
    both = seen["dev"] & seen["devtest"]
    print(f"  sentences matched from BOTH splits: {len(both)} (duplicate rows across splits, if any)")
    print("  reading: a split with a high share is NOT held out from the Belebele cells -- fit the "
          "erasers on the other one, or read the FLORES-independent 20k battery as primary.")
    return seen


def fits_path(name, split="devtest"):
    return part_path(ANALYSIS, name).with_suffix(_split_suffix("erasers", split))


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


PRUNED = "pruned_single_block"     # set once the one-block arms are dropped, so they stay dropped


def single_block_arms(res):
    """Battery arms to score: just the unedited baseline once the one-block arms are pruned."""
    return (NONE,) if res.get(PRUNED) else (NONE, "en", "random")


def english_done(res):
    """Whether an exp 3 part file is complete. run_one's early exit and status.sh both read this,
    so the two cannot drift -- status.sh used to call a file done as soon as the battery held the
    single-block arms, which reported models with no all-depth arm at all as finished.

    A pruned file is not required to carry `erasure_check`: it was dropped deliberately as the
    degenerate single-block arm, and demanding it back would have stage B2 recompute the very thing
    the prune removed."""
    bat = res.get("battery") or {}
    # Key PRESENCE throughout: a stage that recorded an empty dict is falsy, and `.get()` would read
    # it as never having run -- the same trap that broke the FLORES split guard.
    return bool(bat
                and all(e in bat for e in single_block_arms(res))
                and all(k in bat for k in (ALL_EN, ALL_RND))
                # An all-depth arm built from STACKED erasers describes an intervention that was
                # never applied, so it is not a finished result. Without this the early exit in
                # run_one fires before purge_stale_all_depth is ever reached and the job returns in
                # three seconds having done nothing -- which is exactly what happened.
                and res.get("all_depth_fit") == FIT_TAG
                and "reinstatement" in res
                and (res.get(PRUNED) or "erasure_check" in res))


def prune_single_block(models=None):
    """Remove every result produced by erasing at ONE block, and mark the part files.

    Kept: the unedited baseline, `latent`, `shift`, `identity_check`, `chosen_block`, `reinstatement`
    and the all-depth arms. `reinstatement` survives on purpose -- it is a single-block edit, but its
    job is to demonstrate that a single-block edit gets undone, which is the evidence FOR the
    all-depth arm rather than a claim resting on it."""
    import glob
    out = []
    for p in sorted(glob.glob(f"results/interp_{ANALYSIS}_part_*.json")):
        res = read_json(Path(p))
        if not res:
            continue
        bel = res.get("belebele") or {}
        drop = [k for k in bel if ":" in k and not k.startswith("all:")]
        bat = [k for k in ("en", "random") if k in (res.get("battery") or {})]
        for k in drop:
            del bel[k]
        for k in bat:
            del res["battery"][k]
        had_ec = res.pop("erasure_check", None) is not None
        res[PRUNED] = True
        write_json(Path(p), res)
        out.append((res.get("model", p), len(drop), len(bat), had_ec))
        print(f"  {res.get('model', p):15} dropped {len(drop)} Belebele column(s), {len(bat)} battery "
              f"arm(s){', erasure_check' if had_ec else ''}; kept "
              f"{sorted(bel)} + battery {sorted(res.get('battery') or {})}")
    if not out:
        print("  no part files found under results/ -- nothing to prune")
    return out


def fit_all_blocks(student, texts, lang, sid, fit_ids, n, device, big, seed, path):
    """A rank-1 English eraser and a site-matched random control at EVERY block, fitted
    SEQUENTIALLY with the upstream erasers already installed.

    Fitting every block on unedited activations and then stacking the maps does not work, and the
    failure is measurable rather than theoretical: on byte-small the block-2 eraser alone takes the
    English probe to 0.500, but with erasers also live at blocks 0 and 1 the probe at block 2 reads
    0.960. A LEACE map equalises the class means of the distribution it was fitted on; an upstream
    edit changes what arrives, so `mu` and the direction are mis-specified and the map stops erasing.

    Fitting block b while blocks 0..b-1 are already active makes each map valid for the activations
    that actually reach it. The probe is then at chance right after every block's own edit by
    construction: block b+1 rebuilds the concept from whatever survived, and eraser b+1 -- fitted on
    that rebuilt distribution -- removes it again.

    The random control gets its OWN chain rather than riding the English one. It is installed at eval
    time with random edits upstream, so fitting it under English-edited activations would leave its
    whitener and mu describing a distribution that never arrives -- the same mis-specification, in the
    one number that makes this arm interpretable. A mis-centred rank-1 edit adds a constant offset on
    top of removing its direction, so it plausibly damages MORE than a correctly centred one, which
    would inflate the control and drag the English excess toward zero.

    2n passes over the fit-half sentences, which is the only text fit_erasers ever saw. Each pass
    records ONE block, so memory stays flat."""
    if path.exists():
        return load_fits(path)
    keep = [i for i, s in enumerate(sid) if fit_ids[s]]
    ftexts, flang = [texts[i] for i in keep], lang[keep]
    fits = {b: {} for b in range(n)}
    for arm in ("en", "random"):
        hooks = []
        for b in range(n):
            st, idx = block_states(student, ftexts, [b], per_text=PER_TEXT, device=device,
                                   batch_size=(4 if big else 8), seed=seed, hook=hooks or None)
            rows = np.array([t for t, _ in idx])
            e = fit_erasers(st[b], flang[rows], ["en"], seed, block=b)[arm]
            del st
            fits[b][arm] = e
            hooks.append((b, rank1_torch_fn(*e, device)))
        print(f"  [all-depth] {arm!r} chain fitted through all {n} blocks")
    save_fits(path, fits)
    return fits


def purge_stale_all_depth(res, outp):
    """Drop all-depth results produced by the non-composing stacked fit.

    Those numbers are not merely imprecise: the intervention they describe was not applied, since the
    erasers below the first block were fitted for distributions that never arrived. Anything without
    `all_depth_fit == "sequential"` predates the fix and has to be recomputed."""
    if res.get("all_depth_fit") == FIT_TAG or "all_depth_probe" not in res:
        return False
    res.pop("all_depth_probe", None)
    for k in (ALL_EN, ALL_RND):
        (res.get("belebele") or {}).pop(k, None)
        (res.get("battery") or {}).pop(k, None)
    print("  [all-depth] discarding results from the stacked (non-composing) fit -> recomputing")
    write_json(outp, res)
    return True


def fits_all_path(name, split="devtest"):
    return part_path(ANALYSIS, name).with_suffix(_split_suffix("erasers-allseq2", split))


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
    """Cells the chosen-block PLAN writes; belebele_plan is asserted against this, so they cannot
    drift apart."""
    return 1 + 2 + (n_langs - 1) + 2 * (n_depths - 1)


def n_english_cells(n_langs=10, n_depths=4):
    """Every Belebele cell a finished run holds: the chosen-block plan plus the two all-depth arms,
    which stage E writes rather than the plan. status.sh reads this one."""
    return n_belebele_cells(n_langs, n_depths) + 2


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, seed=0, skip_battery=False, flores_split="devtest"):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:            # part files from an older contract are discarded
        if res:
            print(f"  [english] {name}: part file schema {res.get('schema')} != {SCHEMA} -> recomputing")
        res = {"schema": SCHEMA}
    if english_done(res):
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
    # Every eraser in this part file was fitted on ONE split; combining stages fitted on different
    # text would make the depth columns incomparable, so say so rather than proceed.
    # A part file written before this key existed was fitted on devtest, the historical default, so
    # absence is NOT "no opinion": treating it as such would let stage A refit on the new pool while
    # stages B-E stayed cached from the old one, and the depth columns would silently be a mixture.
    # `"latent" in res`, not res.get("latent"): an empty dict is falsy, which would read a legacy
    # part file as having no opinion and re-open exactly the mixing this guard exists to prevent.
    prior = res.get("flores_split") or ("devtest" if "latent" in res else None)
    if prior not in (None, flores_split):
        raise SystemExit(
            f"[english] {outp} holds stages fitted on FLORES '{prior}' but '{flores_split}' was "
            f"requested, and mixing them would make the depth columns incomparable. Move or delete "
            f"that part file and its .erasers*.npz sidecars to refit from scratch on the new split.")
    res["flores_split"] = flores_split
    par = flores_parallel(cache_dir=ckpt_dir, split=flores_split)
    langs = list(par)
    texts, lang, sid = flores_flat(par)
    nsent = len(par[langs[0]])
    # Which sentences fit the erasers, and which are held out for the probe. With two splits
    # concatenated the boundary IS the split boundary -- fit on the first component, probe on the
    # second -- which is a stronger claim than a random half of one pool: the direction has to
    # transport to separately collected text, not merely to unseen sentences from the same batch.
    fit_ids = np.zeros(nsent, dtype=bool)
    parts = flores_splits(flores_split)
    if len(parts) > 1:
        n_first = len(flores_parallel([langs[0]], ckpt_dir, parts[0])[langs[0]])
        fit_ids[:n_first] = True
        print(f"  fitting on {parts[0]} ({n_first} sentences/lang), probing on "
              f"{'+'.join(parts[1:])} ({nsent - n_first})")
    else:
        rng = np.random.default_rng(seed)
        fit_ids[rng.choice(nsent, size=nsent // 2, replace=False)] = True
    big = "large" in name
    bs_enc = 32 if bm.get("kind") == "byte" else 128

    # ---- stage A: states at four depths -> erasers + latent-English probe -> chosen block
    if "latent" not in res or not fits_path(name, flores_split).exists():
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
        save_fits(fits_path(name, flores_split), fits)
        del states
        res["chosen_block"] = choose_block(res["latent"], blocks)
        write_json(outp, res)
        print(f"  chosen block: {res['chosen_block']} of {n}")
    fits = load_fits(fits_path(name, flores_split))
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
    # ---- stage B2: DOES THE ERASURE ACTUALLY ERASE? Re-read the states with the hook active and
    # re-run the English-vs-rest probe at the block it is applied to and at the last block. Without
    # this, "erasing English changed nothing" is indistinguishable from "the erasure did nothing":
    # the LEACE map is correct by construction (it equalises the class means, verified numerically in
    # interp_common's selftest) and the output cosines show English moving while other languages do
    # not, but neither shows the concept is UNRECOVERABLE where it matters, nor whether later blocks
    # rebuild it from other features.
    if "erasure_check" not in res and not res.get(PRUNED):
        last = n - 1
        want = sorted({cb, last})
        chk = {}
        for tag, hk in (("unedited", None), ("erased", (cb, rank1_torch_fn(*fits[cb]["en"], device)))):
            st, idx = block_states(student, texts, want, per_text=PER_TEXT, device=device,
                                   batch_size=(4 if big else 8), seed=seed, hook=hk)
            rows = np.array([t for t, _ in idx])
            for b in want:
                chk.setdefault(str(b), {})[tag] = latent_probe(st[b], lang[rows], fit_ids[sid[rows]], seed)
            del st
        res["erasure_check"] = {
            "block": cb, "last_block": last,
            "at_intervention": {t: chk[str(cb)][t]["bacc"] for t in ("unedited", "erased")},
            "at_last_block": {t: chk[str(last)][t]["bacc"] for t in ("unedited", "erased")},
            "p_en_non_english": {t: round(non_en_mean(chk[str(cb)][t]["p_en"]), 4)
                                 for t in ("unedited", "erased")}}
        ec = res["erasure_check"]
        print(f"  [erasure check] English probe bacc at block {cb}: "
              f"{ec['at_intervention']['unedited']} -> {ec['at_intervention']['erased']} "
              f"(0.5 = erased); at the last block {last}: "
              f"{ec['at_last_block']['unedited']} -> {ec['at_last_block']['erased']} "
              f"(near 0.5 = it does not come back)")
        write_json(outp, res)
    # ---- stage B3: IS IT REBUILT? The pre-registered `choose_block` rule turns out to select the
    # LAST block in every model (non-English positions look steadily more English-like with depth),
    # so stage B2 compares the intervention block against itself and can say nothing about whether
    # later blocks restore the concept. Erasing at the SHALLOWEST depth instead leaves the whole
    # upper encoder free to rebuild, which is the only setting where that question has an answer.
    # The unedited side is read from stage A's `latent`, which probed the same positions under the
    # same seed with no hook, so only the hooked pass is new.
    if "reinstatement" not in res and len(blocks) > 1:
        early = blocks[0]
        st, idx = block_states(student, texts, blocks, per_text=PER_TEXT, device=device,
                               batch_size=(4 if big else 8), seed=seed,
                               hook=(early, rank1_torch_fn(*fits[early]["en"], device)))
        rows = np.array([t for t, _ in idx])
        res["reinstatement"] = {"block": early, "at": {
            str(b): {"unedited": (res["latent"].get(str(b)) or {}).get("bacc"),
                     "erased": latent_probe(st[b], lang[rows], fit_ids[sid[rows]], seed)["bacc"]}
            for b in blocks}}
        del st
        at = res["reinstatement"]["at"]
        print(f"  [reinstatement] English erased at block {early}; probe bacc by depth "
              + "  ".join(f"b{b}: {at[str(b)]['unedited']}->{at[str(b)]['erased']}" for b in blocks)
              + "   (stays ~0.5 = gone for good; climbs back = later blocks rebuild it)")
        write_json(outp, res)
    # ---- stage B4: the ALL-DEPTH erasure, and proof that it leaves nothing to rebuild from.
    # `choose_block` selects the last block in every model, so every single-site arm above is an edit
    # with no encoder left after it. This one holds at all n blocks at once. The probe is then read at
    # the four depths: if it sits at chance everywhere, English is unavailable throughout the encoder
    # and the retrieval numbers that follow are a real test of whether the model needs it.
    purge_stale_all_depth(res, outp)
    if "all_depth_probe" not in res:
        fa = fit_all_blocks(student, texts, lang, sid, fit_ids, n, device, big, seed,
                            fits_all_path(name, flores_split))
        st, idx = block_states(student, texts, blocks, per_text=PER_TEXT, device=device,
                               batch_size=(4 if big else 8), seed=seed,
                               hook=[(b, rank1_torch_fn(*fa[b]["en"], device)) for b in range(n)])
        rows = np.array([t for t, _ in idx])
        res["all_depth_probe"] = {
            str(b): {"unedited": (res["latent"].get(str(b)) or {}).get("bacc"),
                     "erased": latent_probe(st[b], lang[rows], fit_ids[sid[rows]], seed)["bacc"]}
            for b in blocks}
        del st
        res["all_depth_fit"] = FIT_TAG
        ad = res["all_depth_probe"]
        print(f"  [all-depth] English erased at ALL {n} blocks; probe bacc by depth "
              + "  ".join(f"b{b}: {ad[str(b)]['unedited']}->{ad[str(b)]['erased']}" for b in blocks))
        write_json(outp, res)
    if skip_battery:
        return

    # ---- stage C: Belebele matrix (unedited baseline first, then the erasures)
    from byte_embed.config import STUDY_LANGS
    from byte_embed.eval_mteb import eval_battery
    res.setdefault("belebele", {})
    plan = [(NONE, cb, NONE)] if res.get(PRUNED) else belebele_plan(blocks, cb, langs)
    for key, b, e in plan:
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
    for e in single_block_arms(res):
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

    # ---- stage E: the all-depth arm scored on Belebele and the full battery, against a control that
    # is matched site for site. n simultaneous edits do n times the damage, so a drop here means
    # nothing unless the same number of random rank-1 edits at the same blocks is subtracted off.
    fa = load_fits(fits_all_path(name, flores_split))

    def all_encoder(eraser):
        return make_encode(student, device, batch_size=bs_enc,
                           hook=[(b, rank1_torch_fn(*fa[b][eraser], device)) for b in range(n)])
    for key, er in ((ALL_EN, "en"), (ALL_RND, "random")):
        if key not in res["belebele"]:
            print(f"=== {name}: Belebele, {er!r} erased at ALL {n} blocks ===")
            cells = eval_battery(all_encoder(er), STUDY_LANGS)["belebele"]
            if not any(cells.values()):
                raise SystemExit(f"[english] every Belebele language returned None for {key}.")
            res["belebele"][key] = cells
            write_json(outp, res)
        if key in res["battery"]:
            continue
        print(f"=== {name}: full battery, {er!r} erased at ALL {n} blocks ===")
        enc = all_encoder(er)
        qa = eval_qa_retrieval(enc, benchmarks=QA_BENCH, n_queries=250, distractors=20000,
                               cache_dir=ckpt_dir)
        if not any((bd or {}).get("per_lang") for bd in qa.values()):
            raise SystemExit(f"[english] every QA benchmark returned None for {key}.")
        res["battery"][key] = {"belebele": res["belebele"][key],
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
    if not any(k.startswith(f"{cb}:") for k in r["belebele"]):
        return None            # single-block columns pruned: absent, not "present and empty"
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


def flores_hubness(name, k=10):
    """English-centricity of the RETRIEVAL representation, from the FLORES embeddings stage B saved
    for the t-SNE figure. For every ordered language pair, query = sentence i of A against all of B,
    gold = i, scored P@1. Then: mean P@1 over pairs that INVOLVE English vs pairs between two
    non-English languages, and the rank of English among each non-English language's 9 partners
    (1 = English is its nearest language; 5 = no hub). Architecture-neutral, so unlike the
    segmentation probes this covers the subword models too. Returns None when the sidecar is absent."""
    p = embeds_path(name)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    if "raw" not in z.files:
        return None
    E, lang = z["raw"].astype(np.float32), z["lang"].astype(str)
    langs = sorted(set(lang.tolist()), key=lambda l: (l != "en", l))
    by = {l: E[lang == l] for l in langs}
    n = min(len(v) for v in by.values())
    by = {l: v[:n] / (np.linalg.norm(v[:n], axis=1, keepdims=True) + 1e-9) for l, v in by.items()}
    p1 = {}
    for a in langs:
        p1[a] = {}
        for b in langs:
            if a == b:
                continue
            S = by[a] @ by[b].T
            p1[a][b] = float((S.argmax(1) == np.arange(n)).mean())
    pairs = [(a, b) for a in langs for b in langs if a != b]
    en_pairs = [(a, b) for a, b in pairs if "en" in (a, b)]
    non_pairs = [(a, b) for a, b in pairs if "en" not in (a, b)]
    ranks, nearest = [], 0
    for l in langs:
        if l == "en":
            continue
        order = [b for _, b in sorted(((-p1[l][b], b) for b in langs if b != l))]
        ranks.append(order.index("en") + 1)
        nearest += order[0] == "en"
    return {"p1_en_pairs": float(np.mean([p1[a][b] for a, b in en_pairs])),
            "p1_non_en_pairs": float(np.mean([p1[a][b] for a, b in non_pairs])),
            "en_hub_rank": float(np.mean(ranks)), "en_nearest_for": int(nearest),
            "n_non_en": len(ranks), "n_sent": int(n)}


def report_hubness(results="results/retrieval_bgem3.json"):
    print("\n  ENGLISH-CENTRICITY of the retrieval representation (FLORES 10-way, from the saved "
          "embeddings; architecture-neutral, so byte AND subword)")
    print("  pairs INVOLVING English vs pairs between two NON-English languages, cross-lingual P@1; "
          "hub rank = where English sits among each language's 9 partners (1 = nearest, 5 = no hub)")
    print(f"  {'model':15}{'P@1 en-pairs':>14}{'P@1 non-en':>12}{'gap':>8}{'hub rank':>10}{'en nearest':>12}")
    rows = {}
    for n in MAIN_MODELS:
        h = flores_hubness(n)
        if not h:
            continue
        rows[n] = h
        print(f"  {n:15}{h['p1_en_pairs']:>14.3f}{h['p1_non_en_pairs']:>12.3f}"
              f"{h['p1_en_pairs'] - h['p1_non_en_pairs']:>+8.3f}{h['en_hub_rank']:>10.2f}"
              f"{h['en_nearest_for']:>8}/{h['n_non_en']:<3}")
    for size in ("small", "base", "large"):
        b, s = rows.get(f"byte-{size}"), rows.get(f"subword-{size}")
        if b and s:
            db = b["p1_en_pairs"] - b["p1_non_en_pairs"]
            ds = s["p1_en_pairs"] - s["p1_non_en_pairs"]
            print(f"  {size:6} English gap: byte {db:+.3f}  subword {ds:+.3f}  byte−subword {db - ds:+.3f}"
                  f"   (negative = byte LESS English-centric)")
    if rows:
        # Validated on synthetic spaces: pulling every language toward English drives the hub rank
        # from ~3.7 to 1.00 and "en nearest" from 3/9 to 9/9, while the raw P@1 gap moves only
        # +0.02 — because making everything English-like raises ALL pair similarities together.
        # So the RANK columns carry the signal; read the gap as corroboration only.
        print("  reading: hub rank near 1 with English nearest for most languages = English-centric; "
              "hub rank near 5 = English is just another language. The rank statistic is the reliable "
              "one; the P@1 gap is weak by construction and should only corroborate.")


def all_depth_summary(r, n_boot=2000, seed=0):
    """The all-depth arm on Belebele, over the non-English languages, against the UNEDITED pass:
      en      effect of erasing English at every block
      random  effect of the same number of random rank-1 edits at the same blocks   <- matched null
      excess  paired (en - random) per query                                        <- the headline
    Only `excess` is interpretable: n simultaneous edits damage the representation whatever direction
    they point in, and `random` is what measures that damage."""
    if NONE not in r.get("belebele", {}) or ALL_EN not in r.get("belebele", {}):
        return None
    en_v, rnd_v, exc_v = [], [], []
    for l, c in r["belebele"][NONE].items():
        if l == "en" or not c:
            continue
        base = _pq(r, NONE, l)
        if not base:
            continue
        keys = sorted(base)

        def dv(key):
            got = _pq(r, key, l)
            return np.array([got[k] - base[k] for k in keys if k in got]) if got else None
        d_en, d_rnd = dv(ALL_EN), dv(ALL_RND)
        if d_en is not None and len(d_en):
            en_v.append(d_en)
        if d_rnd is not None and len(d_rnd):
            rnd_v.append(d_rnd)
        if d_en is not None and d_rnd is not None and len(d_en) == len(d_rnd):
            exc_v.append(d_en - d_rnd)
    return {"en": _boot(en_v, n_boot, seed), "random": _boot(rnd_v, n_boot, seed),
            "excess": _boot(exc_v, n_boot, seed)}


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
        ec = r.get("erasure_check")
        if ec:
            ai, al = ec["at_intervention"], ec["at_last_block"]
            print(f"    erasure check: English probe balanced accuracy at the intervention block "
                  f"{ec['block']}: {ai['unedited']} -> {ai['erased']}; carried to the last block "
                  f"{ec['last_block']}: {al['unedited']} -> {al['erased']}  (0.5 = English not "
                  f"linearly recoverable; a high value at the last block means later blocks rebuild it)")
        ad = r.get("all_depth_probe")
        if ad:
            blks = sorted(ad, key=int)
            res_v = [ad[b]["erased"] for b in blks if ad[b].get("erased") is not None]
            un_v = [ad[b]["unedited"] for b in blks if ad[b].get("unedited") is not None]
            resid = float(np.mean(res_v)) if res_v else None
            removed = (float(np.mean(un_v)) - resid) / (float(np.mean(un_v)) - 0.5) if (un_v and resid) else None
            print(f"    ALL-DEPTH erasure ({r.get('n_blocks')} blocks at once): probe bacc "
                  + "  ".join(f"b{b}: {ad[b]['unedited']}->{ad[b]['erased']}" for b in blks)
                  + "   (chance everywhere = English is unavailable throughout the encoder)")
            # The intervention is NOT equally strong across models, and the difference runs the wrong
            # way for the hypothesis: a model left with more residual English had LESS removed, so a
            # smaller retrieval excess for it is partly "less was taken away" rather than "it needed
            # English less". Printed next to the excess so the two cannot be read apart.
            if resid is not None:
                print(f"      residual English decodability {resid:.3f} (chance 0.500) = "
                      f"{removed:.0%} of the decodable signal removed; models differ here, and a "
                      f"model with MORE residual had LESS removed -- compare excesses accordingly")
        s_ad = all_depth_summary(r, n_boot, seed)
        if s_ad and s_ad.get("excess"):
            print(f"      Belebele, non-English languages: English erased everywhere "
                  f"{_f(s_ad['en'])}  {_ci(s_ad['en'])};  site-matched random "
                  f"{_f(s_ad['random'])}  {_ci(s_ad['random'])}")
            print(f"      -> EXCESS over the matched control {_f(s_ad['excess'])}  "
                  f"{_ci(s_ad['excess'])}   (the only interpretable number here)")
        # The all-depth arm on the 20k pools. These cells (MIRACL / Amharic-PR / CIRAL / AfriQA) are
        # independent of FLORES, so unlike Belebele they are not scored on text the erasers were
        # fitted on -- this is the primary readout, and it had no row at all until now.
        bat = r.get("battery") or {}
        if NONE in bat and ALL_EN in bat:
            got = {}
            for e in (ALL_EN, ALL_RND):
                if e not in bat:
                    continue
                rows = compare(bat[e], bat[NONE], n_boot)
                mono = [(c, x) for c, x in rows if tuple(c) not in CROSS_CELLS]
                got[e] = {"mono": float(np.mean([x["delta"] for _, x in mono])) if mono else None,
                          "nsig": sum(x["significant"] for _, x in mono), "n": len(mono), "rows": rows}
                print(f"      20k battery, {e}: monolingual mean d {_f(got[e]['mono'])} "
                      f"({got[e]['nsig']}/{got[e]['n']} cells significant)")
            a, b_ = got.get(ALL_EN), got.get(ALL_RND)
            if a and b_ and None not in (a["mono"], b_["mono"]):
                exc = a["mono"] - b_["mono"]
                summary.setdefault(n, {})["battery_alldepth_excess"] = exc
                print(f"      -> 20k battery EXCESS over the site-matched control {exc:+.4f}   "
                      f"(FLORES-independent cells: the primary readout)")
                print("      per cell: " + "  ".join(
                    f"{c[0][:6]}-{c[1]}:{x['delta']:+.3f}{'*' if x['significant'] else ''}"
                    for c, x in a["rows"]))
        rs = r.get("reinstatement")
        if rs:
            at = rs["at"]
            blks = sorted(at, key=int)
            print(f"    reinstatement: English erased at the SHALLOWEST block {rs['block']}, probe "
                  f"balanced accuracy unedited->erased by depth "
                  + "  ".join(f"b{b}: {at[b]['unedited']}->{at[b]['erased']}" for b in blks))
            back = [b for b in blks if int(b) > rs["block"] and (at[b]["erased"] or 0) > 0.7]
            print(f"      -> {'REBUILT at ' + ', '.join('b' + b for b in back) if back else 'not rebuilt'}"
                  f" (>0.7 = the concept is linearly recoverable again downstream of the edit)")
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
            print("    single-block arms pruned -> the all-depth arm above is the whole result")
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
    report_hubness(results)


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
    # the eraser must make English unrecoverable on the data it was fitted on
    Xe = apply_rank1(X, *fits["en"])
    assert latent_probe(Xe, lang_rows, fit_mask)["bacc"] < 0.7 < latent_probe(X, lang_rows, fit_mask)["bacc"]
    # all_depth_summary: only `excess` is interpretable, so it must subtract the matched control.
    # English costs 0.10 here and the site-matched random control 0.06 -> excess must be -0.04, NOT
    # the -0.10 a report that ignored the control would print.
    def _bel(off):
        return {l: {"ndcg@10": 0.8 + off, "per_query": {f"q{i}": 0.8 + off for i in range(40)}}
                for l in ("te", "bn", "en")}
    rad = {"belebele": {NONE: _bel(0.0), ALL_EN: _bel(-0.10), ALL_RND: _bel(-0.06)}}
    sad = all_depth_summary(rad, n_boot=200)
    assert abs(sad["en"]["mean"] + 0.10) < 1e-9 and abs(sad["random"]["mean"] + 0.06) < 1e-9, sad
    assert abs(sad["excess"]["mean"] + 0.04) < 1e-9, sad["excess"]
    assert sad["en"]["n_lang"] == 2, "English itself must be held out of the population"
    assert all_depth_summary({"belebele": {NONE: _bel(0.0)}}) is None      # arm absent -> no row
    # devtest keeps the legacy filename (finished runs stay cached); any other split gets its own,
    # so the flag cannot hand back directions fitted on different text.
    assert single_block_arms({}) == (NONE, "en", "random")
    full = {"battery": {NONE: {}, "en": {}, "random": {}, ALL_EN: {}, ALL_RND: {}},
            "erasure_check": {}, "reinstatement": {}, "all_depth_fit": FIT_TAG}
    assert english_done(full)
    assert not english_done({**full, "battery": {k: v for k, v in full["battery"].items()
                                                 if k != ALL_EN}}), "missing all-depth arm != done"
    assert not english_done({k: v for k, v in full.items() if k != "reinstatement"})
    # pruned: the single-block arms and erasure_check are gone ON PURPOSE and must not block `done`
    pruned = {PRUNED: True, "battery": {NONE: {}, ALL_EN: {}, ALL_RND: {}}, "reinstatement": {}}
    assert not english_done(pruned), "no all_depth_fit marker -> stacked -> not finished"
    pruned = {**pruned, "all_depth_fit": FIT_TAG}
    assert english_done(pruned), "a pruned file with a SEQUENTIAL all-depth arm is finished"
    assert not english_done({**pruned, "all_depth_fit": "stacked"}), "stacked arm must re-run"
    assert not english_done({**pruned, "all_depth_fit": "sequential"}), "single-chain arm must re-run"
    assert not english_done({**full, "all_depth_fit": None}), "unmarked arm must re-run"
    assert not english_done({**pruned, "battery": {NONE: {}}}), "pruned but no all-depth arm"
    assert single_block_arms({PRUNED: True}) == (NONE,)     # pruned -> stage D stops at the baseline
    assert flores_splits("devtest") == ["devtest"] and flores_splits("dev+devtest") == ["dev", "devtest"]
    assert fits_path("m") == fits_path("m", "devtest") != fits_path("m", "dev")
    assert fits_path("m", "dev+devtest") not in (fits_path("m", "dev"), fits_path("m", "devtest"))
    assert fits_all_path("m") != fits_all_path("m", "dev") != fits_path("m", "dev")
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
    ap.add_argument("--flores-split", default="devtest", choices=["devtest", "dev", "dev+devtest"],
                    help="FLORES text the erasers are fitted on. One split = fit on a random half, "
                         "probe on the other. 'dev+devtest' = fit on dev, probe on devtest: ~2x the "
                         "positions on each side and a fit/probe boundary that is a real split "
                         "boundary. Run --flores-overlap first: whichever split Belebele's passages "
                         "came from is not held out from the Belebele cells.")
    ap.add_argument("--prune-single-block", action="store_true",
                    help="drop the one-block erasure arms from every part file and stop them being "
                         "recomputed; the unedited baseline, latent probe and shift are kept")
    ap.add_argument("--flores-overlap", action="store_true",
                    help="measure how much Belebele passage text each FLORES split supplies, then exit")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.prune_single_block:
        return prune_single_block()
    if a.flores_overlap:
        return flores_overlap(a.ckpt_dir)
    if a.merge:
        return merge(a.results, a.n_boot, a.seed)
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.seed, a.skip_battery, a.flores_split)


if __name__ == "__main__":
    main()
