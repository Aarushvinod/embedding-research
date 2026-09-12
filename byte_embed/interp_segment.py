"""Exp 5 — emergent segmentation in the byte students (byte models only).

Claim under test: byte models develop segmentation on their own, and different languages develop
their own rather than sharing one structure imposed by a subword tokenizer.

Setup: all ten languages, 500 FLORES sentences each (300 for byte-large, whose per-position
extraction is ~3x the cost); per-byte-position residual states at every layer (0 = embeddings, then
every block) for small/base — byte-large has 37 states, so it is probed on a subsampled grid (layers
0-4 densely, then every third, plus the last; see `layer_plan`) and its curve is coarser than the
other two models'; per-position labels
  tok_end   last byte of a BGE-M3 (teacher-tokenizer) token, with the standalone-'▁' correction
            (`teacher_cuts`: the XLM-R fast tokenizer gives a lone '▁' the span of the NEXT word's
            first character, which would plant a bogus boundary one character into 1-6% of words);
  interior  a token end strictly inside a run of letters/marks/digits, contrasted ONLY with in-run
            non-boundaries — the non-trivial sub-word boundary (a word boundary is trivially
            decodable from the 0x20 byte in the input: the "spaces trap");
  word_end  last byte before whitespace (undefined for Chinese);
  random    count-matched random character-final positions (chance calibration).
Candidates are character-final bytes only (a mid-character byte can never be a boundary in a
multi-byte script); whitespace bytes are never candidates. One logistic-regression probe per layer,
label and language, balanced accuracy, train/test split BY SENTENCE with bootstrap intervals over
test sentences.
Robustness: (a) a surface-statistics baseline — the same probe on a one-hot window of raw bytes
around the position, no model; (b) the same probes on the untouched PRETRAINED ByT5 encoder, so
what distillation added is separated from what ByT5 already had.
Core result — cross-lingual transfer at the peak layer: a probe trained on language A tested on
held-out sentences of language B for all 100 pairs, reported as AUC(A->B) / AUC(B->B). Each language
is standardized with its own statistics and the score is the threshold-free ROC-AUC of the probe's
decision function, NOT its hard-label accuracy: A's intercept is meaningless in B's re-centred
coordinates, so an accuracy-based matrix would report "no transfer" whenever only the bias failed to
carry over. Balanced accuracy is recorded alongside for reference. One joint probe trained on all
languages is scored per language as well.

Pre-registered readings: within-language accuracy must beat the surface baseline and the random
control for the representation to count as carrying segmentation; the pretrained comparison says
whether distillation added any. Transfer: low ratios even within a script (en->sw, sw->rw) mean
LANGUAGE-specific segmentation (the claim); high within-script / low cross-script mean script-
specific features; high everywhere means one shared feature. Cross-script cells are expected to be
low on byte-overlap grounds alone, so the same-script cells carry the result.

  python -m byte_embed.interp_segment --only byte-small
  python -m byte_embed.interp_segment --only byte-small --langs en,zh --n-sent 50   # fast path
  python -m byte_embed.interp_segment --merge
  python -m byte_embed.interp_segment --selftest
"""
from __future__ import annotations

import argparse
import unicodedata
import zlib

import numpy as np

from byte_embed.interp_common import (LATIN, SCRIPT, byte_offsets, flores_parallel, layer_positions,
                                      load_student, merge_parts, models_in, n_blocks, part_path,
                                      read_json, utf8_stdout, write_json)

ANALYSIS = "segment"
SCHEMA = 2                         # bumped when the part-file contents change meaning
LABELS = ("tok_end", "interior", "word_end", "random")
BYTE_MODELS = ["byte-small", "byte-base", "byte-large"]
NO_SPACES = {"zh"}
SURFACE_K = 4                      # bytes of context on each side for the surface baseline


# ----------------------------------------------------------------------------------------------
# labels (pure python)
# ----------------------------------------------------------------------------------------------
def _is_word_char(ch):
    """Letters, marks (Indic vowel signs, Arabic diacritics) and digits — anything that is not
    whitespace, punctuation, a symbol or a control character."""
    return unicodedata.category(ch)[0] in "LMN"


def position_labels(text, cuts, lang, max_chars=512):
    """Per-byte-position labels for one text. `cuts` = CHARACTER offsets of teacher-token ends: a
    cut at c means a token ends after character c-1, so the boundary sits on the LAST BYTE of
    character c-1. Returns None for texts shorter than two characters, else
    {name: (positives: set(byte_pos) | None if undefined, universe: sorted candidate byte positions
    the label is defined over — negatives are drawn from universe minus positives)}."""
    starts, n_bytes = byte_offsets(text, max_chars)
    text = text[:max_chars]
    L = len(text)
    if L < 2:
        return None

    def last(c):                                            # last byte of character c
        return (starts[c + 1] if c + 1 < L else n_bytes) - 1
    cut_chars = {c - 1 for c in cuts if 0 < c < L}          # a cut at c = boundary after char c-1
    cand = [c for c in range(L - 1) if not text[c].isspace()]
    inrun = [c for c in cand if _is_word_char(text[c]) and _is_word_char(text[c + 1])]
    uni_all = [last(c) for c in cand]
    tok_end = {last(c) for c in cand if c in cut_chars}
    word_end = None if lang in NO_SPACES else {last(c) for c in cand if text[c + 1].isspace()}
    interior = {last(c) for c in inrun if c in cut_chars}
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    k = min(len(tok_end), len(uni_all))
    random = set(rng.choice(uni_all, size=k, replace=False).tolist()) if k else set()
    return {"tok_end": (tok_end, uni_all), "interior": (interior, [last(c) for c in inrun]),
            "word_end": (word_end, uni_all), "random": (random, uni_all)}


def build_dataset(texts, cuts_list, lang, rng, max_chars=512, per_text_cap=40):
    """Balanced positives/negatives per label -> (sel: positions per text, items: {label: [(text_idx,
    pos, y)]}). Negatives come from that label's OWN universe minus its positives, count-matched;
    per_text_cap bounds the extraction cost."""
    sel, items = [], {l: [] for l in LABELS}
    for i, (t, cuts) in enumerate(zip(texts, cuts_list)):
        labs = position_labels(t, cuts, lang, max_chars)
        keep = set()
        if labs is not None:
            for name in LABELS:
                positives, universe = labs[name]
                if positives is None:
                    continue
                pos = sorted(positives)
                if len(pos) > per_text_cap // 2:
                    pos = sorted(rng.choice(pos, size=per_text_cap // 2, replace=False).tolist())
                neg_pool = [u for u in universe if u not in positives]
                neg = sorted(rng.choice(neg_pool, size=min(len(pos), len(neg_pool)),
                                        replace=False).tolist()) if neg_pool and pos else []
                items[name].extend((i, p, 1) for p in pos)
                items[name].extend((i, p, 0) for p in neg)
                keep.update(pos); keep.update(neg)
        sel.append(sorted(keep))
    return sel, items


def teacher_cuts(text, tok):
    """Teacher-token END character offsets, like boundaries.boundary_cuts, minus the ends of standalone
    '▁' tokens (see the module docstring)."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=512)
    toks = tok.convert_ids_to_tokens(enc["input_ids"])
    return sorted({e for t, (_, e) in zip(toks, enc["offset_mapping"]) if t != "▁" and 0 < e < len(text)})


def surface_features(texts, items, k=SURFACE_K, max_chars=512):
    """The no-model baseline: one-hot of the raw bytes in the window [p-k, p+k] around each labelled
    position of the (char-truncated) text."""
    rows = []
    for i, p, _ in items:
        b = texts[i][:max_chars].encode("utf-8")
        f = np.zeros((2 * k + 1, 256), np.float32)
        for j, off in enumerate(range(-k, k + 1)):
            q = p + off
            if 0 <= q < len(b):
                f[j, b[q]] = 1.0
        rows.append(f.ravel())
    return np.stack(rows, 0) if rows else np.zeros((0, (2 * k + 1) * 256), np.float32)


# ----------------------------------------------------------------------------------------------
# probes (numpy + sklearn)
# ----------------------------------------------------------------------------------------------
def group_split(y, groups, seed=0, test_size=0.2):
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(np.zeros(len(y)), y, groups))
    return tr, te


def fit_probe(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=10.0, random_state=seed).fit(sc.transform(X), y)
    return sc, clf


def probe_metrics(X, y, groups, seed=0, n_boot=200):
    """Balanced accuracy of a linear probe with the train/test split BY SENTENCE (groups) and a
    bootstrap interval over the test sentences. None if the label is degenerate."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    X, y, groups = np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)
    if len(set(y.tolist())) < 2 or len(y) < 20 or len(set(groups.tolist())) < 5:
        return None
    tr, te = group_split(y, groups, seed)
    if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
        return None
    sc, clf = fit_probe(X[tr], y[tr], seed)
    pred = clf.predict(sc.transform(X[te]))
    yt, gt = y[te], groups[te]
    ug = np.unique(gt)
    idx = {g: np.flatnonzero(gt == g) for g in ug}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        if len(set(yt[ii].tolist())) > 1:
            boots.append(balanced_accuracy_score(yt[ii], pred[ii]))
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (float("nan"), float("nan")))
    return {"acc": round(float((pred == yt).mean()), 3),
            "bacc": round(float(balanced_accuracy_score(yt, pred)), 3),
            "bacc_ci": [round(float(lo), 3), round(float(hi), 3)],
            "f1": round(float(f1_score(yt, pred)), 3), "n": int(len(y)), "n_test_sent": int(len(ug))}


def layer_plan(n_layers):
    """All layers for small/base; for byte-large (37 states) the early band densely + every 3rd."""
    if n_layers <= 20:
        return list(range(n_layers))
    return sorted(set([0, 1, 2, 3, 4] + list(range(7, n_layers, 3)) + [n_layers - 1]))


def peak_layer(res, key="layers"):
    """Layer with the highest mean interior balanced accuracy across languages (trained probes)."""
    acc = {}
    for lang, d in res["langs"].items():
        for e in d.get(key) or []:
            if e.get("interior"):
                acc.setdefault(e["layer"], []).append(e["interior"]["bacc"])
    if not acc:
        return None
    return int(max(acc, key=lambda l: float(np.mean(acc[l]))))


def _lang_sample(par, lang, n_sent, seed):
    rng = np.random.default_rng([seed, zlib.crc32(lang.encode("utf-8"))])
    idx = rng.choice(len(par[lang]), size=min(n_sent, len(par[lang])), replace=False)
    return [par[lang][i] for i in idx], rng


def _probe_layers(feats, index, items, layers, seed):
    row_of = {k: r for r, k in enumerate(index)}
    out = []
    for l in layers:
        entry = {"layer": int(l)}
        for lab in LABELS:
            it = [(row_of[(i, p)], y, i) for i, p, y in items[lab] if (i, p) in row_of]
            if not it:
                entry[lab] = None
                continue
            rows, y, g = zip(*it)
            entry[lab] = probe_metrics(feats[l][list(rows)], y, g, seed)
        out.append(entry)
    return out


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, langs=None, n_sent=None, seed=0):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:          # first-round part files (position-level splits) are discarded
        res = {"model": name, "langs": {}, "schema": SCHEMA}
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    if bm.get("kind") != "byte":
        print(f"  [segment] {name} is not a byte model -> skip")
        return
    # The pretrained baseline is a SECOND full encoder resident on the same GPU. Load it on first
    # use, so a resumed run whose languages are all done (only `transfer` left) never pays for it.
    pre = None

    def pretrained():
        nonlocal pre
        if pre is None:
            pre = load_student(name, results, ckpt_dir, device, pretrained_only=True)[0]
        return pre
    from byte_embed.boundaries import _teacher_tok
    tok = _teacher_tok()
    par = flores_parallel(cache_dir=ckpt_dir)
    langs = langs or list(par)
    bad = [l for l in langs if l not in par]
    if bad:
        raise SystemExit(f"unknown langs {bad}; choose from {list(par)}")
    n_sent = n_sent or (300 if "large" in name else 500)
    big = "large" in name
    res.update(kind="byte", steps_run=bm.get("steps_run"), n_sent=n_sent)
    layers = layer_plan(n_blocks(student) + 1)                 # embeddings + every block

    for lang in langs:
        d = res["langs"].get(lang) or {}
        if d.get("layers") and d.get("surface") and d.get("pretrained"):
            print(f"  {name}/{lang}: done -> skip")
            continue
        # per-language generator: sampling depends only on (seed, lang), so a requeued run that
        # skips finished languages draws exactly what an uninterrupted run would have drawn
        texts, rng = _lang_sample(par, lang, n_sent, seed)
        cuts_list = [teacher_cuts(t[:student.max_chars], tok) for t in texts]
        sel, items = build_dataset(texts, cuts_list, lang, rng, max_chars=student.max_chars)
        d.setdefault("n_sent", len(texts))
        if not d.get("layers"):
            feats, index = layer_positions(student, texts, sel, device=device, layers=layers,
                                           batch_size=(4 if big else 8))
            d["layers"] = _probe_layers(feats, index, items, layers, seed)
            del feats
        if not d.get("surface"):
            d["surface"] = {}
            for lab in LABELS:
                it = items[lab]
                if not it:
                    d["surface"][lab] = None
                    continue
                X = surface_features(texts, it, max_chars=student.max_chars)
                d["surface"][lab] = probe_metrics(X, [y for _, _, y in it], [i for i, _, _ in it], seed)
        if not d.get("pretrained"):
            feats, index = layer_positions(pretrained(), texts, sel, device=device, layers=layers,
                                           batch_size=(4 if big else 8))
            d["pretrained"] = _probe_layers(feats, index, items, layers, seed)
            del feats
        res["langs"][lang] = d
        write_json(outp, res)
        e1 = next((e for e in d["layers"] if e["layer"] == 1), d["layers"][0])
        pk = max((e for e in d["layers"] if e.get("interior")), key=lambda e: e["interior"]["bacc"], default=None)
        print(f"  {name}/{lang}: interior bacc L1={e1['interior']['bacc'] if e1.get('interior') else None} "
              f"peak={pk['interior']['bacc'] if pk else None}@L{pk['layer'] if pk else '-'} "
              f"surface={d['surface']['interior']['bacc'] if d['surface'].get('interior') else None} "
              f"random={e1['random']['bacc'] if e1.get('random') else None} -> saved")

    # Recompute when the LANGUAGE SET or the sample size changed: the documented fast path
    # (`--langs en,zh --n-sent 50`) would otherwise pin a 2x2 matrix computed from 50 sentences that
    # the later full run never revisits, while status.sh and the figures present it as the 10x10.
    want = {"langs": sorted(res["langs"]), "n_sent": int(n_sent)}
    have = {k: (res.get("transfer") or {}).get(k) for k in want}
    if len(res["langs"]) >= 2 and have != want:
        if res.get("transfer"):
            print(f"  transfer: recomputing - was {have}, now {want}")
        t = transfer_matrix(student, par, list(res["langs"]), res, tok, n_sent, seed, device, big)
        if t:                      # only persist a real matrix, so a skipped/degenerate pass retries
            res["transfer"] = {**t, **want}
            write_json(outp, res)
    print(f"  saved -> {outp}")


def transfer_matrix(student, par, langs, res, tok, n_sent, seed, device, big):
    """Cross-lingual transfer of the INTERIOR-boundary probe at the peak layer, plus a joint probe.
    Each language is standardized with its own training-split statistics."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    # peak_layer is None when no layer has a usable interior probe; layers=[None] then raises
    # TypeError inside _select_layers at the very end of a multi-hour job.
    peak = peak_layer(res)
    if peak is None:
        print("  transfer: no usable interior probe at any layer -> skipped")
        return None
    print(f"  transfer pass at peak layer {peak}")
    data = {}
    for lang in langs:
        texts, rng = _lang_sample(par, lang, n_sent, seed)
        cuts_list = [teacher_cuts(t[:student.max_chars], tok) for t in texts]
        _, items = build_dataset(texts, cuts_list, lang, rng, max_chars=student.max_chars)
        it = items["interior"]
        if len(it) < 40:
            continue
        sel = [[] for _ in texts]
        for i, p, _ in it:
            sel[i].append(p)
        sel = [sorted(set(s)) for s in sel]
        feats, index = layer_positions(student, texts, sel, device=device, layers=[peak],
                                       batch_size=(4 if big else 8))
        row_of = {k: r for r, k in enumerate(index)}
        rows = [(row_of[(i, p)], y, i) for i, p, y in it if (i, p) in row_of]
        X = feats[peak][[r for r, _, _ in rows]]
        y, g = np.array([y for _, y, _ in rows]), np.array([i for _, _, i in rows])
        tr, te = group_split(y, g, seed)
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        data[lang] = {"Xtr": sc.transform(X[tr]), "ytr": y[tr], "Xte": sc.transform(X[te]), "yte": y[te]}
        del feats
    if len(data) < 2:              # `cap = min(...)` below is a ValueError on an empty `data`, and a
        print(f"  transfer: only {len(data)} language(s) usable at layer {peak} -> skipped")
        return None                # 1x1 matrix carries no transfer signal anyway
    from sklearn.linear_model import LogisticRegression
    clfs = {a: LogisticRegression(max_iter=2000, C=10.0, random_state=seed).fit(data[a]["Xtr"], data[a]["ytr"])
            for a in data}
    # AUC of the decision function, not hard-label accuracy: each language is standardized with its
    # OWN statistics, so probe A's intercept does not transport to B's coordinates and a thresholded
    # score would read "no transfer" when only the bias failed to carry.
    auc = {a: {b: round(float(roc_auc_score(data[b]["yte"], clfs[a].decision_function(data[b]["Xte"]))), 3)
               for b in data} for a in data}
    acc = {a: {b: round(float(balanced_accuracy_score(data[b]["yte"], clfs[a].predict(data[b]["Xte"]))), 3)
               for b in data} for a in data}
    ratio = {a: {b: round(auc[a][b] / auc[b][b], 3) if auc[b][b] else None for b in data} for a in data}
    # one subsample per language, seeded by (seed, lang): the joint probe must not depend on the
    # order in which languages happened to finish (a requeue reorders `res["langs"]`).
    cap = min(len(d["ytr"]) for d in data.values())
    parts = {l: np.random.default_rng([seed, zlib.crc32(l.encode("utf-8"))])
             .choice(len(d["ytr"]), size=cap, replace=False) for l, d in data.items()}
    Xj = np.concatenate([data[l]["Xtr"][ii] for l, ii in parts.items()], 0)
    yj = np.concatenate([data[l]["ytr"][ii] for l, ii in parts.items()], 0)
    joint = LogisticRegression(max_iter=2000, C=10.0, random_state=seed).fit(Xj, yj)
    return {"layer": int(peak), "auc": auc, "acc": acc, "ratio": ratio,
            "within": {b: auc[b][b] for b in data},
            "within_bacc": {b: acc[b][b] for b in data},
            "joint": {b: round(float(roc_auc_score(data[b]["yte"], joint.decision_function(data[b]["Xte"]))), 3)
                      for b in data}}


# ----------------------------------------------------------------------------------------------
def _m(vals, w=9):
    return f"{np.mean(vals):>{w}.3f}" if vals else f"{'—':>{w}}"


def merge():
    d = merge_parts(ANALYSIS, schema=SCHEMA)
    M = d["models"]
    print("\nEXP 5 — EMERGENT SEGMENTATION (balanced accuracy of per-byte-position boundary probes, split by "
          "sentence; `random` = chance control)")
    for n in BYTE_MODELS:
        r = M.get(n)
        if not r or not r.get("langs"):
            continue
        langs = list(r["langs"])
        sp = [l for l in langs if l not in NO_SPACES]
        print(f"\n  {n}  (mean over {len(sp)} space-delimited langs; zh separately; "
              f"'pre' = untouched pretrained ByT5)")
        print(f"  {'layer':>6}{'tok_end':>9}{'interior':>10}{'pre int':>9}{'word_end':>10}{'random':>8}"
              f"{'  | zh interior':>15}{'zh pre':>8}")
        layer_ids = [e["layer"] for e in r["langs"][langs[0]]["layers"]]
        for l in layer_ids:
            v = {k: [] for k in ("tok_end", "interior", "pre", "word_end", "random")}
            zh_v, zh_pre = None, None
            for lang in langs:
                e = next((x for x in r["langs"][lang]["layers"] if x["layer"] == l), None)
                p = next((x for x in (r["langs"][lang].get("pretrained") or []) if x["layer"] == l), None)
                if not e:
                    continue
                if lang in NO_SPACES:
                    zh_v = e["interior"]["bacc"] if e.get("interior") else None
                    zh_pre = p["interior"]["bacc"] if p and p.get("interior") else None
                    continue
                for lab in ("tok_end", "interior", "word_end", "random"):
                    if e.get(lab):
                        v[lab].append(e[lab]["bacc"])
                if p and p.get("interior"):
                    v["pre"].append(p["interior"]["bacc"])
            zt = f"{zh_v:>15.3f}" if zh_v is not None else f"{'—':>15}"
            zp = f"{zh_pre:>8.3f}" if zh_pre is not None else f"{'—':>8}"
            print(f"  {l:>6}{_m(v['tok_end'])}{_m(v['interior'], 10)}{_m(v['pre'])}{_m(v['word_end'], 10)}"
                  f"{_m(v['random'], 8)}{zt}{zp}")
        sur = [r["langs"][l]["surface"]["interior"]["bacc"] for l in sp
               if (r["langs"][l].get("surface") or {}).get("interior")]
        print(f"  surface-statistics baseline (interior, mean over space-delimited langs): {_m(sur, 6).strip()}")
    report_by_language(M)
    report_transfer(M)
    print("\n  reading: interior above BOTH the surface baseline and random = the representation carries "
          "segmentation; trained > pretrained = distillation added some; transfer ratios low within a "
          "script -> language-specific segmentation (the claim); high within / low across -> script-specific; "
          "high everywhere -> one shared feature.")


def report_by_language(M):
    print("\n  PER-LANGUAGE interior-boundary decodability: trained model at layer 1 / at its peak "
          "(layer), the PRETRAINED encoder both at that same layer and at its OWN peak, surface "
          "baseline, random.")
    print("  (the trained model's peak is an argmax over test scores; scoring the pretrained encoder "
          "only there would hand the trained model its best layer and the baseline an imposed one, so "
          "the pretrained argmax is shown too — the honest 'what distillation added' is peak vs pre-peak.)")
    for n in BYTE_MODELS:
        r = M.get(n)
        if not r or not r.get("langs"):
            continue
        print(f"\n  {n:12}{'lang':>5}{'script':>7}{'L1':>7}{'peak':>7}{'@L':>4}{'pre@pk':>8}"
              f"{'pre peak':>9}{'@L':>4}{'surface':>9}{'random':>8}")
        peaks = {}
        for lang, d in r["langs"].items():
            def get(e, lab):
                return e[lab]["bacc"] if e and e.get(lab) else None
            vals = [(e["layer"], get(e, "interior")) for e in d["layers"] if get(e, "interior") is not None]
            if not vals:
                continue
            by = dict(vals)
            pk = max(vals, key=lambda t: t[1])
            pre = next((get(e, "interior") for e in (d.get("pretrained") or []) if e["layer"] == pk[0]), None)
            pre_vals = [(e["layer"], get(e, "interior")) for e in (d.get("pretrained") or [])
                        if get(e, "interior") is not None]
            pre_pk = max(pre_vals, key=lambda t: t[1]) if pre_vals else None
            sur = get(d.get("surface"), "interior") if d.get("surface") else None
            rnd = [get(e, "random") for e in d["layers"] if get(e, "random") is not None]
            peaks[lang] = (pk[1], pre_pk[1] if pre_pk else None)
            f = lambda v, w=7: f"{v:>{w}.3f}" if v is not None else f"{'-':>{w}}"  # noqa: E731
            print(f"  {'':12}{lang:>5}{SCRIPT.get(lang, '?'):>7}{f(by.get(1))}{f(pk[1])}{pk[0]:>4}{f(pre, 8)}"
                  f"{f(pre_pk[1] if pre_pk else None, 9)}{(pre_pk[0] if pre_pk else '-'):>4}"
                  f"{f(sur, 9)}{f(float(np.mean(rnd)) if rnd else None, 8)}")
        lat = [v for l, (v, _) in peaks.items() if l in LATIN]
        non = [v for l, (v, _) in peaks.items() if l not in LATIN]
        gap = [v - p for v, p in peaks.values() if p is not None]
        print(f"  {'':12}peak mean: Latin-script {np.mean(lat):.3f} ({len(lat)})   non-Latin "
              f"{np.mean(non):.3f} ({len(non)})"
              + (f"   trained peak − pretrained peak: {np.mean(gap):+.3f}" if gap else ""))


def report_transfer(M):
    print("\n  CROSS-LINGUAL TRANSFER of the interior-boundary probe at the peak layer: "
          "AUC(A->B) / AUC(B->B) (rows = trained on A, columns = tested on B; threshold-free, so only "
          "the direction has to transfer); joint = one probe trained on all languages")
    for n in BYTE_MODELS:
        r = M.get(n)
        t = (r or {}).get("transfer")
        if not t:
            continue
        langs = list(t.get("auc") or t["acc"])
        print(f"  {'':14}(computed over {len(t.get('langs') or langs)} languages, "
              f"{t.get('n_sent', '?')} sentences each)")
        corner = "A|B"
        print(f"\n  {n} (layer {t['layer']})  {corner:>6}" + "".join(f"{b:>6}" for b in langs))
        for a in langs:
            print(f"  {'':14}{a:>6}" + "".join(f"{(t['ratio'][a][b] if t['ratio'][a][b] is not None else float('nan')):>6.2f}"
                                             for b in langs))
        print(f"  {'':14}{'joint':>6}" + "".join(f"{t['joint'][b] / t['within'][b] if t['within'][b] else float('nan'):>6.2f}"
                                               for b in langs) + "   (joint / within)")
        same = [t["ratio"][a][b] for a in langs for b in langs if a != b and SCRIPT[a] == SCRIPT[b]
                and t["ratio"][a][b] is not None]
        cross = [t["ratio"][a][b] for a in langs for b in langs if a != b and SCRIPT[a] != SCRIPT[b]
                 and t["ratio"][a][b] is not None]
        A = t.get("auc") or {}
        raw = lambda pairs: (np.mean([A[a][b] for a, b in pairs]) if pairs and A else float("nan"))  # noqa: E731
        same_p = [(a, b) for a in langs for b in langs if a != b and SCRIPT[a] == SCRIPT[b]]
        cross_p = [(a, b) for a in langs for b in langs if a != b and SCRIPT[a] != SCRIPT[b]]
        print(f"  {'':14}mean ratio same-script pairs {np.mean(same) if same else float('nan'):.2f} (n={len(same)})"
              f"   cross-script pairs {np.mean(cross) if cross else float('nan'):.2f} (n={len(cross)})"
              f"   within-language AUC {np.mean(list(t['within'].values())):.3f}")
        # The ratio's denominator is the diagonal at a layer chosen to maximise exactly that, so the
        # RAW AUCs are printed too: the absolute "does the direction transfer at all" reading should
        # rest on them (0.5 = no transfer), not on a ratio with a selected denominator.
        print(f"  {'':14}raw AUC       same-script {raw(same_p):.3f}   cross-script {raw(cross_p):.3f}"
              f"   within-language {np.mean(list(t['within'].values())):.3f}   (0.5 = chance)")


def _selftest():
    labs = position_labels("ab cd", cuts=[1, 2], lang="en")     # tokens: a | b | ' cd'
    assert labs["tok_end"] == ({0, 1}, [0, 1, 3]), labs
    assert labs["word_end"][0] == {1} and labs["interior"] == ({0}, [0, 3]), labs
    labs_zh = position_labels("字字字", cuts=[1, 2], lang="zh")
    assert labs_zh["word_end"][0] is None and labs_zh["tok_end"][0] == {2, 5}, labs_zh
    labs_te = position_labels("తెలుగు వికీ", cuts=[2, 4], lang="te")
    assert labs_te["interior"][0] == {5, 11} and labs_te["word_end"][0] == {17}, labs_te
    class FakeTok:
        def __call__(self, text, **kw):
            return {"input_ids": [7, 6, 8], "offset_mapping": [(0, 2), (3, 4), (3, 6)]}

        def convert_ids_to_tokens(self, ids):
            return ["▁ab", "▁", "fox"]
    assert teacher_cuts("ab fox", FakeTok()) == [2]
    rng = np.random.default_rng(0)
    texts = [f"ab cd{i}" for i in range(30)]
    sel, items = build_dataset(texts, [[1, 2]] * 30, "en", rng)
    assert all(len(s) > 0 for s in sel) and len(items["interior"]) == 60
    S = surface_features(texts, items["interior"])
    assert S.shape == (60, (2 * SURFACE_K + 1) * 256) and S.sum(1).max() <= 2 * SURFACE_K + 1
    y = [y for _, _, y in items["interior"]]
    g = [i for i, _, _ in items["interior"]]
    m = probe_metrics(S, y, g)
    assert m is not None and m["bacc"] > 0.9 and m["bacc_ci"][0] <= m["bacc"] <= m["bacc_ci"][1], m
    X = rng.standard_normal((600, 16)); yy = (rng.random(600) > 0.5).astype(int)
    X[:, 0] += 4 * yy                                            # ~2% Bayes error
    gg = np.repeat(np.arange(60), 10)
    assert probe_metrics(X, yy, gg)["bacc"] > 0.9
    assert abs(probe_metrics(X, rng.permutation(yy), gg)["bacc"] - 0.5) < 0.15
    tr, te = group_split(yy, gg)
    assert not (set(gg[tr]) & set(gg[te]))                        # no sentence on both sides
    assert layer_plan(13) == list(range(13)) and 36 in layer_plan(37)
    res = {"langs": {"en": {"layers": [{"layer": 0, "interior": {"bacc": 0.6}}, {"layer": 1, "interior": {"bacc": 0.8}}]},
                     "te": {"layers": [{"layer": 0, "interior": {"bacc": 0.6}}, {"layer": 1, "interior": {"bacc": 0.7}}]}}}
    assert peak_layer(res) == 1
    print("selftest OK: labels (zh/te multibyte, punctuation/whitespace excluded), balanced dataset, surface "
          "features, sentence-grouped probes with bootstrap CI, peak layer")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--langs", default=None, help="comma list (default: all 10)")
    ap.add_argument("--n-sent", type=int, default=None, help="FLORES sentences/lang (500; 300 large)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge()
    langs = a.langs.split(",") if a.langs else None
    names = [a.only] if a.only else [n for n in BYTE_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, langs, a.n_sent, a.seed)


if __name__ == "__main__":
    main()
