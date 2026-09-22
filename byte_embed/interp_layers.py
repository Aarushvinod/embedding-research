"""Experiments 8 + 9 -- where each architecture becomes multilingual, layer by layer.

Representational, not causal: one forward pass per sentence with every hidden state kept, then two
readouts on the same masked-mean-pooled states. Depth-resolved on purpose -- at the OUTPUT,
distillation into BGE-M3's shared space pushes byte and subword toward the same answer, so any
architectural difference has to show up in the intermediate layers.

EXP 8 -- THE CROSSOVER. Per layer, two nearest-neighbour accuracies in the same geometry:

  LID-centroid   is each sentence closer to its own LANGUAGE's centroid than to any other?
                 -> the representation is organized by language
  P@1            is each sentence closest to its own TRANSLATION?
                 -> the representation is organized by content

They compete for the same neighbourhood structure, so language organization should fall with depth
while content organization rises, and the depth where they cross is where the representation
becomes multilingual. Plus two context columns:

  LID-probe      a 10-way linear probe. Expected at ceiling everywhere -- five of the ten languages
                 are the only users of their script -- so it answers "is language decodable at
                 all", not "is the space organized by it". Reported so nobody mistakes the centroid
                 measure for the probe.
  LID-cent(Latn) the centroid measure on the five LATIN-script languages only (sw yo ha rw en),
                 where no script cue exists. The script-controlled version of the column above.

EXP 9 -- TWO-WAY VARIANCE DECOMPOSITION. On parallel FLORES every point is a (sentence, language)
cell, so total variance splits exactly (balanced two-way ANOVA, no replication) into

  content    the sentence main effect -- what every translation of a sentence shares
  language   the language main effect -- one offset per language, i.e. language identity
  residual   the interaction -- how each language encodes the SAME sentence differently beyond that
             offset; where language-specific encoding of content lives

A multilingual representation is content-heavy and language-light. The decomposition is STANDARDIZED
per dimension by default: T5/mT5 residual streams carry massive-activation dimensions (|x| ~ 1e5,
interp_common.layer_positions) that would otherwise make the raw decomposition a statement about a
handful of pathological coordinates. The raw version is stored alongside, with each layer's
top-dimension variance share, so the two can be compared rather than one silently chosen.

Splits are by SENTENCE: every translation of a sentence lands on the same side, so a probe or a
centroid never sees the test content in another language. Centroids and centring means come from
the training half; everything is scored on the test half.

  python -m byte_embed.interp_layers --selftest
  python -m byte_embed.interp_layers --only byte-small
  python -m byte_embed.interp_layers --merge
"""
from __future__ import annotations

import argparse

import numpy as np

from byte_embed.interp_common import (LATIN, MAIN_MODELS, _hidden_states, flores_parallel,
                                      load_student, merge_parts, merged_path, models_in, n_blocks,
                                      part_path, read_json, tokenize_like_forward, utf8_stdout,
                                      write_json)
from common.eval import l2norm

ANALYSIS = "layers"
SCHEMA = 1
SPLIT = "devtest"                  # 1012 rows: the split every other experiment scores on
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ----------------------------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------------------------
def pooled_layers(student, texts, device, batch_size):
    """([N, n_blocks+1, d] masked-mean hidden states, [N, d_out] real output embeddings).

    Layer indices follow `_hidden_states` (0 = embeddings, N = final-LN output). Float32 throughout:
    mean-pooling does not tame a massive-activation dimension, and 1e5 overflows fp16. The output
    embedding comes from a SEPARATE call to the model's own forward rather than a re-implementation
    of its attentive pooler, so the last point of the depth curve is exactly the vector retrieval
    uses, at the cost of a second pass."""
    import torch
    layers, outs = [], []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            bt = texts[i:i + batch_size]
            b = tokenize_like_forward(student, bt, device)
            hs = _hidden_states(student, student.enc(**b, output_hidden_states=True))
            m = b["attention_mask"].unsqueeze(-1).float()
            den = m.sum(1).clamp(min=1.0)
            layers.append(torch.stack([(h.float() * m).sum(1) / den for h in hs], 1).cpu().numpy())
            outs.append(student(bt, device=device).float().cpu().numpy())
    return np.concatenate(layers, 0), l2norm(np.concatenate(outs, 0))


# ----------------------------------------------------------------------------------------------
# readouts (pure numpy / sklearn -- the selftest covers every one)
# ----------------------------------------------------------------------------------------------
def sentence_split(n_sent, seed=0):
    """(train, test) SENTENCE indices. Grouping by sentence keeps every translation of a sentence on
    one side, so no readout is scored on content it was fitted on in another language."""
    idx = np.random.default_rng(seed).permutation(n_sent)
    return np.sort(idx[:n_sent // 2]), np.sort(idx[n_sent // 2:])


def _cells(X3, sents, lang_idx):
    """Flatten [n_sent, n_lang, d] to (X [k, d], language index per row) for the given subsets."""
    Xs = X3[np.ix_(sents, lang_idx)]
    return Xs.reshape(-1, X3.shape[2]), np.tile(np.arange(len(lang_idx)), len(sents))


def lid_probe(X3, train, test, lang_idx, seed=0):
    """10-way linear language-ID probe accuracy. Standardized, because the massive-activation
    dimensions would otherwise dominate the fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = _cells(X3, train, lang_idx)
    Xte, yte = _cells(X3, test, lang_idx)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


def lid_centroid(X3, train, test, lang_idx):
    """Cosine nearest-centroid language accuracy -- how ORGANIZED by language the space is, which is
    what competes with translation retrieval for the same neighbourhoods. Centred on the training
    mean (the common direction every anisotropic encoder has would otherwise dominate cosine);
    centroids from training sentences, scored on test sentences."""
    Xtr, ytr = _cells(X3, train, lang_idx)
    Xte, yte = _cells(X3, test, lang_idx)
    mu = Xtr.mean(0)
    Ztr, Zte = l2norm(Xtr - mu), l2norm(Xte - mu)
    C = l2norm(np.stack([Ztr[ytr == c].mean(0) for c in range(len(lang_idx))]))
    return float((np.argmax(Zte @ C.T, 1) == yte).mean())


def xling_p1(X3, train, test, lang_center=False):
    """Mean P@1 over every ordered language pair: test sentence i in A must retrieve sentence i in B.

    lang_center=False  subtract the GLOBAL training mean -- removes the anisotropic common direction
                       and nothing else, so this is the representation as the model has it
    lang_center=True   subtract each LANGUAGE's training mean -- removes the language offset too;
                       the gap to the column above is how much the language component is costing
                       alignment at this depth"""
    n_lang, d = X3.shape[1], X3.shape[2]
    mu = (X3[train].mean(0, keepdims=True) if lang_center
          else X3[train].reshape(-1, d).mean(0)[None, None, :])
    Z = X3[test] - mu
    Z = Z / np.clip(np.linalg.norm(Z, axis=2, keepdims=True), 1e-12, None)
    ar, ps = np.arange(len(test)), []
    for a in range(n_lang):
        for b in range(a + 1, n_lang):
            S = Z[:, a] @ Z[:, b].T
            ps += [float((S.argmax(1) == ar).mean()), float((S.argmax(0) == ar).mean())]
    return float(np.mean(ps))


def var_decomp(X3, standardize=True):
    """Exact two-way decomposition of total variance into content / language / residual fractions.

    Balanced design with no replication: Xc = s_i + l_L + e_iL with zero-sum effects, so the cross
    terms vanish and the three fractions sum to 1 to machine precision (asserted in the selftest)."""
    X = X3.astype(np.float64)
    if standardize:
        flat = X.reshape(-1, X.shape[2])
        sd = flat.std(0)
        sd[sd < 1e-12] = 1.0                       # constant coordinates contribute nothing either way
        X = (X - flat.mean(0)) / sd
    Xc = X - X.mean((0, 1))
    s, lg = Xc.mean(1), Xc.mean(0)                 # [n_sent, d] content, [n_lang, d] language
    e = Xc - s[:, None, :] - lg[None, :, :]
    tot = float((Xc ** 2).sum())
    if tot <= 0:
        return {"content": 0.0, "language": 0.0, "residual": 0.0}
    return {"content": float(X.shape[1] * (s ** 2).sum() / tot),
            "language": float(X.shape[0] * (lg ** 2).sum() / tot),
            "residual": float((e ** 2).sum() / tot)}


def top_dim_frac(X3):
    """Share of total variance in the single largest coordinate. The massive-activation diagnostic:
    when this is large, raw-space numbers at this layer are about that one coordinate."""
    v = X3.reshape(-1, X3.shape[2]).var(0)
    return float(v.max() / max(v.sum(), 1e-30))


def crossover(depths, a, b):
    """First depth at which curve `a` rises to meet curve `b`, linearly interpolated. None when a
    stays below b throughout; the first depth when a starts at or above b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d[0] >= 0:
        return float(depths[0])
    for k in range(1, len(d)):
        if d[k - 1] < 0 <= d[k]:
            t = -d[k - 1] / (d[k] - d[k - 1])
            return float(depths[k - 1] + t * (depths[k] - depths[k - 1]))
    return None


def analyse(X3, train, test, langs, seed=0):
    """Every readout for ONE layer's [n_sent, n_lang, d] states."""
    all_idx = list(range(len(langs)))
    latn = [i for i, l in enumerate(langs) if l in LATIN]
    return {"lid_probe": round(lid_probe(X3, train, test, all_idx, seed), 4),
            "lid_centroid": round(lid_centroid(X3, train, test, all_idx), 4),
            "lid_centroid_latn": (round(lid_centroid(X3, train, test, latn), 4)
                                  if len(latn) >= 2 else None),
            "p1": round(xling_p1(X3, train, test, lang_center=False), 4),
            "p1_langcentered": round(xling_p1(X3, train, test, lang_center=True), 4),
            "var": {k: round(v, 5) for k, v in var_decomp(X3, standardize=True).items()},
            "var_raw": {k: round(v, 5) for k, v in var_decomp(X3, standardize=False).items()},
            "top_dim_frac": round(top_dim_frac(X3), 5)}


# ----------------------------------------------------------------------------------------------
def run_one(name, results, ckpt_dir, device, seed=0, split=SPLIT):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA or res.get("split") != split or res.get("seed") != seed:
        res = {"schema": SCHEMA, "split": split, "seed": seed, "layers": {}}
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    nb = n_blocks(student)
    if res.get("out") and len(res["layers"]) == nb + 1:
        print(f"  [layers] {name}: done -> skip")
        return
    par = flores_parallel(cache_dir=ckpt_dir, split=split)
    langs = list(par)
    n_sent = len(par[langs[0]])
    bs = 16 if bm.get("kind") == "byte" else 64
    res.update(model=name, kind=bm.get("kind"), n_blocks=nb, langs=langs, n_sent=n_sent)

    # one pass per language, every layer at once; [n_sent, n_lang, n_layers, d] after stacking
    per_lang, outs = [], []
    for l in langs:
        H, O = pooled_layers(student, par[l], device, bs)
        per_lang.append(H)
        outs.append(O)
        print(f"  [layers] {name}/{l}: {H.shape[1]} layers x d={H.shape[2]}")
    H = np.stack(per_lang, 1)                      # [n_sent, n_lang, n_layers, d]
    O = np.stack(outs, 1)                          # [n_sent, n_lang, d_out]
    del per_lang, outs
    res["d_model"], res["d_out"] = int(H.shape[3]), int(O.shape[2])
    train, test = sentence_split(n_sent, seed)

    for k in range(H.shape[2]):
        if str(k) in res["layers"]:
            continue
        r = analyse(H[:, :, k, :], train, test, langs, seed)
        r["depth"] = round(k / nb, 4)
        res["layers"][str(k)] = r
        write_json(outp, res)
        print(f"    layer {k:>2}/{nb} (depth {k / nb:.2f}): LID-cent {r['lid_centroid']:.3f} "
              f"(Latn {r['lid_centroid_latn']}) P@1 {r['p1']:.3f}  "
              f"content {r['var']['content']:.3f} language {r['var']['language']:.3f}")
    if not res.get("out"):
        res["out"] = analyse(O, train, test, langs, seed)
        write_json(outp, res)
        r = res["out"]
        print(f"    out (d={O.shape[2]}): LID-cent {r['lid_centroid']:.3f} P@1 {r['p1']:.3f}  "
              f"content {r['var']['content']:.3f} language {r['var']['language']:.3f}")
    print(f"  saved -> {outp}")


# ----------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------
def profile(r, key, sub=None):
    """(depths, values) along the encoder, the output excluded -- it is not a depth."""
    L = r.get("layers") or {}
    ks = sorted(L, key=int)
    get = (lambda d: d[key][sub]) if sub else (lambda d: d[key])
    return [L[k]["depth"] for k in ks], [get(L[k]) for k in ks]


def _f(v, w=8, p=3):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"


def report_model(name, r):
    print(f"\n  {name}  ({r.get('n_blocks')} blocks, d_model {r.get('d_model')}, "
          f"{r.get('n_sent')} sentences x {len(r.get('langs') or [])} languages)")
    print(f"    {'layer':>6}{'depth':>7}{'LIDprobe':>9}{'LIDcent':>8}{'(Latn)':>8}{'P@1':>7}"
          f"{'(lctr)':>8}{'content':>9}{'language':>9}{'resid':>7}{'topdim':>8}")
    L = r.get("layers") or {}
    rows = [(k, L[k]) for k in sorted(L, key=int)] + ([("out", r["out"])] if r.get("out") else [])
    for k, d in rows:
        print(f"    {k:>6}{_f(d.get('depth'), 7, 2)}{_f(d['lid_probe'], 9)}{_f(d['lid_centroid'])}"
              f"{_f(d['lid_centroid_latn'])}{_f(d['p1'], 7)}{_f(d['p1_langcentered'])}"
              f"{_f(d['var']['content'], 9)}{_f(d['var']['language'], 9)}"
              f"{_f(d['var']['residual'], 7)}{_f(d['top_dim_frac'], 8)}")
    dp, p1 = profile(r, "p1")
    _, lc = profile(r, "lid_centroid")
    _, co = profile(r, "var", "content")
    _, la = profile(r, "var", "language")
    x8, x9 = crossover(dp, p1, lc), crossover(dp, co, la)
    print(f"    crossover  P@1 over LID-centroid: "
          + ("never" if x8 is None else f"depth {x8:.2f}")
          + "   content over language variance: "
          + ("never" if x9 is None else f"depth {x9:.2f}"))
    return x8, x9


def report_paired(M):
    """byte vs subword at matched size, read at matched FRACTIONAL depth -- the architectures have
    different block counts (8/12/18/24), so layer k means nothing across them; depth k/N does."""
    print("\n  BYTE - SUBWORD at matched size and fractional depth (interpolated along each profile)")
    cols = (("LID-cent(Latn)", "lid_centroid_latn", None), ("P@1", "p1", None),
            ("language var", "var", "language"), ("content var", "var", "content"))
    for size in ("small", "base", "large"):
        b, s = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        if not (b and s and b.get("layers") and s.get("layers")):
            continue
        print(f"\n    {size}: " + "".join(f"{c:>17}" for c, _, _ in cols))
        for dep in DEPTHS:
            cells = ""
            for _, key, sub in cols:
                db, vb = profile(b, key, sub)
                ds, vs = profile(s, key, sub)
                if None in vb or None in vs:
                    cells += f"{'-':>17}"
                    continue
                x, y = float(np.interp(dep, db, vb)), float(np.interp(dep, ds, vs))
                cells += f"{x:>6.3f}-{y:.3f}={x - y:>+.3f}".rjust(17)
            print(f"    {'d=' + f'{dep:.2f}':>6}{cells}")
        cells = ""
        for _, key, sub in cols:
            gb = b["out"][key][sub] if sub else b["out"][key]
            gs = s["out"][key][sub] if sub else s["out"][key]
            cells += (f"{gb:>6.3f}-{gs:.3f}={gb - gs:>+.3f}".rjust(17)
                      if gb is not None and gs is not None else f"{'-':>17}")
        print(f"    {'out':>6}{cells}")


def merge():
    M = merge_parts(ANALYSIS, SCHEMA)["models"]
    if not M:
        print("no layers part files yet")
        return
    M = {m: M[m] for m in MAIN_MODELS if m in M}
    print("\n=== EXP 8 + 9: where each architecture becomes multilingual ===")
    print("  LIDcent / (Latn): nearest-centroid language accuracy, all 10 / Latin-script 5 only")
    print("  P@1 / (lctr): translation retrieval, globally centred / each language's mean removed")
    print("  content / language / resid: standardized two-way variance decomposition (sums to 1)")
    print("  topdim: variance share of the largest coordinate -- high means raw-space numbers at")
    print("          that layer are about one massive-activation dimension")
    cx = {n: report_model(n, r) for n, r in M.items()}
    report_paired(M)
    print("\n  CROSSOVER DEPTHS (fraction of the encoder; earlier = multilingual sooner)")
    print(f"    {'model':16}{'P@1 > LIDcent':>16}{'content > lang':>16}")
    for n, (x8, x9) in cx.items():
        print(f"    {n:16}{_f(x8, 16, 2)}{_f(x9, 16, 2)}")
    print(f"\nmerged -> {merged_path(ANALYSIS)}")


# ----------------------------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    langs = ["te", "bn", "en", "sw", "yo"]         # three Latin (en sw yo), two not
    n, d = 200, 32
    train, test = sentence_split(n, 0)
    assert not set(train) & set(test) and len(train) + len(test) == n, "split must partition"

    def make(content_w, lang_w, noise=0.3):
        c = rng.normal(size=(n, 1, d))                       # shared by every translation
        g = rng.normal(size=(1, len(langs), d)) * 3          # one offset per language
        return content_w * c + lang_w * g + noise * rng.normal(size=(n, len(langs), d))

    # SAME content strength in both; only the language offset differs. That isolates the one thing
    # language-centring claims to do -- remove the offset and expose the content underneath -- from
    # how strong the content signal is, which a weaker-content fixture would conflate with it.
    lang_heavy, content_heavy = make(1.0, 1.0), make(1.0, 0.05)

    # the crossover's two sides move in OPPOSITE directions as the space reorganizes
    lh = analyse(lang_heavy, train, test, langs)
    ch = analyse(content_heavy, train, test, langs)
    assert lh["lid_centroid"] > 0.95 and lh["p1"] < ch["p1"], (lh, ch)
    assert ch["p1"] > 0.9 and ch["lid_centroid"] < lh["lid_centroid"], (lh, ch)
    # removing each language's mean must RESCUE alignment, back to what the same content gives
    # with no offset at all
    assert lh["p1_langcentered"] > lh["p1"] + 0.1, lh
    assert abs(lh["p1_langcentered"] - ch["p1"]) < 0.1, (lh["p1_langcentered"], ch["p1"])

    # the decomposition is exact and reads the planted structure
    for X in (lang_heavy, content_heavy):
        for std in (True, False):
            v = var_decomp(X, standardize=std)
            assert abs(sum(v.values()) - 1.0) < 1e-9, f"fractions must sum to 1: {v}"
    assert var_decomp(lang_heavy)["language"] > 0.7, var_decomp(lang_heavy)
    assert var_decomp(content_heavy)["content"] > 0.7, var_decomp(content_heavy)

    # standardizing is what stops one massive coordinate from owning the decomposition
    massive = content_heavy.copy()
    massive[:, :, 0] += rng.normal(size=(1, len(langs))) * 1e5        # a language-driven 1e5 dim
    assert var_decomp(massive, standardize=False)["language"] > 0.99, "raw is owned by the dim"
    assert var_decomp(massive, standardize=True)["content"] > 0.6, "standardized must not be"
    assert top_dim_frac(massive) > 0.99 and top_dim_frac(content_heavy) < 0.2

    # the Latin-only centroid measure uses only Latin languages
    assert lh["lid_centroid_latn"] is not None
    assert analyse(lang_heavy[:, :2], train, test, ["te", "bn"])["lid_centroid_latn"] is None

    # crossover arithmetic
    assert abs(crossover([0, 1], [0.0, 1.0], [0.5, 0.5]) - 0.5) < 1e-9
    assert crossover([0, 1], [0.0, 0.1], [0.5, 0.5]) is None, "never crossing must say so"
    assert crossover([0, 1], [0.9, 1.0], [0.5, 0.5]) == 0.0, "starting above crosses at once"
    print("selftest OK: sentence-grouped split, crossover readouts move oppositely, language "
          "centring rescues alignment, exact two-way decomposition, standardization defeats a "
          "massive coordinate, Latin-only subset, crossover arithmetic")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--split", default=SPLIT, help="FLORES split spec (default devtest, 1012 rows)")
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
        run_one(n, a.results, a.ckpt_dir, a.device, a.seed, a.split)


if __name__ == "__main__":
    main()
