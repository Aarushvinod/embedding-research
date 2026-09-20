"""Experiment 7 -- the English pivot: is non-English alignment routed through English?

When a student matches a Telugu sentence to its Bengali translation, are the directions carrying
that alignment the SAME directions that carry Telugu<->English alignment? If so English is a hub:
the space is organized around English and te<->bn holds only as a side effect of each being aligned
to English. The thesis predicts the byte student is less English-pivoted than its subword twin.

The measurement is a subspace removal. Fit CCA between a language and a MEDIATOR on a held-out half
of FLORES, project the top-k canonical directions out of BOTH sides of a pair, and re-score P@1:

    mediation(A,B|C) = P@1(A->B) - P@1(A->B after removing each side's C-predictable subspace)

Taken alone that number is worthless, because English-predictable is very nearly MEANING -- two
translations share their content with the English translation too, so removing the English-shared
subspace removes semantics and retrieval falls for an entirely uninteresting reason. Every mediator
would show a drop. The control IS the experiment: run the identical procedure with every other
language as the mediator, each of which also removes meaning by the same mechanism, and ask whether
English is anomalous against that distribution:

    excess(A,B|F) = mediation(A,B|F) - mean over C not in {A,B,F} of mediation(A,B|C)

Baseline retrieval quality and generic meaning-removal both cancel. excess ~ 0 says the focal
language is just another language; excess > 0 says it is a pivot.

Three design points decide whether the number means anything:

  * `--focal` runs the whole thing with zh and ar in English's place. A byte/subword gap on English
    ALONE is English-specific and supports the thesis; a gap on all three is a high-resource-mediator
    effect, which is a different and much weaker claim. This control determines the reading.
  * Removals are nested over K_GRID and the variance each one destroys is recorded, so mediators can
    be compared at EQUAL variance removed as well as at equal rank. English is the best-represented
    language and plausibly has higher canonical correlations with everything, which would make its
    k directions carry more than another mediator's k -- a strength effect posing as a pivot.
  * Everything is read out in the real 1024-d embedding space (the PCA basis is a fitting device,
    not a measurement space), so `base` is an actual retrieval number comparable to the rest of the
    paper. Every student projects to the teacher's 1024 dims, so no arm has a width advantage.

FLORES is used because the design needs strictly n-way PARALLEL rows -- row k must be the same
sentence in A, B and C simultaneously. Wikipedia (the training corpus) is monolingual and OPUS-100
is English-centric bitext with a different English side per pair, so neither can serve here.

This is OBSERVATIONAL: it shows the shared subspace CARRIES the alignment, not that the model
computes alignment through English. Exp 6 (interp_heads) is the causal counterpart.

  python -m byte_embed.interp_pivot --selftest        # planted-structure validation, no GPU
  python -m byte_embed.interp_pivot --only byte-small
  python -m byte_embed.interp_pivot --merge
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np

from byte_embed.interp_common import (MAIN_MODELS, flores_parallel, make_encode, merge_parts,
                                      merged_path, models_in, part_path, read_json, utf8_stdout,
                                      write_json)
from byte_embed.interp_script import TEACHER, load_for
from common.eval import l2norm

ANALYSIS = "pivot"
SCHEMA = 1
K0 = 128                          # PCA dims per language before CCA (see cca_ok for why)
K_GRID = (4, 8, 16, 32, 64)       # nested canonical-subspace removals
FOCAL = ("en", "zh", "ar")        # en is the hypothesis; zh/ar decide whether it is English-specific
MIN_MEDIATION = 0.02              # pre-registered floor: below this the intervention is not biting
SPLIT = "dev+devtest"             # 2009 parallel rows, the whole public set
PIVOT_MODELS = MAIN_MODELS + [TEACHER]


# ----------------------------------------------------------------------------------------------
# linear algebra
# ----------------------------------------------------------------------------------------------
def split_rows(n, seed=0):
    """(fit, test) row indices. One fixed seed for every model, pair and mediator, so an arm re-run
    months later is scored on exactly the sentences the others were."""
    idx = np.random.default_rng(seed).permutation(n)
    h = n // 2
    return np.sort(idx[:h]), np.sort(idx[h:])


def pca_basis(X, k):
    """(mu, V[d,k]) -- top-k principal directions, FIT HALF ONLY.

    Fitting this on the test half would let the removal see the rows it is scored on."""
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, np.ascontiguousarray(Vt[:k].T)


def cca(X, Y, tol=1e-8):
    """Classical CCA by SVD: (A[p,r], rho[r]), A's columns the X-side canonical directions in X's
    own coordinates, ordered by canonical correlation.

    Route: Xc = Ux diag(sx) Vtx, so Xc @ (Vtx.T / sx) = Ux is the whitened score matrix and the SVD
    of Ux.T @ Uy gives the correlations directly -- no explicit inverse of a covariance. Components
    under `tol` of the leading singular value are dropped, which is what stops a rank-deficient half
    from manufacturing unit correlations out of numerical dust.

    The canonical directions are orthogonal in the WHITENED metric, not the Euclidean one, so they
    are not a projector as they stand; nested_basis fixes that."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Ux, sx, Vtx = np.linalg.svd(Xc, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Yc, full_matrices=False)
    rx = int((sx > tol * sx[0]).sum()) or 1
    ry = int((sy > tol * sy[0]).sum()) or 1
    Ux, sx, Vtx = Ux[:, :rx], sx[:rx], Vtx[:rx]
    P, rho, _ = np.linalg.svd(Ux.T @ Uy[:, :ry], full_matrices=False)
    return Vtx.T @ (P / sx[:, None]), np.clip(rho, 0.0, 1.0)


def cca_ok(n_fit, k0):
    """CCA at d dims with n ~ d rows has as many free parameters as observations and returns
    canonical correlations near 1 for random vectors -- it would 'find' an English-shared subspace
    in noise. k0 = 128 against ~1000 fit rows is the ratio that keeps that from happening, and it is
    identical for every arm so it introduces no asymmetry between architectures."""
    return n_fit >= 4 * k0


def nested_basis(A, V):
    """Canonical directions -> a NESTED orthonormal basis in the full embedding space.

    A's columns live in PCA coordinates; V maps them back to d dims. QR in canonical order yields
    orthonormal columns whose first k span exactly the top-k canonical subspace for EVERY k, which
    makes the K_GRID sweep one nested sequence of removals rather than five unrelated ones -- so the
    k-curve reads as 'more of the same subspace', not 'a different subspace each time'."""
    Q, R = np.linalg.qr(V @ A)
    rank = int((np.abs(np.diag(R)) > 1e-10 * max(abs(R[0, 0]), 1e-30)).sum())
    return np.ascontiguousarray(Q[:, :rank], dtype=np.float32)


def remove(Z, Q, k):
    """Z with its top-k canonical subspace projected out and re-normalized."""
    k = min(int(k), Q.shape[1])
    if k <= 0:
        return l2norm(Z)
    B = Q[:, :k]
    return l2norm(Z - (Z @ B) @ B.T)


def var_removed(Z, Q, k):
    """Fraction of squared norm the top-k removal destroys -- the x-axis of the variance-matched
    comparison, and the direct check on 'English simply removes more'."""
    k = min(int(k), Q.shape[1])
    if k <= 0:
        return 0.0
    P = Z @ Q[:, :k]
    return float((P * P).sum() / max((Z * Z).sum(), 1e-12))


def p_at_1(Za, Zb):
    """Index-aligned P@1: row i of Za must retrieve row i of Zb. Both L2-normalized."""
    return float((np.argmax(Za @ Zb.T, axis=1) == np.arange(len(Za))).mean())


def p_at_1_both(Za, Zb):
    """(A->B, B->A) from ONE similarity matrix -- the matmul is the cost, the argmax is free."""
    S = Za @ Zb.T
    ar = np.arange(len(Za))
    return float((np.argmax(S, axis=1) == ar).mean()), float((np.argmax(S, axis=0) == ar).mean())


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------
def embed_all(enc, par, ckpt_dir, name, split, langs):
    """{lang: [n, d]} FLORES embeddings, cached under checkpoints/.

    Encoding 10 x ~2000 sentences is the only slow stage here -- everything below is linear algebra
    on the result -- so a preempted job that re-encoded would pay the whole cost again for nothing.
    The key carries the split and the row counts are re-checked, so a changed spec cannot silently
    reuse the old matrix."""
    cp = Path(ckpt_dir) / f"pivot_emb_{name}_{split}.npz"
    if cp.exists():
        try:
            z = np.load(cp)
            if set(langs) <= set(z.files) and all(len(z[l]) == len(par[l]) for l in langs):
                print(f"  [pivot] cached embeddings <- {cp.name}")
                return {l: np.ascontiguousarray(z[l], dtype=np.float32) for l in langs}
            print(f"  [pivot] {cp.name} does not match this split/langs -> re-encoding")
        except (OSError, ValueError):
            print(f"  [pivot] unreadable {cp.name} (preempted write?) -> re-encoding")
    out = {}
    for l in langs:
        out[l] = np.ascontiguousarray(enc(par[l]), dtype=np.float32)
        print(f"  [pivot] encoded {l}: {out[l].shape}")
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, **out)
    return out


def run_one(name, results, ckpt_dir, device, seed=0, split=SPLIT, k0=K0, langs=None):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:
        res = {"schema": SCHEMA}
    par = flores_parallel(cache_dir=ckpt_dir, split=split)
    langs = list(langs or par)
    bad = [l for l in langs if l not in par]
    if bad:
        raise SystemExit(f"unknown langs {bad}; choose from {list(par)}")
    # Re-run when any part of the spec changed: a documented fast path (--langs en,te --k0 32) must
    # not leave a two-language table on disk that the merge presents as the full 10x10.
    want = {"split": split, "k0": int(k0), "seed": int(seed), "langs": sorted(langs),
            "k_grid": list(K_GRID)}
    if res.get("med") and all(res.get(k) == v for k, v in want.items()):
        print(f"  [pivot] {name}: done under this spec -> skip")
        return
    loaded = load_for(name, results, ckpt_dir, device)
    if loaded is None:
        return
    model, bm = loaded
    enc = make_encode(model, device, batch_size=(32 if bm.get("kind") == "byte" else 128))
    Z = embed_all(enc, par, ckpt_dir, name, split, langs)
    del model, enc                       # the rest is numpy; drop the GPU copy

    n = len(Z[langs[0]])
    fit, test = split_rows(n, seed)
    if not cca_ok(len(fit), k0):
        print(f"  [pivot] {name}: {len(fit)} fit rows is too few for k0={k0} (want >= {4 * k0}); "
              f"CCA would overfit and report a pivot in noise -> skipping")
        return
    res.update(model=name, kind=bm.get("kind"), steps_run=bm.get("steps_run"),
               n_fit=int(len(fit)), n_test=int(len(test)), d=int(Z[langs[0]].shape[1]), **want)

    # ---- per-language PCA (fit half only), then one CCA per (language, mediator)
    bas = {l: pca_basis(Z[l][fit], k0) for l in langs}
    Q, rho = {}, {}
    for l in langs:
        mu_l, V_l = bas[l]
        Xl = (Z[l][fit] - mu_l) @ V_l
        for c in langs:
            if c == l:
                continue
            mu_c, V_c = bas[c]
            A, r = cca(Xl, (Z[c][fit] - mu_c) @ V_c)
            Q[(l, c)] = nested_basis(A, V_l)
            rho[(l, c)] = r
    print(f"  [pivot] {len(Q)} CCA fits at k0={k0} on {len(fit)} rows; scoring {len(test)} rows")

    Zt = {l: l2norm(Z[l][test]) for l in langs}
    res["rho"] = {l: {c: [round(float(x), 5) for x in rho[(l, c)][:k0]]
                      for c in langs if c != l} for l in langs}
    res["var"] = {l: {c: {str(k): round(var_removed(Zt[l], Q[(l, c)], k), 5) for k in K_GRID}
                      for c in langs if c != l} for l in langs}

    # ---- baselines, then every (pair, mediator, k) cell
    base = {a: {} for a in langs}
    for i, a in enumerate(langs):
        for b in langs[i + 1:]:
            ab, ba = p_at_1_both(Zt[a], Zt[b])
            base[a][b], base[b][a] = round(ab, 5), round(ba, 5)
    res["base"] = base

    med = {a: {b: {} for b in langs if b != a} for a in langs}
    for k in K_GRID:
        for c in langs:
            # one removal per language per (mediator, k), reused across every partner
            Zr = {l: remove(Zt[l], Q[(l, c)], k) for l in langs if l != c}
            others = [l for l in langs if l != c]
            for i, a in enumerate(others):
                for b in others[i + 1:]:
                    ab, ba = p_at_1_both(Zr[a], Zr[b])
                    med[a][b].setdefault(c, {})[str(k)] = round(base[a][b] - ab, 5)
                    med[b][a].setdefault(c, {})[str(k)] = round(base[b][a] - ba, 5)
        cells = [v[str(k)] for a in med for b in med[a] for v in med[a][b].values()]
        print(f"    k={k}: mean mediation over all (pair, mediator) {np.mean(cells):.4f}")
    res["med"] = med
    write_json(outp, res)
    print(f"  saved -> {outp}")


# ----------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------
def _boot(vals, n=2000, seed=0):
    """Percentile CI over PAIRS. Pair-to-pair heterogeneity dominates the uncertainty here -- the
    within-pair P@1 sampling error is second order at ~1000 scored rows -- so pairs are what get
    resampled."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 3:
        return None, None
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _sign(diffs):
    """Two-sided exact sign test on paired differences: (p, n_nonzero). Paired across the SAME pairs
    for two arms, so it assumes nothing about the distribution of the excess itself."""
    d = np.asarray([x for x in diffs if x is not None], float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return None, 0
    k = min(int((d > 0).sum()), int((d < 0).sum()))
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n), n


def pairs_for(med, focal):
    """Ordered (A,B) with A,B != focal. When the focal IS one of the pair, 'mediation by the focal'
    is trivially total and the cell says nothing about pivoting."""
    return [(a, b) for a in med for b in med[a] if focal not in (a, b)]


def excess(med, focal, k, pairs=None):
    """[excess(A,B|focal)] over pairs: the focal's mediation minus the mean of every other admissible
    mediator's. Removing ANY language's predictable component removes meaning, so the other
    mediators are the only thing that makes the focal's drop interpretable."""
    out = []
    for a, b in (pairs if pairs is not None else pairs_for(med, focal)):
        cells = med[a][b]
        f = (cells.get(focal) or {}).get(str(k))
        rest = [v[str(k)] for c, v in cells.items() if c != focal and str(k) in v]
        out.append(None if f is None or not rest else f - float(np.mean(rest)))
    return out


def med_at(med, focal, k):
    """[mediation(A,B|focal)] over the pairs `excess` uses -- the raw drop, before the control."""
    return [(med[a][b].get(focal) or {}).get(str(k)) for a, b in pairs_for(med, focal)]


def _interp(xs, ys, target):
    """y at a target x by linear interpolation INSIDE the measured range; None outside it, so a
    mediator that never destroys that much variance is dropped rather than extrapolated."""
    o = np.argsort(xs)
    x, y = np.asarray(xs, float)[o], np.asarray(ys, float)[o]
    if target < x[0] or target > x[-1]:
        return None
    return float(np.interp(target, x, y))


def excess_varmatched(res, focal, target):
    """excess at EQUAL VARIANCE REMOVED rather than equal rank.

    English is the best-represented language and plausibly has higher canonical correlations with
    everything, so its k directions can carry more than another mediator's k and produce a larger
    drop with no pivot involved. Comparing mediators at the same fraction of destroyed variance
    removes that confound. Each (language, mediator) has its own k -> variance curve, so the match is
    per side and interpolated, and a pair is dropped when either side cannot reach the target."""
    med, var = res["med"], res["var"]
    out = []
    for a, b in pairs_for(med, focal):
        vals = {}
        for c, cells in med[a][b].items():
            ks = [k for k in res["k_grid"] if str(k) in cells]
            # the removal hit BOTH sides, so the variance it cost this pair is the mean of the two
            xs = [0.5 * (var[a][c][str(k)] + var[b][c][str(k)]) for k in ks]
            vals[c] = _interp(xs, [cells[str(k)] for k in ks], target)
        f = vals.get(focal)
        rest = [v for c, v in vals.items() if c != focal and v is not None]
        out.append(None if f is None or not rest else f - float(np.mean(rest)))
    return out


def _fmt(v, w=8, p=4):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"


def remaining_at(res, k):
    """Mean P@1 SURVIVING the removal, over all (pair, mediator) cells. The guard against the other
    failure mode: if removal craters retrieval toward chance, every mediator saturates and excess
    goes to 0 by floor effect, which is not evidence that the focal is not a pivot."""
    med, base = res.get("med") or {}, res.get("base") or {}
    v = [base[a][b] - c[str(k)] for a in med for b in med[a]
         for c in med[a][b].values() if str(k) in c]
    return float(np.mean(v)) if v else None


def report_bite(M, k_grid):
    """Does the intervention do anything at all, and does it leave anything standing? Pre-registered:
    under MIN_MEDIATION it is not biting, and near chance it has saturated. Either way there is
    nothing to interpret in the excess tables below, and the k to report is one that does neither."""
    print("\n(1) Does the removal bite? mean mediation over ALL (pair, mediator) cells")
    print(f"  {'model':16}" + "".join(f"{'k=' + str(k):>9}" for k in k_grid)
          + f"{'P@1 left':>10}{'chance':>9}   verdict")
    for name, r in M.items():
        med = r.get("med") or {}
        row, last = [], None
        for k in k_grid:
            v = [c[str(k)] for a in med for b in med[a] for c in med[a][b].values() if str(k) in c]
            last = float(np.mean(v)) if v else None
            row.append(last)
        left = remaining_at(r, k_grid[-1])
        chance = 1.0 / max(r.get("n_test") or 1, 1)
        if last is None or last < MIN_MEDIATION:
            verdict = f"BELOW {MIN_MEDIATION} - null"
        elif left is not None and left <= 4 * chance:
            verdict = "SATURATED at the top k - read a lower k"
        else:
            verdict = "bites"
        print(f"  {name:16}" + "".join(_fmt(v, 9) for v in row)
              + _fmt(left, 10, 3) + _fmt(chance, 9, 4) + f"   {verdict}")


def report_excess(M, k, focals=FOCAL):
    """The headline. excess > 0 means the focal is an anomalously load-bearing mediator."""
    print(f"\n(2) excess(A,B|focal) at k={k}  [focal mediation - mean of the other mediators]")
    print(f"  {'model':16}" + "".join(f"{f:>26}" for f in focals))
    print(f"  {'':16}" + "".join(f"{'excess':>10}{'95% CI':>16}" for _ in focals))
    for name, r in M.items():
        med = r.get("med") or {}
        cells = ""
        for f in focals:
            e = [x for x in excess(med, f, k) if x is not None] if med else []
            lo, hi = _boot(e)
            ci = "-" if lo is None else f"[{lo:+.3f},{hi:+.3f}]"
            cells += _fmt(float(np.mean(e)) if e else None, 10) + f"{ci:>16}"
        print(f"  {name:16}{cells}")
    print("  excess ~ 0: the focal is just another language.  > 0: it is a pivot.")


def report_paired(M, k, focals=FOCAL):
    """byte vs subword at matched size, paired on the SAME (A,B) pairs -- the actual thesis test."""
    print(f"\n(3) byte - subword at k={k}, paired over pairs (negative = byte LESS pivoted)")
    print(f"  {'size':10}{'focal':>7}{'byte':>9}{'subword':>9}{'diff':>9}{'sign p':>9}{'n':>5}")
    for size in ("small", "base", "large"):
        bm, sm = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        if not (bm and sm and bm.get("med") and sm.get("med")):
            continue
        for f in focals:
            pr = pairs_for(bm["med"], f)
            eb, es = excess(bm["med"], f, k, pr), excess(sm["med"], f, k, pr)
            d = [x - y for x, y in zip(eb, es) if x is not None and y is not None]
            p, n = _sign(d)
            mb = [x for x in eb if x is not None]
            ms = [x for x in es if x is not None]
            print(f"  {size:10}{f:>7}{_fmt(float(np.mean(mb)) if mb else None, 9)}"
                  f"{_fmt(float(np.mean(ms)) if ms else None, 9)}"
                  f"{_fmt(float(np.mean(d)) if d else None, 9)}"
                  f"{_fmt(p, 9, 3)}{n:>5}")


def reachable_target(M):
    """A variance-removed level EVERY (language, mediator) curve in every arm actually spans.

    Each curve runs from var(k_min) to var(k_max); `_interp` refuses to extrapolate outside that,
    so a hand-picked target below the highest start or above the lowest end silently empties the
    table. The midpoint of the common interval is reachable by construction, and it has to be shared
    across arms or the byte and subword columns are read at different levels."""
    lo, hi = 0.0, 1.0
    for r in M.values():
        var, grid = r.get("var") or {}, r.get("k_grid") or list(K_GRID)
        for per in var.values():
            for v in per.values():
                lo = max(lo, v[str(grid[0])])
                hi = min(hi, v[str(grid[-1])])
    return None if lo >= hi else 0.5 * (lo + hi)


def report_varmatched(M, target=None, focals=FOCAL):
    """(2) again with rank matching replaced by variance matching. A result that survives only one
    of the two matchings is a strength effect, not a pivot, and this is where that shows."""
    auto = target is None
    target = reachable_target(M) if auto else target
    if target is None:
        print("\n(4) variance-matched: the arms' k-grids do not overlap in variance removed -> "
              "no common level exists; widen K_GRID")
        return
    print(f"\n(4) excess at EQUAL variance removed ({target:.1%}"
          f"{', auto-chosen as reachable by every curve' if auto else ''}) "
          f"-- guards 'English removes more'")
    print(f"  {'model':16}" + "".join(f"{f:>12}" for f in focals) + f"{'n pairs':>9}")
    for name, r in M.items():
        if not r.get("med"):
            continue
        cells, n_ok = "", 0
        for f in focals:
            e = [x for x in excess_varmatched(r, f, target) if x is not None]
            n_ok = max(n_ok, len(e))
            cells += _fmt(float(np.mean(e)) if e else None, 12)
        print(f"  {name:16}{cells}{n_ok:>9}")
    print("  compare against (2): an excess that holds at equal RANK but vanishes at equal VARIANCE")
    print("  is English removing more, not English being a pivot.")


def report_diag(M, k, focals=FOCAL):
    """The confounds worth seeing beside the headline rather than buried: how well each arm does
    English at all (a better English encoder mediates more for reasons unrelated to pivoting), how
    strongly each focal correlates with the rest, and how much variance its removal costs."""
    print(f"\n(5) diagnostics at k={k}")
    print(f"  {'model':16}{'base P@1':>10}{'en P@1':>9}" +
          "".join(f"{'rho[' + f + ']':>11}{'var[' + f + ']':>11}" for f in focals))
    for name, r in M.items():
        rho, var, base = r.get("rho") or {}, r.get("var") or {}, r.get("base") or {}
        allp = [v for a in base for v in base[a].values()]
        enp = [v for a in base for b, v in base[a].items() if "en" in (a, b)]
        cells = ""
        for f in focals:
            rr = [float(np.mean(rho[l][f][:k])) for l in rho if f in rho[l]]
            vv = [var[l][f][str(k)] for l in var if f in var[l] and str(k) in var[l][f]]
            cells += _fmt(float(np.mean(rr)) if rr else None, 11, 3)
            cells += _fmt(float(np.mean(vv)) if vv else None, 11, 3)
        print(f"  {name:16}{_fmt(float(np.mean(allp)) if allp else None, 10, 3)}"
              f"{_fmt(float(np.mean(enp)) if enp else None, 9, 3)}{cells}")
    print("  a large excess on a focal whose rho and var match the others is a real pivot;")
    print("  one that tracks rho/var is a representation-STRENGTH effect and reads as such.")


def merge(k=None, target=None, focals=FOCAL):
    M = merge_parts(ANALYSIS, SCHEMA)["models"]
    if not M:
        print("no pivot part files yet")
        return
    order = [m for m in PIVOT_MODELS if m in M] + [m for m in M if m not in PIVOT_MODELS]
    M = {m: M[m] for m in order}
    any_r = next(iter(M.values()))
    grid = any_r.get("k_grid") or list(K_GRID)
    k = k or grid[len(grid) // 2]
    print(f"\n=== EXP 7: the English pivot ({len(M)} arms, k0={any_r.get('k0')}, "
          f"{any_r.get('n_fit')} fit / {any_r.get('n_test')} scored rows, "
          f"split {any_r.get('split')}) ===")
    report_bite(M, grid)
    report_excess(M, k, focals)
    report_paired(M, k, focals)
    report_varmatched(M, target, focals)
    report_diag(M, k, focals)
    print(f"\nmerged -> {merged_path(ANALYSIS)}")


# ----------------------------------------------------------------------------------------------
def _selftest():
    """Plant a known structure and check the measurement recovers it.

    Languages are built the way distilled students actually are -- every language maps into ONE
    shared frame (the teacher's), which is why their embeddings are mutually comparable at all.
    Independent random projections of the same latent are NOT aligned and retrieve at chance, so a
    construction without a shared frame tests nothing.

    Two arms over a content channel `c` that English also expresses:
      thru  A and B share ONLY c            -> removing the English-shared subspace should gut A->B
      priv  A and B also share `q`, a channel English never sees -> A->B should largely survive
    If `priv` dropped as hard as `thru`, the removal would be deleting generic meaning rather than
    the English-shared component, and every number this module prints would be an artifact."""
    rng = np.random.default_rng(0)
    n, m, d, npriv = 800, 24, 64, 8
    W = lambda k: rng.normal(size=(k, d)) / np.sqrt(k)          # noqa: E731
    nz = lambda s: s * rng.normal(size=(n, d))                  # noqa: E731
    c, q = rng.normal(size=(n, m)), rng.normal(size=(n, npriv))
    t, u = c @ W(m), q @ W(npriv)      # shared content, and the A/B-private channel
    E = l2norm(t + nz(0.5))
    arms = {"thru": (l2norm(t + nz(0.5)), l2norm(t + nz(0.5))),
            "priv": (l2norm(t + u + nz(0.5)), l2norm(t + u + nz(0.5)))}
    fit, test = split_rows(n, 0)
    k0, k = 32, 24                     # k0 > m + npriv so PCA keeps both channels; k ~ rank of `t`
    ok = True

    assert abs(p_at_1(E[test], E[test]) - 1.0) < 1e-9, "P@1 against itself must be 1"
    ab, ba = p_at_1_both(arms["thru"][0][test], arms["thru"][1][test])
    assert abs(ab - p_at_1(arms["thru"][0][test], arms["thru"][1][test])) < 1e-9, "both: A->B"
    assert abs(ba - p_at_1(arms["thru"][1][test], arms["thru"][0][test])) < 1e-9, "both: B->A"

    # CCA recovers a planted rotation: Y a rotation of X's first r dims -> rho ~ 1 for r of them
    r = 6
    X0 = rng.normal(size=(n, 12))
    Y0 = np.c_[X0[:, :r] @ np.linalg.qr(rng.normal(size=(r, r)))[0], rng.normal(size=(n, 12 - r))]
    _, rr = cca(X0, Y0)
    assert (rr[:r] > 0.99).all() and rr[r] < 0.5, f"CCA rho wrong: {np.round(rr, 3)}"

    mu_e, V_e = pca_basis(E[fit], k0)
    Ef = (E[fit] - mu_e) @ V_e
    drops = {}
    for tag, (A, B) in arms.items():
        qs = []
        for Zx in (A, B):
            mu, V = pca_basis(Zx[fit], k0)
            Ax, _ = cca((Zx[fit] - mu) @ V, Ef)
            Q = nested_basis(Ax, V)
            assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-4), "basis not orthonormal"
            # NESTING: the first 4 columns must span the same subspace as a basis built from the
            # top 4 canonical directions alone -- otherwise the K_GRID sweep is five unrelated
            # removals and its curve cannot be read as "more of the same subspace".
            q4 = nested_basis(Ax[:, :4], V)
            assert np.allclose(Q[:, :4] @ Q[:, :4].T, q4 @ q4.T, atol=1e-4), "basis not nested"
            qs.append(Q)
        b = p_at_1(A[test], B[test])
        a = p_at_1(remove(l2norm(A[test]), qs[0], k), remove(l2norm(B[test]), qs[1], k))
        drops[tag] = b - a
        print(f"  {tag}: P@1 {b:.3f} -> {a:.3f}   mediation {b - a:+.3f}   "
              f"var removed {var_removed(l2norm(A[test]), qs[0], k):.3f}")
    # a random subspace of the same rank is the null: it removes variance but not the shared channel
    rnd = [remove(l2norm(Zx[test]), np.linalg.qr(rng.normal(size=(d, k)))[0].astype(np.float32), k)
           for Zx in arms["thru"]]
    d_rand = p_at_1(arms["thru"][0][test], arms["thru"][1][test]) - p_at_1(rnd[0], rnd[1])
    print(f"  random subspace of rank {k}: mediation {d_rand:+.3f}")

    for cond, msg in ((drops["thru"] > 0.10, "English-mediated arm barely dropped"),
                      (drops["thru"] > drops["priv"] + 0.05, "private channel dropped as hard"),
                      (drops["thru"] > d_rand + 0.05, "no better than a random subspace"),
                      (not cca_ok(64, 128), "cca_ok must reject 64 rows at k0=128"),
                      (cca_ok(1004, 128), "cca_ok must accept 1004 rows at k0=128")):
        if not cond:
            print(f"  FAIL: {msg}")
            ok = False

    # excess arithmetic: a focal that mediates no more than the others must come out at ~0
    med = {"a": {"b": {"en": {"8": 0.30}, "sw": {"8": 0.10}, "te": {"8": 0.20}}},
           "b": {"a": {"en": {"8": 0.30}, "sw": {"8": 0.10}, "te": {"8": 0.20}}}}
    assert all(abs(x - 0.15) < 1e-9 for x in excess(med, "en", 8)), "excess arithmetic wrong"
    flat = {"a": {"b": {"en": {"8": 0.2}, "sw": {"8": 0.2}, "te": {"8": 0.2}}}}
    assert abs(excess(flat, "en", 8)[0]) < 1e-9, "a non-pivot focal must give excess 0"
    assert pairs_for(med, "a") == [], "pairs including the focal must be excluded"
    assert med_at(med, "en", 8) == [0.30, 0.30], "med_at wrong"
    assert _interp([0.1, 0.5], [1.0, 3.0], 0.3) == 2.0, "interp wrong"
    assert _interp([0.1, 0.5], [1.0, 3.0], 0.9) is None, "interp must not extrapolate"
    assert _sign([1, 1, 1, 1, 1, 1])[0] < 0.05, "sign test too weak"
    assert _sign([1, -1, 1, -1])[0] > 0.9, "sign test too strong"

    print("  SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--langs", default=None, help="comma list (default: all 10)")
    ap.add_argument("--split", default=SPLIT, help="FLORES split spec (default dev+devtest, 2009)")
    ap.add_argument("--k0", type=int, default=K0, help="PCA dims per language before CCA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None, help="--merge: rank to report (default: mid grid)")
    ap.add_argument("--var-target", type=float, default=None,
                    help="--merge: variance-removed level (default: auto, the level every curve reaches)")
    ap.add_argument("--focal", default=",".join(FOCAL),
                    help="--merge: focal mediators; en is the hypothesis, the rest are its control")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.merge:
        return merge(a.k, a.var_target, tuple(a.focal.split(",")))
    known = models_in(a.results)[0]
    names = [a.only] if a.only else [n for n in PIVOT_MODELS if n == TEACHER or n in known]
    langs = a.langs.split(",") if a.langs else None
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.seed, a.split, a.k0, langs)


if __name__ == "__main__":
    main()
