"""Paired bootstrap significance testing for byte-vs-subword (and boundary-arm) comparisons.

Both models in a comparison score the SAME queries on the SAME pools, so their per-query nDCG@10
vectors are naturally paired by query id. We resample query ids with replacement (the paired
bootstrap: Koehn 2004), which respects that pairing — the right test when the margin is small and
the two systems' per-query scores are correlated. A plain unpaired CI would be far too wide.

Per-query nDCG@10 is emitted by every scorer (miracl.eval_miracl, qa_retrieval._score_pool,
eval_mteb.eval_belebele) under the `per_query` key, so this reads finished results — no re-eval.

  python -m byte_embed.stats                 # byte vs subword at each size, from full_eval part files
  python -m byte_embed.stats --arms          # boundary B/C vs A (raw), per byte size
  python -m byte_embed.stats --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

_MIRACL_KEYS = ("miracl_full", "miracl")
_QA_KEYS = ("qa_full", "qa_retrieval")


def paired_bootstrap(a: dict, b: dict, n_boot=10000, seed=0, alpha=0.05):
    """a, b: {qid: score}. Returns the a-minus-b mean difference with a bootstrap CI + two-sided
    p-value over the shared query ids. `significant` = the (1-alpha) CI excludes 0. None if no
    shared queries."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())  # >1 when a point mass sits on 0
    p = min(1.0, max(float(p), 1.0 / n_boot))                # clamp to [1/n_boot, 1]; never exactly 0
    return {"delta": round(float(d.mean()), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p_value": round(p, 4), "n_paired": n, "significant": bool(lo > 0 or hi < 0)}


def iter_perquery(d: dict):
    """Yield ((benchmark, lang), {qid: score}) for every per_query block in a model-result or
    full_eval-part dict (handles both the pooled training results and the full-corpus final eval)."""
    for mk in _MIRACL_KEYS:
        mf = d.get(mk)
        if mf:
            for lang, m in (mf.get("per_lang") or {}).items():
                if m and m.get("per_query"):
                    yield ("MIRACL", lang), m["per_query"]
    for qk in _QA_KEYS:
        qa = d.get(qk)
        if isinstance(qa, dict):
            for bench, bd in qa.items():
                if isinstance(bd, dict):
                    for lang, m in (bd.get("per_lang") or {}).items():
                        if m and m.get("per_query"):
                            yield (bench, lang), m["per_query"]
    for extra in ("afriqa_100k",):
        af = d.get(extra)
        if af and (af.get("per_lang")):
            for lang, m in af["per_lang"].items():
                if m and m.get("per_query"):
                    yield (extra, lang), m["per_query"]
    bel = d.get("belebele")
    if isinstance(bel, dict):
        for lang, m in bel.items():
            if isinstance(m, dict) and m.get("per_query"):
                yield ("belebele", lang), m["per_query"]


def compare(better: dict, worse: dict, n_boot=10000):
    """Paired bootstrap of `better` minus `worse` over every shared (benchmark, lang) cell."""
    bmap, wmap = dict(iter_perquery(better)), dict(iter_perquery(worse))
    rows = []
    for key in sorted(set(bmap) & set(wmap)):
        r = paired_bootstrap(bmap[key], wmap[key], n_boot=n_boot)
        if r:
            rows.append((key, r))
    return rows


def _print_rows(title, rows):
    print(f"\n{title}")
    print(f"  {'benchmark':14}{'lang':6}{'Δ nDCG@10':>11}{'95% CI':>20}{'p':>9}{'n':>7}  sig")
    for (bench, lang), r in rows:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        star = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else \
               "*" if r["significant"] else ""
        print(f"  {bench:14}{lang:6}{r['delta']:>+11.4f}{ci:>20}{r['p_value']:>9.4f}"
              f"{r['n_paired']:>7}  {star}")


def _load_part(label, model):
    p = Path(f"results/full_eval_part_{label}_{model}.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def report_bytevssub(n_boot=10000):
    """byte-X minus subword-X at each size, from the full_eval part files."""
    any_found = False
    for size in ("small", "base", "large"):
        b, s = _load_part("main", f"byte-{size}"), _load_part("main", f"subword-{size}")
        if not (b and s):
            print(f"[byte-{size} vs subword-{size}] part file(s) missing — skip")
            continue
        any_found = True
        _print_rows(f"byte-{size} − subword-{size}  (Δ>0 ⇒ byte wins; * CI excludes 0)",
                    compare(b, s, n_boot))
    if not any_found:
        print("No full_eval part files found — run slurm/submit_full_eval.sh first.")


def report_arms(n_boot=10000):
    """Boundary arms B (teacher) and C (random) minus A (raw main) per byte size."""
    for size in ("small", "base", "large"):
        a = _load_part("main", f"byte-{size}")
        if not a:
            continue
        for arm, lab in (("teacher", "bteacher"), ("random", "brandom")):
            x = _load_part(lab, f"byte-{size}")
            if x:
                _print_rows(f"byte-{size}: {arm}-boundary − raw  (Δ>0 ⇒ markers help)",
                            compare(x, a, n_boot))


def _selftest():
    rng = np.random.default_rng(0)
    # identical-ish systems: byte slightly better on a correlated per-query signal
    base = rng.random(300)
    a = {f"q{i}": float(min(1.0, base[i] + 0.05 + 0.02 * rng.standard_normal())) for i in range(300)}
    b = {f"q{i}": float(base[i]) for i in range(300)}
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["delta"] > 0 and r["significant"] and r["ci_low"] > 0, r
    # no difference -> not significant
    c = {f"q{i}": float(base[i]) for i in range(300)}
    r2 = paired_bootstrap(c, b, n_boot=2000)
    assert not r2["significant"] and abs(r2["delta"]) < 1e-9, r2
    # disjoint ids -> None
    assert paired_bootstrap({"x": 1.0}, {"y": 1.0}) is None
    # degenerate: identical all-zero systems -> p clamped to <=1, not significant
    z = paired_bootstrap({f"q{i}": 0.0 for i in range(50)}, {f"q{i}": 0.0 for i in range(50)})
    assert z["p_value"] <= 1.0 and not z["significant"], z
    print(f"selftest OK: real Δ={r['delta']} CI[{r['ci_low']},{r['ci_high']}] p={r['p_value']} sig; "
          f"null Δ={r2['delta']} sig={r2['significant']}; degenerate p={z['p_value']} (<=1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", action="store_true", help="boundary B/C vs A instead of byte vs subword")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.arms:
        report_arms(a.n_boot)
    else:
        report_bytevssub(a.n_boot)


if __name__ == "__main__":
    main()
