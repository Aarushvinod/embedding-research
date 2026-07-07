"""Generate the curated PAPER figures (PNGs in figures/) from the run JSONs.

ByteEmbed numbers are read from results/byte_paper.json (run_paper.py). Run after the paper
experiment completes:
  python gen_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "savefig.bbox": "tight"})


def _load():
    p = Path("results/byte_paper.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _m(d, k):
    return (d.get(k) if d else None)


def fig_byte_objective(R):
    order = ["byte-cosine", "byte-contrastive", "byte-both", "subword-mt5-both", "byte-random"]
    cfgs = [(n, R["configs"][n]) for n in order if n in R.get("configs", {})]
    if not cfgs:
        return
    import numpy as np
    metrics = [("tatoeba_mean", "Tatoeba retrieval"), ("sib_mean", "SIB classification"),
               ("sts_mean", "STS")]
    x = np.arange(len(cfgs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (key, lab) in enumerate(metrics):
        ax.bar(x + (i - 1) * w, [c[1].get(key) or 0 for c in cfgs], w, label=lab)
    # teacher reference lines
    t = R["teacher"]
    for key, lab in [("tatoeba", "teacher Tatoeba"), ("sib", "teacher SIB")]:
        vals = [v for v in t[key].values() if v is not None]
        if vals:
            ax.axhline(sum(vals) / len(vals), ls=":", lw=1, c="#888")
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cfgs], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_title("ByteEmbed objective comparison + iso-compute subword baseline (dotted = teacher)")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig2_byte_objective.png")
    plt.close(fig)


def fig_byte_robustness(R):
    cfg = R["configs"].get("byte-both") or next(iter(R["configs"].values()), None)
    if not cfg:
        return
    import numpy as np
    perts = [p for p in R["perts"] if p in cfg["robust_agg"]]
    stu = [cfg["robust_agg"][p]["student"] for p in perts]
    tea = [cfg["robust_agg"][p]["teacher"] for p in perts]
    x = np.arange(len(perts))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(x - 0.2, stu, 0.4, label="byte student", color="#2a7")
    ax.bar(x + 0.2, tea, 0.4, label="subword teacher", color="#e94")
    ax.set_xticks(x)
    ax.set_xticklabels(perts, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(min(min(stu), min(tea)) - 0.02, 1.005)
    ax.set_ylabel("embedding stability\n(cosine, clean vs perturbed)")
    ax.set_title("H2: byte student is more robust than the subword teacher (8 perturbations)")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig3_byte_robustness.png")
    plt.close(fig)


def fig_downstream(R):
    import numpy as np
    bb = R["configs"].get("byte-both")
    mt = R["configs"].get("subword-mt5-both")
    t = R["teacher"]
    if not bb:
        return
    groups = ["SIB clean", "SIB romanized", "Tatoeba clean", "Tatoeba rom-src"]
    teacher_vals = [_m(t["sib"] and {"m": sum(v for v in t["sib"].values() if v is not None) /
                                     max(1, len([v for v in t["sib"].values() if v is not None]))}, "m"),
                    None, None, None]
    # simpler: use precomputed means
    def mean(d):
        vals = [v for v in d.values() if v is not None]
        return sum(vals) / len(vals) if vals else 0
    teacher_vals = [mean(t["sib"]), mean(t["sib_rom"]), mean(t["tatoeba"]), mean(t["tatoeba_rom"])]
    byte_vals = [bb["sib_mean"] or 0, bb["sib_rom_mean"] or 0,
                 bb["tatoeba_mean"] or 0, bb["tatoeba_rom_mean"] or 0]
    x = np.arange(len(groups))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - w, teacher_vals, w, label="subword teacher", color="#e94")
    ax.bar(x, byte_vals, w, label="byte student (both)", color="#2a7")
    if mt:
        sub_vals = [mt["sib_mean"] or 0, mt["sib_rom_mean"] or 0,
                    mt["tatoeba_mean"] or 0, mt["tatoeba_rom_mean"] or 0]
        ax.bar(x + w, sub_vals, w, label="subword student (mt5)", color="#59f")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_title("Downstream robustness: byte student degrades least from clean → romanized")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig4_downstream_robustness.png")
    plt.close(fig)


def fig_fertility(R):
    import numpy as np
    fert = R.get("fertility", {})
    bb = R["configs"].get("byte-both")
    if not fert or not bb:
        return
    within = ["diacritics", "spelling", "keyboard", "swap", "delete", "case"]
    langs = [lng for lng in fert if lng in bb["robust"]]
    fx = np.array([fert[lng] for lng in langs])
    g_in = np.array([np.mean([bb["robust"][lng][p]["gap"] for p in within if p in bb["robust"][lng]])
                     for lng in langs])
    g_rom = np.array([bb["robust"][lng].get("romanize", {}).get("gap", np.nan) for lng in langs])
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhline(0, c="#bbb", lw=0.8)
    ax.scatter(fx, g_in, c="#2a7", s=60, label="within-script noise (typos/diacritics/case/…)")
    ax.scatter(fx, g_rom, c="#c33", s=60, marker="s", label="romanization (script change)")
    for lng, a, b in zip(langs, fx, g_in):
        ax.annotate(lng, (a, b), fontsize=8, xytext=(3, 3), textcoords="offset points")
    r = np.corrcoef(fx, g_rom)[0, 1]
    m, c = np.polyfit(fx, g_rom, 1)
    xs = np.linspace(min(fx), max(fx), 50)
    ax.plot(xs, m * xs + c, ls="--", c="#c33", lw=1)
    ax.set_xlabel("teacher subword fertility (tokens / char)")
    ax.set_ylabel("byte robustness gap (student − teacher)")
    ax.set_title("Byte gain is within-script & uniform (+); romanization loss scales\n"
                 f"with fertility, i.e. non-Latin (romanize r={r:.2f}, within-script r="
                 f"{np.corrcoef(fx, g_in)[0, 1]:.2f})")
    ax.legend(fontsize=8.5, loc="center left")
    fig.savefig(FIG / "fig5_fertility.png")
    plt.close(fig)


def fig_tradeoff():
    """The contribution figure: robustness vs retrieval for every objective. Robustness axis =
    mean gap on HELD-OUT within-script perturbations (diacritics/swap/delete — never in any
    training augmentation, so fair for all configs). byte-robust should sit top-right."""
    import json
    import numpy as np
    cfgs = {}
    for p in ("results/byte_paper.json", "results/byte_robust.json"):
        pp = Path(p)
        if pp.exists():
            cfgs.update(json.loads(pp.read_text(encoding="utf-8")).get("configs", {}))
    held = ["diacritics", "swap", "delete"]
    pts = {}
    for name, c in cfgs.items():
        if name == "byte-random":
            continue
        x = c.get("tatoeba_mean")
        ra = c.get("robust_agg", {})
        ys = [ra[p]["gap"] for p in held if p in ra]
        if x is not None and ys:
            pts[name] = (x, float(np.mean(ys)))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0, c="#bbb", lw=0.8)
    for name, (x, y) in pts.items():
        hl = name.startswith("byte-robust")
        ax.scatter(x, y, s=100 if hl else 60, c="#2a7" if hl else "#27a", zorder=3)
        ax.annotate(name, (x, y), fontsize=9, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("retrieval — Tatoeba accuracy  (→ better)")
    ax.set_ylabel("robustness — held-out within-script gap vs teacher  (↑ better)")
    ax.set_title("Breaking the tradeoff: top-right = robust AND retrieves")
    fig.savefig(FIG / "fig6_tradeoff.png")
    plt.close(fig)


def fig_isocompute():
    """Quality-vs-parameters curve: byte vs subword students, with strong baselines as stars.
    (Sizes are not iso-param — byte spends params on the transformer, subword on a 250k vocab —
    so this is a quality-per-parameter view, the honest framing.)"""
    import json
    p = Path("results/byte_scale.json")
    if not p.exists():
        return
    M = json.loads(p.read_text(encoding="utf-8")).get("models", {})
    if not M:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in [(axes[0], "tatoeba_mean", "Tatoeba (bitext mining)"),
                              (axes[1], "sib_mean", "SIB-200 (classification)")]:
        for kind, color, marker in [("byte", "#2a7", "o"), ("subword", "#59f", "s")]:
            xy = sorted((r["params"] / 1e6, r.get(metric)) for n, r in M.items()
                        if r.get("kind") == kind and r.get(metric) is not None and r.get("params"))
            if xy:
                ax.plot([a for a, b in xy], [b for a, b in xy], marker=marker, c=color,
                        label=f"{kind} student", lw=1.5, ms=8)
        for n, r in M.items():
            if r.get("kind") == "baseline" and r.get(metric) is not None:
                ax.scatter(r["params"] / 1e6, r[metric], marker="*", s=170, c="#c33", zorder=3)
                ax.annotate(n, (r["params"] / 1e6, r[metric]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("parameters (M)")
        ax.set_ylabel(title)
        ax.legend(fontsize=8.5)
    fig.suptitle("Quality vs parameters: byte vs subword students (★ = strong baselines)")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_isocompute.png")
    plt.close(fig)


# ============================ low-resource study (run_lowresource.py) ============================
def _load_lowres():
    p = Path("results/byte_lowresource.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_lowres_tax(res):
    """HONEST tokenization-cost figure: subword TOKEN tax vs byte UTF-8 tax (both vs English, on the
    parallel FLORES-1012). Byte does NOT uniformly win — it costs MORE for non-Latin scripts."""
    eff = res.get("efficiency") or {}
    items = [(l, e) for l, e in eff.items() if e]
    if not items:
        return
    items.sort(key=lambda kv: -kv[1]["subword_tax"])
    import numpy as np
    x = np.arange(len(items)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - w / 2, [e["subword_tax"] for _, e in items], w, label="subword token tax (mt5)", color="#59f")
    ax.bar(x + w / 2, [e["byte_tax"] for _, e in items], w, label="byte UTF-8 tax", color="#2a7")
    ax.axhline(1.0, c="#bbb", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([l for l, _ in items])
    ax.set_ylabel("cost vs English (same content)")
    ax.set_title("Tokenization cost: subword token tax vs byte UTF-8 tax (lower = fairer)")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig_lowres_tax.png")
    plt.close(fig)


def fig_lowres_quality_params(res):
    """Quality-vs-parameters Pareto: byte vs subword students + baselines, on the uniform battery."""
    M = res.get("models", {})
    if not M:
        return
    panels = [("belebele_ndcg@10", "Belebele nDCG@10"), ("flores_p@1", "FLORES P@1"),
              ("sts_spearman", "STS Spearman")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, panels):
        for kind, color, marker in [("byte", "#2a7", "o"), ("subword", "#59f", "s")]:
            xy = sorted((r["params"] / 1e6, (r.get("means") or {}).get(key))
                        for r in M.values()
                        if r.get("kind") == kind and (r.get("means") or {}).get(key) is not None)
            if xy:
                ax.plot([a for a, b in xy], [b for a, b in xy], marker=marker, c=color,
                        label=f"{kind} student", lw=1.5, ms=8)
        for n, r in M.items():
            if r.get("kind") == "baseline" and (r.get("means") or {}).get(key) is not None:
                ax.scatter(r["params"] / 1e6, r["means"][key], marker="*", s=150, c="#c33", zorder=3)
                ax.annotate(n, (r["params"] / 1e6, r["means"][key]), fontsize=7.5,
                            xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("parameters (M)"); ax.set_ylabel(title); ax.legend(fontsize=8.5)
    fig.suptitle("Quality vs parameters — low-resource study (★ = baselines)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_lowres_quality_params.png")
    plt.close(fig)


def fig_lowres_delta_fertility(res, metric="belebele", size="base"):
    """Mechanism check: per-language byte−subword quality gap vs subword fertility. Tests whether
    byte wins MORE where the subword tokenizer fragments more (the hypothesized link)."""
    import numpy as np
    M = res.get("models", {}); eff = res.get("efficiency") or {}
    b, s = M.get(f"byte-{size}"), M.get(f"subword-{size}")
    if not (b and s and eff):
        return
    key = "ndcg@10" if metric == "belebele" else "p@1"
    xs, ys, labels = [], [], []
    for lang, e in eff.items():
        bl = (b.get(metric) or {}).get(lang); sl = (s.get(metric) or {}).get(lang)
        if e and bl and sl and bl.get(key) is not None and sl.get(key) is not None:
            xs.append(e["mt5_tok_per_char"]); ys.append(bl[key] - sl[key]); labels.append(lang)
    if len(xs) < 3:
        return
    r = float(np.corrcoef(xs, ys)[0, 1])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(0, c="#bbb", lw=0.8)
    ax.scatter(xs, ys, s=70, c="#2a7", zorder=3)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), fontsize=9, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("subword fertility (mt5 tokens/char)")
    ax.set_ylabel(f"byte − subword  ({metric} {key})")
    ax.set_title(f"Does byte win where subword fragments?  (r = {r:.2f}, {size} models)")
    fig.savefig(FIG / "fig_lowres_delta_fertility.png")
    plt.close(fig)


def main_lowres():
    R = _load_lowres()
    if R is None:
        print("results/byte_lowresource.json not found — low-resource figures skipped")
        return
    for fn in (fig_lowres_tax, fig_lowres_quality_params, fig_lowres_delta_fertility):
        try:
            fn(R)
            print(f"{fn.__name__} written")
        except Exception as e:  # noqa: BLE001
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")


def main():
    R = _load()
    if R is None:
        print("results/byte_paper.json not found — ByteEmbed figures skipped")
        return
    for fn in (fig_byte_objective, fig_byte_robustness, fig_downstream, fig_fertility):
        try:
            fn(R)
            print(f"{fn.__name__} written")
        except Exception as e:  # noqa: BLE001
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
    try:
        fig_tradeoff()
        print("fig6 (tradeoff) written")
    except Exception as e:  # noqa: BLE001
        print(f"fig_tradeoff FAILED: {type(e).__name__}: {e}")
    try:
        fig_isocompute()
        print("fig7 (iso-compute / quality-vs-params) written")
    except Exception as e:  # noqa: BLE001
        print(f"fig_isocompute FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
    main_lowres()
