"""Per-language, per-task figures from a *_lowresource.json results file.

For every task (SIB-200, Belebele, STS, FLORES bitext, MIRACL) draws languages on the x-axis,
segmented by model, in two styles:
  * grouped bar charts  -> figures/perlang/perlang_bar_<task>.png
  * model x language heatmaps (cleaner for 8 models x 9 langs) -> perlang_heat_<task>.png

Reusable across runs (byte study, UCD study, ...):
  python make_perlang_figures.py results/byte_lowresource.json
  python make_perlang_figures.py <results.json> [outdir]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/byte_lowresource.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/perlang"
os.makedirs(OUT, exist_ok=True)

data = json.loads(open(SRC, encoding="utf-8").read())
M = data["models"]
LANGS = data["langs"]
LANG_NAME = {"te": "Telugu", "ta": "Tamil", "mr": "Marathi", "am": "Amharic", "ha": "Hausa",
             "rw": "Kinyarw.", "en": "English", "zh": "Chinese", "ar": "Arabic"}
LOW_RES = {"te", "ta", "mr", "am", "ha", "rw"}

# model display order + a color scheme that encodes structure: byte = blues, subword = oranges
# (light->dark = small->base->large), baselines = green/grey. Only models present are drawn.
ORDER = ["byte-small", "byte-base", "byte-large",
         "subword-small", "subword-base", "subword-large",
         "ucd-small", "ucd-base", "ucd-large", "mE5-base", "LaBSE"]
COLORS = {"byte-small": "#9ecae1", "byte-base": "#4292c6", "byte-large": "#08519c",
          "subword-small": "#fdae6b", "subword-base": "#f16913", "subword-large": "#a63603",
          "ucd-small": "#a1d99b", "ucd-base": "#41ab5d", "ucd-large": "#006d2c",
          "mE5-base": "#9e9ac8", "LaBSE": "#bdbdbd"}
MODELS = [m for m in ORDER if m in M and M[m].get("kind") != "baseline"]  # students only: byte vs subword


def v_sib(m, l):
    return M[m].get("sib", {}).get(l)


def v_bel(m, l):
    x = M[m].get("belebele", {}).get(l)
    return x.get("ndcg@10") if x else None


def v_sts(m, l):
    x = M[m].get("sts", {}).get(l)
    return x.get("spearman") if x else None


def v_flo(m, l):
    x = M[m].get("flores_bitext", {}).get(l)
    return x.get("p@1") if x else None


def v_mir(m, l):
    x = (M[m].get("miracl") or {}).get("per_lang", {}).get(l)
    return x.get("ndcg@10") if x else None


FLO_LANGS = [l for l in LANGS if any(v_flo(m, l) is not None for m in MODELS)]
MIR_LANGS = [l for l in LANGS if any(v_mir(m, l) is not None for m in MODELS)]

# (title, key, value-fn, langs, metric label, (ylim_lo, ylim_hi))
TASKS = [
    ("SIB-200 classification", "sib", v_sib, LANGS, "accuracy", (0, 1.0)),
    ("Belebele retrieval", "belebele", v_bel, LANGS, "nDCG@10", (0, 1.0)),
    ("STS semantic similarity", "sts", v_sts, LANGS, "Spearman", (0, 1.0)),
    ("FLORES-1012 bitext mining (->English)", "flores", v_flo, FLO_LANGS, "P@1", (0.95, 1.004)),
    ("MIRACL deep retrieval", "miracl", v_mir, MIR_LANGS, "nDCG@10", (0, 1.0)),
]


def _sep_x(langs):
    """x position to separate the leading low-resource block from the high-resource anchors."""
    n_lr = sum(1 for l in langs if l in LOW_RES)
    return (n_lr - 0.5) if 0 < n_lr < len(langs) else None


def grouped_bar(title, key, fn, langs, metric, ylim):
    fig, ax = plt.subplots(figsize=(max(11, 1.5 * len(langs) + 3.5), 6.2))
    n, x = len(MODELS), np.arange(len(langs))
    w = 0.82 / n
    for i, m in enumerate(MODELS):
        vals = [fn(m, l) or 0 for l in langs]
        ax.bar(x + i * w - 0.41 + w / 2, vals, w, label=m, color=COLORS[m],
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{LANG_NAME[l]}" for l in langs])
    ax.set_ylim(*ylim)
    ax.set_ylabel(metric)
    ax.set_title(f"{title} - per language, by model   (| = low-resource / high-resource anchors)",
                 fontsize=12, fontweight="bold", pad=12)
    sep = _sep_x(langs)
    if sep is not None:
        ax.axvline(sep, color="grey", ls="--", lw=1, alpha=0.6)
    ax.legend(ncol=min(len(MODELS), 6), loc="upper center", bbox_to_anchor=(0.5, -0.10),
              frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, f"perlang_bar_{key}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def heatmap(title, key, fn, langs, metric, ylim):
    lo, hi = ylim
    grid = np.array([[fn(m, l) if fn(m, l) is not None else np.nan for l in langs] for m in MODELS])
    fig, ax = plt.subplots(figsize=(max(7, 0.95 * len(langs) + 3.2), 0.55 * len(MODELS) + 2.2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=lo, vmax=hi)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([f"{l}\n{LANG_NAME[l]}" for l in langs])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(len(MODELS)):
        for j in range(len(langs)):
            val = grid[i, j]
            if not np.isnan(val):
                norm = (val - lo) / (hi - lo + 1e-9)
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if norm < 0.55 else "black", fontsize=8)
    # separate students from baseline rows
    n_base = sum(1 for m in MODELS if M[m].get("kind") == "baseline")
    if 0 < n_base < len(MODELS):
        ax.axhline(len(MODELS) - n_base - 0.5, color="white", lw=2)
    sep = _sep_x(langs)
    if sep is not None:
        ax.axvline(sep, color="white", lw=2)
    ax.set_title(f"{title} ({metric}) - model x language", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    p = os.path.join(OUT, f"perlang_heat_{key}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


written = []
for t in TASKS:
    written.append(grouped_bar(*t))
    written.append(heatmap(*t))
print(f"models: {MODELS}")
print(f"wrote {len(written)} figures to {OUT}/:")
for p in written:
    print(" ", os.path.basename(p))
