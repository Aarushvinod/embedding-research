"""Figures for the three interpretability experiments (results/interp_*.json -> figures/fig_interp_*.png).

  python gen_interp_figures.py            # run after each experiment's `--merge`

  fig_interp_english.png        latent-English curve, the erasure bars with CIs, and the 10x10 matrix
  fig_interp_english_tsne.png   FLORES final embeddings, raw vs English-erased, coloured by language
  fig_interp_script.png         RR / RN drops per language, byte vs subword at each size
  fig_interp_segment.png        interior-boundary decodability by layer + the transfer matrix

A missing measurement is never drawn as zero: bars for cells that have not been computed are NaN, so
they are visibly absent rather than reading as "the intervention had no effect".
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.bbox": "tight"})
MODELS = ["byte-small", "subword-small", "byte-base", "subword-base", "byte-large", "subword-large"]
COLOR = {"byte-small": "#9ecae1", "byte-base": "#4292c6", "byte-large": "#08519c",
         "subword-small": "#fdae6b", "subword-base": "#f16913", "subword-large": "#a63603"}
LANG_COLOR = {"te": "#1f77b4", "bn": "#aec7e8", "am": "#2ca02c", "zh": "#d62728", "ar": "#9467bd",
              "sw": "#8c564b", "yo": "#e377c2", "ha": "#7f7f7f", "rw": "#bcbd22", "en": "#17becf"}
MARK = {"Latn": "o", "Telu": "s", "Beng": "^", "Ethi": "D", "Hans": "v", "Arab": "P"}
NAN = float("nan")


def _load(analysis):
    p = Path(f"results/interp_{analysis}.json")
    return json.loads(p.read_text(encoding="utf-8"))["models"] if p.exists() else None


def _mean(x):
    """A bootstrap block -> its mean, or NaN. Never 0: a missing cell must not render as a null
    effect."""
    return x["mean"] if isinstance(x, dict) and x.get("mean") is not None else NAN


def _err(x):
    if not isinstance(x, dict) or x.get("mean") is None:
        return [[0.0], [0.0]]
    return [[x["mean"] - x["ci_low"]], [x["ci_high"] - x["mean"]]]


# ----------------------------------------------------------------------------------------------
def fig_english(n_boot=2000):
    from byte_embed.interp_english import NONE, en_by_depth, matrix_summary, non_en_mean
    M = _load("english")
    if not M:
        return
    fig = plt.figure(figsize=(13, 4.2))
    ax0, ax1, ax2 = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))
    for n in MODELS:
        r = M.get(n)
        if not r or "latent" not in r:
            continue
        xs = [(b + 1) / r["n_blocks"] for b in r["blocks"]]
        ax0.plot(xs, [non_en_mean(r["latent"][str(b)]["p_en"]) for b in r["blocks"]], "-o",
                 color=COLOR[n], label=n)
    ax0.set_xlabel("relative depth of the block"); ax0.set_ylabel("mean P(English) of non-English positions")
    ax0.set_title("Latent English by depth"); ax0.legend(fontsize=7)

    names, summ = [], {}
    for n in MODELS:
        r = M.get(n)
        s = matrix_summary(r, n_boot) if r else None
        if s:
            names.append(n); summ[n] = s
    if names:
        x = np.arange(len(names))
        for off, key, lab, kw in ((-0.27, "en_col", "English erased", {}),
                                  (0.0, "other_cols", "another language erased", dict(alpha=0.55, hatch="//")),
                                  (0.27, "random", "random direction", dict(alpha=0.3))):
            ax1.bar(x + off, [_mean(summ[n][key]) for n in names], 0.26,
                    color=[COLOR[n] for n in names], label=lab,
                    yerr=np.concatenate([_err(summ[n][key]) for n in names], axis=1), capsize=2, **kw)
        ax1.axhline(0, color="k", lw=0.8); ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=30, ha="right")
        ax1.set_ylabel("Δ nDCG@10 vs the unedited pass"); ax1.legend(fontsize=7)
        ax1.set_title("Erasure inside the encoder (Belebele, 95% CI)")

    # the 10x10 matrix itself, for the first model that has one: rows = language erased,
    # columns = language measured. This is what the nine control columns were computed for.
    for n in names:
        r = M[n]
        cb, bel = r.get("chosen_block"), r.get("belebele", {})
        langs = [l for l, c in (bel.get(NONE) or {}).items() if c]
        rows = [m for m in langs if f"{cb}:{m}" in bel]
        if not rows:
            continue
        Z = np.full((len(rows), len(langs)), NAN)
        for i, m in enumerate(rows):
            for j, l in enumerate(langs):
                a, b = (bel[f"{cb}:{m}"] or {}).get(l), (bel[NONE] or {}).get(l)
                if a and b:
                    Z[i, j] = a["ndcg@10"] - b["ndcg@10"]
        v = np.nanmax(np.abs(Z)) or 1.0
        im = ax2.imshow(Z, cmap="RdBu", vmin=-v, vmax=v)
        ax2.set_xticks(range(len(langs))); ax2.set_xticklabels(langs, fontsize=7)
        ax2.set_yticks(range(len(rows))); ax2.set_yticklabels(rows, fontsize=7)
        ax2.set_xlabel("effect on"); ax2.set_ylabel("direction erased")
        ax2.set_title(f"{n}: Δ nDCG@10 @ block {cb}", fontsize=9)
        fig.colorbar(im, ax=ax2, fraction=0.046)
        break
    fig.savefig(FIG / "fig_interp_english.png"); plt.close(fig)


def fig_english_tsne(n_per_lang=300, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from byte_embed.interp_common import SCRIPT
    from byte_embed.interp_english import embeds_path
    names = [n for n in MODELS if embeds_path(n).exists()]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 4 * len(names)), squeeze=False)
    rng = np.random.default_rng(seed)
    for row, n in enumerate(names):
        z = np.load(embeds_path(n), allow_pickle=True)
        if "erased_en" not in z.files:
            continue
        lang = z["lang"].astype(str)
        keep = np.concatenate([rng.choice(np.flatnonzero(lang == l),
                                          size=min(n_per_lang, int((lang == l).sum())), replace=False)
                               for l in sorted(set(lang))])
        for col, key in enumerate(("raw", "erased_en")):
            E = z[key].astype(np.float32)[keep]
            Y = TSNE(n_components=2, init="pca", perplexity=30,
                     random_state=seed).fit_transform(PCA(50).fit_transform(E))
            ax = axes[row, col]
            for l in sorted(set(lang)):
                m = lang[keep] == l
                ax.scatter(Y[m, 0], Y[m, 1], s=6, color=LANG_COLOR.get(l, "k"),
                           marker=MARK.get(SCRIPT.get(l), "o"), label=l, alpha=0.7, linewidths=0)
            ax.set_title(f"{n}: {'unedited' if key == 'raw' else 'English erased at the chosen block'}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].legend(fontsize=6, markerscale=2, ncol=2)
    fig.savefig(FIG / "fig_interp_english_tsne.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig_script(n_boot=2000):
    from byte_embed.interp_script import NN, baseline_for
    from byte_embed.stats import compare
    M = _load("script")
    if not M:
        return
    sizes = [s for s in ("small", "base", "large") if M.get(f"byte-{s}") and M.get(f"subword-{s}")]
    if not sizes:
        return
    fig, axes = plt.subplots(len(sizes), 2, figsize=(11, 3.4 * len(sizes)), squeeze=False)
    for row, size in enumerate(sizes):
        for col, cond in enumerate(("RR", "RN")):
            ax, key = axes[row, col], f"uroman:{cond}"
            got = {}
            for n in (f"byte-{size}", f"subword-{size}"):
                r = M[n]
                base, _ = baseline_for(r, n, "results/retrieval_bgem3.json")
                if base and r.get("cond", {}).get(key) and key != NN:
                    got[n] = dict(compare(r["cond"][key], base, n_boot))
            # one x-axis for both models: the UNION of their cells, so bar i is the same cell in both
            cells = sorted({c for rows in got.values() for c in rows})
            if not cells:
                continue
            x = np.arange(len(cells))
            for k, (n, rows) in enumerate(got.items()):
                ax.bar(x + (k - 0.5) * 0.4, [rows[c]["delta"] if c in rows else NAN for c in cells], 0.4,
                       color=COLOR[n], label=n,
                       yerr=np.array([[rows[c]["delta"] - rows[c]["ci_low"] if c in rows else 0 for c in cells],
                                      [rows[c]["ci_high"] - rows[c]["delta"] if c in rows else 0 for c in cells]]),
                       capsize=2)
            ax.set_xticks(x); ax.set_xticklabels([f"{c[0][:3]}-{c[1]}" for c in cells], rotation=40,
                                                 ha="right", fontsize=7)
            ax.axhline(0, color="k", lw=0.8); ax.legend(fontsize=7)
            ax.set_ylabel("Δ nDCG@10 romanized − native")
            ax.set_title(f"{size}: {'both sides romanized' if cond == 'RR' else 'romanized queries vs native passages'}")
    fig.savefig(FIG / "fig_interp_script.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig_segment():
    M = _load("segment")
    if not M:
        return
    names = [n for n in ("byte-small", "byte-base", "byte-large") if M.get(n) and M[n].get("langs")]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.8 * len(names)), squeeze=False)
    for row, n in enumerate(names):
        r = M[n]
        langs = [l for l in r["langs"] if l != "zh"] or list(r["langs"])
        layer_ids = [e["layer"] for e in r["langs"][langs[0]]["layers"]]
        ax = axes[row, 0]

        def curve(key, lab):
            ys = []
            for l in layer_ids:
                vals = [e[lab]["bacc"] for lang in langs for e in (r["langs"][lang].get(key) or [])
                        if e["layer"] == l and e.get(lab)]
                ys.append(np.mean(vals) if vals else NAN)
            return ys
        ax.plot(layer_ids, curve("layers", "interior"), "-o", color=COLOR[n], label="interior, trained")
        ax.plot(layer_ids, curve("pretrained", "interior"), "--s", color=COLOR[n], alpha=0.6,
                label="interior, pretrained ByT5")
        ax.plot(layer_ids, curve("layers", "random"), ":", color="k", label="random control")
        sur = [r["langs"][l]["surface"]["interior"]["bacc"] for l in langs
               if (r["langs"][l].get("surface") or {}).get("interior")]
        if sur:
            ax.axhline(np.mean(sur), color="gray", lw=1, label="surface-statistics baseline")
        ax.set_ylim(0.4, 1.0); ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy")
        ax.set_title(f"{n}: sub-word boundary decodability (mean over space-delimited languages)", fontsize=9)
        ax.legend(fontsize=7)
        t, ax = r.get("transfer"), axes[row, 1]
        if t:
            L = list(t.get("auc") or t["acc"])
            Z = np.array([[t["ratio"][a][b] if t["ratio"][a][b] is not None else NAN for b in L] for a in L])
            im = ax.imshow(Z, vmin=0.5, vmax=1.05, cmap="viridis")
            ax.set_xticks(range(len(L))); ax.set_xticklabels(L); ax.set_yticks(range(len(L)))
            ax.set_yticklabels(L)
            ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
            ax.set_title(f"{n}: transfer AUC(A→B)/AUC(B→B) @ layer {t['layer']} "
                         f"({len(L)} langs, {t.get('n_sent', '?')} sent)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis("off")
    fig.savefig(FIG / "fig_interp_segment.png"); plt.close(fig)


def main():
    for fn in (fig_english, fig_english_tsne, fig_script, fig_segment):
        try:
            fn()
            print(f"{fn.__name__}: ok")
        except Exception as e:  # noqa: BLE001 — one bad figure never kills the batch
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
