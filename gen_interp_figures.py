"""Figures for the three interpretability experiments (results/interp_*.json -> figures/fig_interp_*.png).

  python gen_interp_figures.py            # run after each experiment's `--merge`

  fig_interp_english.png        latent-English curves + the Belebele erasure matrix summary per model
  fig_interp_english_tsne.png   FLORES final embeddings, raw vs English-erased, coloured by language
  fig_interp_script.png         RR / RN drops per language, byte vs subword at each size
  fig_interp_segment.png        interior-boundary decodability by layer (trained / pretrained / surface /
                                random) + the cross-lingual transfer matrix per byte model
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


def _load(analysis):
    p = Path(f"results/interp_{analysis}.json")
    return json.loads(p.read_text(encoding="utf-8"))["models"] if p.exists() else None


# ----------------------------------------------------------------------------------------------
def fig_english():
    from byte_embed.interp_common import stored_results
    from byte_embed.interp_english import matrix_summary, non_en_mean
    M = _load("english")
    if not M:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in MODELS:
        r = M.get(n)
        if not r or "latent" not in r:
            continue
        xs = [b / r["n_blocks"] for b in r["blocks"]]
        ax[0].plot(xs, [non_en_mean(r["latent"][str(b)]["p_en"]) for b in r["blocks"]], "-o", color=COLOR[n], label=n)
    ax[0].set_xlabel("relative depth of the block"); ax[0].set_ylabel("mean P(English) of non-English positions")
    ax[0].set_title("Latent English by depth"); ax[0].legend(fontsize=7)
    names, cols, en_col, other, rnd = [], [], [], [], []
    for n in MODELS:
        r = M.get(n)
        base = stored_results(n)
        if not (r and base and r.get("belebele")):
            continue
        s = matrix_summary(r, base)
        names.append(n); cols.append(COLOR[n])
        en_col.append(s["en_col_non_en"] or 0); other.append(s["other_cols_off_diag"] or 0); rnd.append(s["random"] or 0)
    if names:
        x = np.arange(len(names))
        ax[1].bar(x - 0.27, en_col, 0.27, color=cols, label="English erased, effect on other languages")
        ax[1].bar(x, other, 0.27, color=cols, alpha=0.55, hatch="//", label="other language erased, effect on others")
        ax[1].bar(x + 0.27, rnd, 0.27, color=cols, alpha=0.3, label="random direction")
        ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, ha="right")
        ax[1].set_ylabel("Δ nDCG@10 (Belebele)"); ax[1].set_title("Erasure inside the encoder"); ax[1].legend(fontsize=7)
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
        lang = z["lang"].astype(str)
        keep = np.concatenate([rng.choice(np.flatnonzero(lang == l), size=min(n_per_lang, int((lang == l).sum())),
                                          replace=False) for l in sorted(set(lang))])
        for col, key in enumerate(("raw", "erased_en")):
            E = z[key].astype(np.float32)[keep]
            Y = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(PCA(50).fit_transform(E))
            ax = axes[row, col]
            for l in sorted(set(lang)):
                m = lang[keep] == l
                ax.scatter(Y[m, 0], Y[m, 1], s=6, color=LANG_COLOR.get(l, "k"), marker=MARK.get(SCRIPT.get(l), "o"),
                           label=l, alpha=0.7, linewidths=0)
            ax.set_title(f"{n}: {'unedited' if key == 'raw' else 'English erased at the chosen block'}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].legend(fontsize=6, markerscale=2, ncol=2)
    fig.savefig(FIG / "fig_interp_english_tsne.png"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def fig_script():
    from byte_embed.interp_common import stored_results
    from byte_embed.stats import compare
    M = _load("script")
    if not M:
        return
    sizes = [s for s in ("small", "base", "large") if M.get(f"byte-{s}") and M.get(f"subword-{s}")]
    if not sizes:
        return
    fig, axes = plt.subplots(len(sizes), 2, figsize=(10, 3.4 * len(sizes)), squeeze=False)
    for row, size in enumerate(sizes):
        for col, cond in enumerate(("RR", "RN")):
            ax = axes[row, col]
            key = f"uroman:{cond}"
            for k, n in enumerate((f"byte-{size}", f"subword-{size}")):
                r, base = M[n], stored_results(n)
                if not (base and r.get("cond", {}).get(key)):
                    continue
                rows = compare(r["cond"][key], base, n_boot=2000)
                labels = [f"{c[0][:3]}-{c[1]}" for c, _ in rows]
                x = np.arange(len(rows))
                ax.bar(x + (k - 0.5) * 0.4, [v["delta"] for _, v in rows], 0.4, color=COLOR[n], label=n,
                       yerr=[[v["delta"] - v["ci_low"] for _, v in rows], [v["ci_high"] - v["delta"] for _, v in rows]],
                       capsize=2)
                ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
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
                ys.append(np.mean(vals) if vals else np.nan)
            return ys
        ax.plot(layer_ids, curve("layers", "interior"), "-o", color=COLOR[n], label="interior, trained")
        ax.plot(layer_ids, curve("pretrained", "interior"), "--s", color=COLOR[n], alpha=0.6, label="interior, pretrained ByT5")
        ax.plot(layer_ids, curve("layers", "random"), ":", color="k", label="random control")
        sur = [r["langs"][l]["surface"]["interior"]["bacc"] for l in langs if (r["langs"][l].get("surface") or {}).get("interior")]
        if sur:
            ax.axhline(np.mean(sur), color="gray", lw=1, label="surface-statistics baseline")
        ax.set_ylim(0.4, 1.0); ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy")
        ax.set_title(f"{n}: sub-word boundary decodability (mean over space-delimited languages)", fontsize=9)
        ax.legend(fontsize=7)
        t = r.get("transfer")
        ax = axes[row, 1]
        if t:
            L = list(t["acc"])
            Mx = np.array([[t["ratio"][a][b] if t["ratio"][a][b] is not None else np.nan for b in L] for a in L])
            im = ax.imshow(Mx, vmin=0.5, vmax=1.05, cmap="viridis")
            ax.set_xticks(range(len(L))); ax.set_xticklabels(L); ax.set_yticks(range(len(L))); ax.set_yticklabels(L)
            ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
            ax.set_title(f"{n}: transfer ratio acc(A→B)/acc(B→B) at layer {t['layer']}", fontsize=9)
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
