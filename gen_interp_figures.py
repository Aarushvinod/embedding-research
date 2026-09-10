"""Figures for the five interpretability analyses (results/interp_*.json -> figures/fig_interp_*.png).

  python gen_interp_figures.py            # run after `python -m byte_embed.interp_<x> --merge`
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
plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "savefig.bbox": "tight"})
MODELS = ["byte-small", "subword-small", "byte-base", "subword-base", "byte-large", "subword-large"]
COLOR = {"byte-small": "#9ecae1", "byte-base": "#4292c6", "byte-large": "#08519c",
         "subword-small": "#fdae6b", "subword-base": "#f16913", "subword-large": "#a63603"}


def _load(analysis):
    p = Path(f"results/interp_{analysis}.json")
    return json.loads(p.read_text(encoding="utf-8"))["models"] if p.exists() else None


def fig_params():
    M = _load("params")
    if not M:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    names = [n for n in MODELS if n in M]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [M[n]["params"]["dense_frac"] for n in names], 0.4, label="dense (non-vocab) fraction",
              color=[COLOR[n] for n in names])
    ax[0].bar(x + 0.2, [M[n]["vocab_util"]["frac_hit"] for n in names], 0.4, label="vocab rows hit",
              color=[COLOR[n] for n in names], hatch="//", alpha=0.7)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Parameter allocation & vocabulary use"); ax[0].legend(fontsize=8)
    for n in names:
        L = M[n]["layers"]
        ax[1].plot([e["layer"] / max(L[-1]["layer"], 1) for e in L],
                   [e["std"]["participation_ratio"] for e in L], color=COLOR[n], label=n, marker=".")
    ax[1].set_xlabel("relative depth"); ax[1].set_ylabel("participation ratio (standardized)")
    ax[1].set_title("Effective dimensionality by layer"); ax[1].legend(fontsize=7)
    fig.savefig(FIG / "fig_interp_params.png"); plt.close(fig)


def fig_alignuni():
    M = _load("alignuni")
    if not M:
        return
    names = [n for n in MODELS if n in M and M[n].get("teacher_align")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(names))
    ax[0].bar(x, [M[n]["teacher_align"]["mean"] for n in names], color=[COLOR[n] for n in names])
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right")
    ax[0].set_title("Alignment to teacher (lower = closer)")
    ax[1].bar(x, [M[n]["uniformity"]["all"] for n in names], color=[COLOR[n] for n in names])
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, ha="right")
    ax[1].set_title("Uniformity (more negative = better spread)")
    fig.savefig(FIG / "fig_interp_alignuni.png"); plt.close(fig)


def fig_langgeom():
    M = _load("langgeom")
    if not M:
        return
    from byte_embed.interp_langgeom import is_treatment
    from byte_embed.stats import compare
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in MODELS:
        r = M.get(n)
        if not r or "layers" not in r:
            continue
        L = r["layers"]
        ax[0].plot([e["layer"] / max(L[-1]["layer"], 1) for e in L], [e["lang_probe_acc"] for e in L],
                   color=COLOR[n], label=n, marker=".")
    ax[0].set_xlabel("relative depth"); ax[0].set_ylabel("language-ID probe accuracy")
    ax[0].set_title("Where language identity lives"); ax[0].legend(fontsize=7)
    labels, T, C, cols = [], [], [], []
    for n in MODELS:
        r = M.get(n)
        if not r or "leace" not in r.get("variants", {}) or "none" not in r.get("variants", {}):
            continue
        rows = compare(r["variants"]["leace"], r["variants"]["none"], n_boot=2000)
        t = [x["delta"] for c, x in rows if is_treatment(c)]
        c = [x["delta"] for c_, x in rows if not is_treatment(c_)]
        labels.append(n); T.append(np.mean(t) if t else 0); C.append(np.mean(c) if c else 0); cols.append(COLOR[n])
    if labels:
        x = np.arange(len(labels))
        ax[1].bar(x - 0.2, T, 0.4, color=cols, label="cross-lingual (treatment)")
        ax[1].bar(x + 0.2, C, 0.4, color=cols, alpha=0.5, hatch="//", label="monolingual (control)")
        ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=30, ha="right"); ax[1].legend(fontsize=8)
        ax[1].set_ylabel("Δ nDCG@10 after LEACE erasure"); ax[1].set_title("Erasing language identity")
    fig.savefig(FIG / "fig_interp_langgeom.png"); plt.close(fig)


def fig_script():
    M = _load("script")
    if not M:
        return
    names = [n for n in MODELS if n in M]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [M[n]["cells"]["p@1"]["same_script"] for n in names], 0.4,
              color=[COLOR[n] for n in names], label="same script (Latin–Latin)")
    ax[0].bar(x + 0.2, [M[n]["cells"]["p@1"]["cross_script"] for n in names], 0.4,
              color=[COLOR[n] for n in names], alpha=0.5, hatch="//", label="cross script")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right")
    ax[0].set_ylabel("FLORES cross-lingual P@1"); ax[0].set_title("Script-pair alignment"); ax[0].legend(fontsize=8)
    langs = list(next(iter(M.values()))["shift"])
    w = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        ax[1].bar(np.arange(len(langs)) + i * w, [M[n]["shift"][l]["cos_mean"] for l in langs], w,
                  color=COLOR[n], label=n)
    ax[1].set_xticks(np.arange(len(langs)) + 0.4 - w / 2); ax[1].set_xticklabels(langs)
    ax[1].set_ylabel("cos(native, romanized)"); ax[1].set_title("Romanization shift"); ax[1].legend(fontsize=7)
    fig.savefig(FIG / "fig_interp_script.png"); plt.close(fig)


def fig_segment():
    M = _load("segment")
    if not M:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for n in ["byte-small", "byte-base", "byte-large"]:
        r = M.get(n)
        if not r or not r.get("langs"):
            continue
        langs = [l for l in r["langs"] if l != "zh"]
        layer_ids = [e["layer"] for e in r["langs"][langs[0]]["layers"]]
        for lab, ls in (("interior", "-"), ("random", ":")):
            ys = []
            for l in layer_ids:
                vals = [e[lab]["bacc"] for lang in langs
                        for e in r["langs"][lang]["layers"] if e["layer"] == l and e.get(lab)]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(layer_ids, ys, ls, color=COLOR[n], marker="." if lab == "interior" else None,
                    label=f"{n} {lab}")
    ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Sub-word boundary decodability (byte students)"); ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "fig_interp_segment.png"); plt.close(fig)


def main():
    for fn in (fig_params, fig_alignuni, fig_langgeom, fig_script, fig_segment):
        try:
            fn()
            print(f"{fn.__name__}: ok")
        except Exception as e:  # noqa: BLE001 — one bad figure never kills the batch
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
