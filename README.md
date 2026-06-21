# embedding-research

Feasibility experiments for two embedding research ideas:

1. **GATE-GRAFT** (`gate_graft/`) — bitext-free cross-lingual alignment of a frozen
   text embedder via Gromov–Wasserstein structure-matching, with a **per-token
   reliability gate** that trusts the structural map only where local geometry is
   isomorphic and falls back otherwise. *Runs fully on a laptop (no training).*
2. **ByteEmbed** (`byte_embed/`) — a tokenizer-free **byte-level dense retriever**
   distilled from a frozen subword teacher, to test whether removing the subword
   vocabulary helps the long tail. *Training-heavy — runs on Colab A100 / cloud GPU.*

## Results (start here)
- **`PAPER.md`** — the **curated, paper-ready** results for both threads + the combined narrative
  and honest scope. This is what would go in a paper.
- **`RESULTS.md`** (GRAFT) and **`byte_embed/RESULTS.md`** (ByteEmbed) — full/detailed **reference**
  results, controls, ablations, and the experimental audit.
- **`figures/`** — paper figures (PNG); regenerate with `python gen_figures.py`.
- Raw run JSON lives under `results/` (gitignored; kept locally for reference).

Headline: contrastive distillation closes ByteEmbed's retrieval gap (Tatoeba 0.12→0.56); at
iso-compute the byte student beats a subword (mt5) student and beats the teacher on *romanized*
classification; robustness is within-script (it loses on romanization). GRAFT recovers 43–85% of a
trained multilingual model bitext-free but is resource-bounded. See `PAPER.md` for the honest scope.

## Layout
```
PAPER.md       curated paper-ready results (both threads) — START HERE
RESULTS.md     GRAFT detailed/reference results + audit
figures/       paper figures (PNG); built by gen_figures.py
common/        shared loaders (fastText, MUSE dicts) + eval (BLI, CSLS, SIB, STS)
gate_graft/    GW/anchored alignment + reliability gate + graft + 21-lang study  (LOCAL, no training)
byte_embed/    byte student + distillation (cosine/contrastive/both) + run_paper + benchmark + RESULTS.md
scripts/       check_env.py, download_data.py
data/          downloaded vectors/dictionaries (gitignored)
results/       raw run JSON (gitignored; kept locally for reference)
```

## Quick start

### GATE-GRAFT (local — your laptop is plenty)
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-local.txt
pip install -e .
python scripts/check_env.py
python -m gate_graft.run_feasibility --langs ca tr bn eu --k 2000
```
Inference + CPU optimal transport only. ~1 GB VRAM, a few GB RAM, minutes/language.

### ByteEmbed (Colab A100 80GB primary; RunPod/Vast.ai fallback)
- **Colab (recommended):** open `byte_embed/colab_feasibility.ipynb` in Colab, set
  runtime to A100, Run All. Self-contained (installs deps, downloads data, distills,
  evaluates robustness).
- **Cloud fallback / reproducibility:** see `byte_embed/cloud/` (`setup.sh` provisions a
  RunPod/Vast.ai 4090/A100). Estimated cost on a rented 4090: **~$12** for the full
  feasibility pass (cap ~$25). Colab A100 80GB is free-er if you already have credits and
  unlocks larger batches / the optional contrastive route.

## Hardware notes
- Author machine: RTX 5070 Ti **Laptop** GPU (12 GB, Blackwell/sm_120), 32 GB RAM,
  Core Ultra 9 275HX. GATE-GRAFT runs here; ByteEmbed training does not fit comfortably.
- **Blackwell caveat:** local GPU work needs PyTorch built for CUDA 12.8+ (cu128).
  `requirements-cloud.txt` documents the right install. Colab/cloud images handle this.

## Status
Feasibility **and** paper-level experiments have been **run** — locally on the RTX 5070 Ti
(byt5-small/base byte students, the mt5-small subword baseline, contrastive distillation, the
21-language GRAFT study; everything fit on 12 GB). Curated results in `PAPER.md`. Reproduce with
`python -m byte_embed.run_paper --steps 2000 --batch 16` (→ `python gen_figures.py`) and
`python -m gate_graft.run_matrix`.
