# embedding-research

Feasibility experiments for two embedding research ideas:

1. **GATE-GRAFT** (`gate_graft/`) — bitext-free cross-lingual alignment of a frozen
   text embedder via Gromov–Wasserstein structure-matching, with a **per-token
   reliability gate** that trusts the structural map only where local geometry is
   isomorphic and falls back otherwise. *Runs fully on a laptop (no training).*
2. **ByteEmbed** (`byte_embed/`) — a tokenizer-free **byte-level dense retriever**
   distilled from a frozen subword teacher, to test whether removing the subword
   vocabulary helps the long tail. *Training-heavy — runs on Colab A100 / cloud GPU.*

These are **feasibility smoke tests**, not the full proposals. The goal is to answer
"is the core mechanism worth pursuing?" cheaply, before committing to a full study.
Proposals live in the conversation that generated this repo; see `gate_graft/README.md`
and `byte_embed/README.md` for what each test does and does not show.

## Layout
```
common/        shared loaders (fastText, MUSE dicts) + eval (BLI, CSLS)
gate_graft/    GW alignment + reliability gate + feasibility run   (LOCAL, no GPU training)
byte_embed/    byte student + distillation + robustness + Colab notebook + cloud scripts
scripts/       check_env.py, download_data.py
data/          downloaded vectors/dictionaries (gitignored)
results/       run outputs (gitignored)
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
Scaffolding + runnable scripts. **No experiments have been run yet.**
