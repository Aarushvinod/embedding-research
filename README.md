# embedding-research

Two embedding research threads on the **subword bottleneck** in multilingual sentence embeddings:

1. **ByteEmbed** (`byte_embed/`) — *eliminate* the vocabulary: a tokenizer-free **byte-level** (ByT5)
   student distilled from a frozen **SONAR** teacher, compared head-to-head with an identically-trained
   **subword** (mT5) student on **low-resource** languages. The clean question: at matched compute, is
   removing the 250k vocab table a better **parameter allocation** for low-resource retrieval?
   *Training-heavy — runs on Colab A100.*
2. **GATE-GRAFT** (`gate_graft/`) — *extend* the vocabulary: bitext-free cross-lingual alignment of a
   frozen embedder via Gromov–Wasserstein structure-matching + a per-token reliability gate.
   *Runs fully on a laptop (no training).* Kept in its own folder for reuse.

## Results (start here)
- **`PAPER.md`** — the curated, paper-ready narrative for both threads + honest scope.
- **`byte_embed/RESULTS.md`** — the ByteEmbed study design, *validated dataset sizes*, the measured
  tokenization-tax numbers, and prior (superseded) feasibility results.
- **`RESULTS.md`** — GRAFT detailed/reference results + audit.
- **`figures/`** — figures (PNG); regenerate with `python gen_figures.py`. Raw run JSON is under
  `results/` (gitignored).

**Headline.** GRAFT recovers 43–85% of a trained multilingual model bitext-free but is resource-bounded.
ByteEmbed is re-scoped to a clean SONAR-teacher low-resource study (results pending); its motivation is
**parameter allocation, not lower cost** — byte's UTF-8 sequence cost is actually *higher* than subword
for non-Latin scripts (see `PAPER.md` Part III).

## Layout
```
PAPER.md            curated paper-ready narrative (both threads) — START HERE
RESULTS.md          GRAFT detailed/reference results + audit
gen_figures.py      builds figures/ from results/*.json
figures/            figures (PNG)
common/             shared eval (SIB, STS, BLI, CSLS) + GRAFT data loaders (fastText, MUSE)
byte_embed/         the SONAR low-resource byte-vs-subword study (+ RESULTS.md, README.md)
  run_lowresource.py  orchestrator   teachers.py  SONAR(+LaBSE)   eval_mteb.py  uniform battery
  efficiency.py  tokenization tax    distill.py / model.py / data.py / miracl.py / config.py
  robustness.py  orthographic perturbations (deferred; kept for a follow-up run)
gate_graft/         GW/anchored alignment + reliability gate + graft + 21-lang study (LOCAL, no training)
notebooks/          byteembed_lowresource_a100.ipynb (the A100 runner)
scripts/            check_env.py, download_data.py
data/ results/      downloaded data / raw run JSON (gitignored)
```

## Quick start

### ByteEmbed (Colab A100)
Open `notebooks/byteembed_lowresource_a100.ipynb` in Colab, set runtime = A100, run top-to-bottom (do
the smoke cell first; it confirms SONAR-vs-LaBSE and validates the whole pipeline). Resumable across
sessions. CLI equivalent:
```bash
pip install -r requirements-cloud.txt && pip install -e .
python -m byte_embed.run_lowresource --smoke      # ~5-min self-test
python -m byte_embed.run_lowresource              # the full study (resumable)
```

### GATE-GRAFT (local — laptop is plenty)
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-local.txt && pip install -e .
python scripts/check_env.py
python -m gate_graft.run_feasibility --langs ca tr bn eu --k 2000
```
Inference + CPU optimal transport only. ~1 GB VRAM, minutes/language.

## Hardware notes
- Author machine: RTX 5070 Ti **Laptop** GPU (12 GB, Blackwell/sm_120). GATE-GRAFT runs here; ByteEmbed
  training targets a Colab A100 (80 GB).
- **Blackwell caveat:** local GPU work needs PyTorch for CUDA 12.8+ (cu128); see `requirements-cloud.txt`.
  Colab/cloud images handle this.

## Status
- **GATE-GRAFT:** 21-language study complete (`PAPER.md` / `RESULTS.md`); reproduce with
  `python -m gate_graft.run_matrix`.
- **ByteEmbed:** the SONAR low-resource study is **implemented and locally validated** (loaders, exact
  dataset sizes, the eval→figures contract); the 6-model A100 run is **pending**. Prior mE5-teacher
  feasibility results are preserved in `byte_embed/RESULTS.md` and marked superseded.
