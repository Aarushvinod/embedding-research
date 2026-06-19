# ByteEmbed — feasibility

**Question this answers:** is a tokenizer-free **byte-level** dense retriever worth
pursuing? Specifically: can a byte student *reproduce* a strong multilingual subword
teacher via distillation, and is it **more robust to messy orthography** (the long-tail
reality) than the subword teacher?

**What it does:**
1. Freeze a multilingual subword teacher (`multilingual-e5-base`).
2. Train a byte-level student (`byt5-small` encoder + pooling + projection) to match the
   teacher's sentence embeddings (cosine distillation — no contrastive negatives, so it's
   memory-light).
3. Evaluate: (1) teacher-alignment cosine, (2) cross-lingual retrieval P@1 on FLORES
   (student vs teacher), (3) **orthographic-robustness stability** under romanization /
   diacritic loss / spelling noise (student vs teacher).

**Where to run:**
- **Colab A100 (primary):** open `colab_feasibility.ipynb`, set runtime = A100, Run All.
- **Cloud (fallback / reproducibility):** `cloud/setup.sh` on a RunPod/Vast.ai 4090 or
  A100 — see `cloud/README.md`. Full feasibility pass ≈ **$12** on a rented 4090.
- **Laptop smoke (12 GB, proof-of-life only):**
  `python -m byte_embed.run_feasibility --smoke` (tiny model budget; just checks it learns).

**Reading the result.** Encouraging = distillation loss drops + teacher-alignment cosine
high (student reproduces the teacher), cross-lingual P@1 in the teacher's ballpark, and
**student robustness > teacher robustness** under perturbation (especially `romanize` /
`spelling`). That justifies the full ByteEmbed study (iso-compute + fertility curves vs
subword retrievers). Discouraging = student can't match the teacher even on English, or it
is no more robust than the subword teacher.

**What this does NOT show.** It does not run the *iso-compute* comparison or the
fertility-vs-quality curve (the headline claims) — those need the larger multilingual
training + matched-FLOPs baselines described in the proposal. This is the cheap "is the
core mechanism alive?" test first.
