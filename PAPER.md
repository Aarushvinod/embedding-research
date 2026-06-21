# Beyond the Subword Bottleneck: Tokenizer-Flexible Multilingual Sentence Embeddings

**This file is the CURATED, paper-ready subset of results.** Full/supplementary results and the
experimental audit live in `RESULTS.md` (GRAFT) and `byte_embed/RESULTS.md` (ByteEmbed), and the
raw run JSON under `results/` (gitignored). Anything not here is reference-only.

---

## Thesis
Multilingual sentence embedders are bottlenecked by their **subword tokenizer** in two ways:
1. **Coverage** — a fixed, English-centric vocabulary cannot represent words in unseen languages.
2. **Robustness** — subword segmentation fragments under orthographic variation (romanization,
   diacritic loss, typos), and fragmentation is worst on low-resource / non-Latin scripts.

We study two *tokenizer-flexible* remedies at opposite ends of the design space, unified by a
single mechanism (tokenizer fertility):
- **GRAFT** — *extend* the vocabulary: training-free, bitext-free grafting of word-aligned
  embedding rows into a frozen English encoder, giving it new languages with **no training**.
- **ByteEmbed** — *eliminate* the vocabulary: distill a multilingual subword teacher into a
  **byte-level** (ByT5) student, removing tokenization entirely.

**Unifying claim:** subword **fertility** (tokens/char) predicts *both* GRAFT's difficulty *and*
ByteEmbed's robustness gain — the same fragmentation problem, attacked from two directions.

## Narrative arc
1. The subword bottleneck (coverage + robustness), quantified by fertility.
2. **GRAFT** — a training-free remedy that works for mid/high-resource languages but is
   fundamentally **resource-bounded**; this motivates removing the tokenizer rather than patching it.
3. **ByteEmbed** — a trained, tokenizer-free remedy: matches the teacher on clean classification,
   **exceeds it under orthographic noise**; a contrastive objective closes the retrieval gap; an
   iso-compute byte-vs-subword comparison isolates the byte effect.
4. **Honest scope** stated up front: GRAFT is *not algorithmically novel* (a bitext-free
   translate-test) and resource-bounded; ByteEmbed is a *robust classifier* whose retrieval
   quality depends on the contrastive objective and scale.

---

## Part I — GRAFT (training-free vocabulary extension)

**Setup.** Frozen `e5-base-v2` (English). Bitext-free alignment of target fastText → English
fastText (anchored Procrustes + CSLS self-learning), a learned lift into e5's input-embedding
space, grafted as new tokenizer rows; the frozen body composes. Eval: OPUS-100 target→en
retrieval P@1; `multilingual-e5-base` topline; recovery = graft/topline. 21 languages / 6 scripts.
Bootstrap 95% CIs, random-init control, leakage audit (§ RESULTS.md 10).

**Paper-worthy results.**
- **Cross-lingual retrieval recovery 43–85%** (mid/high-resource): ca 78%, de/id 69%, ru 52%,
  tr 55%, bn 46%, ar 44%, hi 43%. Absolute graft P@1 is strikingly flat (0.37–0.61) across scripts.
- **Resource-bounded, not script-bounded:** in *every* script the high-resource language works and
  the low-resource one craters (Amharic/Georgian ≈ 0). Performance tracks fastText quality.
- **Monolingual target quality:** SIB-200 classification recovers **85–95%** of a trained
  multilingual model (flat across scripts); STS **63–76%** → *real-but-coarse* in-language structure.
- **Controls:** random-init ≈ floor (graft CIs non-overlapping → significant); the LaBSE-bridge
  variant rescues weak languages (uk +77%, fa +74%) → the *bridge/representation* is the limiter,
  the graft mechanism is sound.
- **Honest framing:** GRAFT ≈ a bitext-free, embedding-space *translate-test* (closest prior:
  GiBERT-style input injection). Not algorithmically novel; the value is the training-free,
  bitext-free recipe and the resource-bound characterization.

| script | high-resource (recovery) | low-resource (recovery) |
|--------|--------------------------|--------------------------|
| Latin | de/id ~0.6 (69%) · ca (78%) | vi 0.31 (39%) |
| Cyrillic | bg/ru ~0.5 (53–61%) | uk 0.23 (26%) |
| Arabic | ar 0.44 (49%) | ur 0.11 (13%) |
| Devanagari | hi 0.39 (43%) | ne 0.11 (16%) |
| low-resource tier | — | am/ka ≈ 0 |

---

## Part II — ByteEmbed (tokenizer-free distillation)

**Setup.** ByT5-small byte student distilled from frozen `multilingual-e5-base` (768-d).
Objectives: **cosine** (alignment), **contrastive** (in-batch InfoNCE → discriminative), **both**.
**Iso-compute subword baseline:** `mt5-small`, identical recipe (isolates byte vs subword at
matched compute). 8 languages / 5 scripts (en/tr/sw/bn/ar/ru/hi/vi). Eval battery: teacher
alignment; **Tatoeba** bitext mining (standard retrieval, clean + romanized); **SIB-200**
classification (standard, clean + romanized); **STS22**; **8-perturbation** robustness with
bootstrap 95% CIs; random-init control; tokenizer-fertility analysis.

**Paper-worthy results.** _[FILL FROM run results/byte_paper.json — bxxctmokr]_
- Objective comparison (cosine vs contrastive vs both): retrieval gap closure + the
  robustness↔retrieval tradeoff.  _[pending]_
- Iso-compute byte (ByT5) vs subword (mt5) student: the byte effect at matched compute.  _[pending]_
- Standard benchmarks (Tatoeba / SIB / STS), clean vs romanized: downstream robustness.  _[pending]_
- 8-perturbation robustness vs teacher, CIs + per-language wins.  _[pending]_
- Random-init control (alignment causality).  _[pending]_

**Already established (feasibility, byte_embed/RESULTS.md):** byte student reproduces the teacher
(align 0.92 across 5 scripts); **beats the subword teacher on romanized SIB-200** (0.611 vs 0.600,
drops less under noise) → robustness translates downstream for classification; cosine-only
retrieval is weak (Tatoeba 0.21 vs 0.89) → the retrieval gap the contrastive objective targets.

---

## Part III — the unifying fertility analysis  _[FILL FROM run]_
Teacher subword fertility (tokens/char) per language, correlated with (a) GRAFT recovery and
(b) ByteEmbed's robustness gap. Hypothesis: higher fertility (more fragmentation) → harder GRAFT
*and* larger byte robustness advantage. _[pending teacher-fertility numbers + correlation]_

---

## Limitations & honest scope (state up front in the paper)
- **GRAFT** is a bitext-free translate-test (not a new algorithm) and **resource-bounded**: it
  fails on the genuine low-resource tail where fastText quality collapses.
- **ByteEmbed** is a robust *classifier*; competitive *retrieval* depends on the contrastive
  objective + scale (the run quantifies how far contrastive closes the gap).
- **Combined-paper risk (honest):** two methods under one theme is broader than a single focused
  contribution; a reviewer may prefer ByteEmbed alone. The fertility analysis is what justifies the
  pairing — if it doesn't show a clean correlation, split into two papers.
- Feasibility scale throughout (≤ a few thousand steps; ≤ 21 langs). The full paper needs the
  iso-compute curves, MMTEB/MIRACL, and the c-RoLASER head-to-head (byte_embed/RESULTS.md §6).

## Full experimental setup (methods section)
- **Teacher:** `intfloat/multilingual-e5-base` (frozen, 768-d).
- **GRAFT student:** frozen `intfloat/e5-base-v2`; fastText-157 reps; anchored Procrustes+CSLS.
- **ByteEmbed student:** `google/byt5-small` (byte) / `google/mt5-small` (subword baseline);
  mean-pool + linear projection to 768-d; cosine / InfoNCE(τ=0.05) / both objectives.
- **Data:** Wikipedia (distillation), OPUS-100 / Tatoeba / FLORES (retrieval), SIB-200
  (classification), STS22 (STS) — all public.
- **Stats:** bootstrap 95% CIs on all headline gaps; random-init controls on both methods.
