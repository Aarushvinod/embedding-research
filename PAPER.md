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

**Paper-worthy results (run `results/byte_paper.json`; teacher = m-e5: Tatoeba 0.892, SIB 0.882,
SIB-rom 0.600, STS 0.637).**

| config | align | Tatoeba | Tat-rom | SIB | SIB-rom | STS | mean rob-gap |
|--------|------:|--------:|--------:|----:|--------:|----:|-------------:|
| byte-cosine | 0.917 | 0.121 | 0.041 | 0.791 | 0.600 | 0.387 | **+0.021** |
| byte-contrastive | 0.305 | **0.558** | 0.202 | 0.836 | 0.643 | 0.410 | −0.028 |
| **byte-both** | 0.767 | 0.514 | 0.178 | 0.826 | **0.644** | 0.419 | −0.007 |
| subword-mt5-both (iso-compute) | 0.707 | 0.240 | 0.058 | 0.782 | 0.536 | 0.520 | −0.021 |
| byte-random (control) | 0.014 | 0.023 | 0.022 | 0.601 | 0.467 | 0.122 | −0.003 |
| **byte-robust** (augmentation) | 0.747 | 0.383 | 0.159 | 0.818 | 0.626 | **0.424** | **+0.008** |
| byte-robust-queue (augment + MoCo) | 0.460 | **0.543** | **0.252** | 0.798 | 0.613 | 0.418 | −0.003 |

1. **Contrastive closes the retrieval gap.** Tatoeba **0.121 → 0.558** (cosine → contrastive),
   ~63% of the teacher (vs 14%). The gap was an objective problem (cosine makes embeddings aligned
   but not *discriminative*), now largely fixed. `byte-both` keeps it (0.514) with far better
   alignment (0.767 vs 0.305).
2. **A robustness↔retrieval tradeoff** (a finding, not a bug): cosine = robust (+0.021) but can't
   retrieve (0.12); contrastive = retrieves (0.56) but loses the robustness edge (−0.028); `both`
   balances. The earlier "byte more robust on everything" was a *cosine-only artifact* of
   over-smooth embeddings.
3. **Iso-compute byte > subword.** At matched compute/recipe, `byte-both` beats the mt5 subword
   student on retrieval (0.51 vs 0.24), classification (0.83 vs 0.78), and **romanized** tasks
   (SIB-rom 0.64 vs 0.54; Tat-rom 0.18 vs 0.06) — though subword wins STS (0.52 vs 0.42). Part of
   byte's edge: it spends parameters on the transformer, not a 250k-row subword embedding table.
4. **Downstream robustness win:** `byte-both` **beats the teacher on romanized SIB-200**
   (0.644 vs 0.600) while nearly matching it clean (0.826 vs 0.882) → the robustness translates to
   a real downstream classification gain, *even after* the contrastive objective.
5. **Robustness is within-script.** Byte > teacher on all 8 languages for typos/diacritics/case/
   swap/delete/keyboard (uniform, +0.002…+0.013, CIs exclude 0), but byte **loses on romanization**
   (−0.087; non-Latin transliteration changes every byte) and punctuation (−0.009).
6. **Random control:** alignment (0.014) and retrieval (0.023) are ≈0 untrained → 100% from
   distillation. (Classification has a random-features floor of 0.60 — SIB is easy enough that even
   random byte features classify; so the causal control is cleanest for alignment/retrieval.)
7. **Breaking the tradeoff (partial, honest).** Augmentation-consistency (`byte-robust`) is the only
   contrastive-trained config with **positive** robustness (+0.008): it pushes the robustness↔retrieval
   frontier outward and **fixes the romanization failure** (romanize −0.087 → **+0.017**), giving
   positive gaps on **7/8** perturbations incl. *held-out* ones (diacritics/swap/delete — generalization,
   not memorization). It's the **best-balanced** config (uniform robustness + best STS 0.424 + strong
   classification), at a retrieval cost (0.383 vs both's 0.514). The MoCo queue (`byte-robust-queue`)
   instead **scales retrieval to 0.543** (best byte config) but trades the robustness back. → the
   tradeoff is a **frontier you can push and re-balance, not a free lunch you escape.**
8. **Toward a better retriever (modest).** The queue lifts retrieval 0.514 → 0.543 (~61% of the
   teacher) — scaling negatives helps, but the byte student is not yet SOTA-competitive at feasibility
   scale; more negatives/steps/data is the lever, and the positive slope suggests headroom.

---

## Part III — fertility analysis (the hypothesized unifier did NOT hold; reported honestly)
Teacher subword fertility (tokens/char): en 0.26, tr 0.24, ru 0.26, vi 0.27, sw 0.28, ar 0.30,
bn 0.32, hi 0.33 — highest on non-Latin scripts, as expected.

**The hypothesis "fertility → byte robustness gain" is REJECTED:**
- within-script byte gain vs fertility: **r = −0.18** (no relationship; the gain is roughly uniform,
  ~+0.007 across languages).
- romanization byte gap vs fertility: **r = −0.77** — high-fertility non-Latin scripts are exactly
  where byte *loses* (romanizing them rewrites every byte). Fertility predicts byte's romanization
  **vulnerability**, not a gain.

**Honest unifying thread (qualitative, not a clean law):** subword tokenization limits multilingual
embeddings two ways — *coverage* (GRAFT) and *robustness* (ByteEmbed) — and **non-Latin /
high-fertility scripts are the persistent hard frontier for BOTH** (GRAFT recovers least there;
byte struggles with romanization there). The clean quantitative "fertility → byte gain" mechanism
I hoped would unify the two methods does not exist; the link is thematic, not mechanistic.

---

## Limitations & honest scope (state up front in the paper)
- **GRAFT** is a bitext-free translate-test (not a new algorithm), **resource-bounded** (fails on
  the low-resource tail where fastText collapses).
- **ByteEmbed** retrieval is competitive via contrastive (Tatoeba 0.51–0.54, ~58–61% of the teacher)
  but below it. Cosine-only robustness is within-script and *loses* on romanization; **augmentation
  (byte-robust) fixes the romanization loss** (−0.087 → +0.017) and makes robustness uniform across
  all 8 perturbations — but at a retrieval cost (0.51 → 0.38), so it re-balances the frontier rather
  than escaping it. byte is still below the teacher on STS and absolute retrieval.
- **Novelty is still empirical, not methodological.** The new objectives (augmentation-consistency,
  MoCo queue) are standard techniques; they make the robustness story more *complete* (it now covers
  the script-change case) and map the frontier, but they are not a new mechanism. The contribution
  remains "a rigorous study," now stronger — not a methods paper.
- **Combined-paper verdict (honest):** the fertility analysis did NOT quantitatively unify the two
  methods (Part III), so the pairing is *thematic, not mechanistic*. For a top venue, **ByteEmbed is
  the stronger standalone** (contrastive retrieval fix + iso-compute byte>subword + within-script
  robustness + the augmentation romanization fix + downstream classification win); GRAFT is best
  used as the training-free context/baseline. My recommendation: **lead with ByteEmbed**; fold
  GRAFT in only if a reviewer-proof framing for the pairing emerges.
- **No single close prior; cite honestly, don't claim a "rebuttal."** The work is an intersection
  of disjoint lines (multilingual distillation = Reimers & Gurevych 2020; byte-LM distillation =
  Bolmo/ALM, generative; char/UGC robustness = c-RoLASER, *monolingual English, character-CNN,
  UGC benchmarks — not ours*). RoLASER's noisy→clean-teacher recipe is the **method ancestor** of
  `byte-robust`, so that experiment is *less* novel, not a rebuttal of a negative.
- Feasibility scale (≤2k steps, ≤21 langs, byt5/mt5-small). A full paper still needs: iso-compute
  *curves* (matched FLOPs, not just matched recipe), MMTEB/MIRACL, scale (the contrastive retrieval
  gap may keep closing with more steps/negatives), and careful positioning vs RoLASER + Bolmo/ALM.

## Full experimental setup (methods section)
- **Teacher:** `intfloat/multilingual-e5-base` (frozen, 768-d).
- **GRAFT student:** frozen `intfloat/e5-base-v2`; fastText-157 reps; anchored Procrustes+CSLS.
- **ByteEmbed student:** `google/byt5-small` (byte) / `google/mt5-small` (subword baseline);
  mean-pool + linear projection to 768-d; cosine / InfoNCE(τ=0.05) / both objectives.
- **Data:** Wikipedia (distillation), OPUS-100 / Tatoeba / FLORES (retrieval), SIB-200
  (classification), STS22 (STS) — all public.
- **Stats:** bootstrap 95% CIs on all headline gaps; random-init controls on both methods.
