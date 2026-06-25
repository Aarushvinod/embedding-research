# Beyond the Subword Bottleneck: Tokenizer-Flexible Multilingual Sentence Embeddings

**This file is the curated, paper-ready narrative.** Detailed/reference results and the experimental
audit live in `RESULTS.md` (GRAFT) and `byte_embed/RESULTS.md` (ByteEmbed); raw run JSON is under
`results/` (gitignored). The **ByteEmbed thread has been re-scoped** to a clean SONAR-teacher
low-resource study — its model numbers are **pending the A100 run**, so that part is stated as design +
hypotheses + the (already-measured) tokenization analysis, with prior feasibility results marked
preliminary.

---

## Thesis
Multilingual sentence embedders are bottlenecked by their **subword tokenizer** in two ways:
1. **Coverage** — a fixed, English-centric vocabulary cannot represent words in unseen languages.
2. **Fragmentation** — subword segmentation fragments low-resource / non-Latin scripts much more than
   English (a per-language "tokenization tax").

We study two *tokenizer-flexible* remedies at opposite ends of the design space:
- **GRAFT** — *extend* the vocabulary: training-free, bitext-free grafting of word-aligned embedding
  rows into a frozen English encoder.
- **ByteEmbed** — *eliminate* the vocabulary: distill a multilingual teacher into a **byte-level**
  (ByT5) student, removing tokenization entirely.

## Narrative arc
1. The subword bottleneck (coverage + fragmentation), quantified by fertility / the tokenization tax.
2. **GRAFT** — a training-free remedy that works for mid/high-resource languages but is fundamentally
   **resource-bounded**; this motivates *removing* the tokenizer rather than patching it.
3. **ByteEmbed** — a trained, tokenizer-free remedy, studied as a **clean iso-compute, low-resource,
   parameter-allocation** question: at matched teacher/recipe/budget, does byte beat an identically-
   trained subword student on low-resource retrieval?
4. **Honest scope** up front: GRAFT is a bitext-free translate-test, resource-bounded; ByteEmbed's case
   is *parameter allocation*, **not** that byte is cheaper (it is not — see Part III).

---

## Part I — GRAFT (training-free vocabulary extension)

**Setup.** Frozen `e5-base-v2` (English). Bitext-free alignment of target fastText → English fastText
(anchored Procrustes + CSLS self-learning), a learned lift into e5's input-embedding space, grafted as
new tokenizer rows; the frozen body composes. Eval: OPUS-100 target→en retrieval P@1; `multilingual-e5-
base` topline; recovery = graft/topline. 21 languages / 6 scripts. Bootstrap 95% CIs, random-init
control, leakage audit.

**Paper-worthy results.**
- **Cross-lingual retrieval recovery 43–85%** (mid/high-resource): ca 78%, de/id 69%, ru 52%, tr 55%,
  bn 46%, ar 44%, hi 43%.
- **Resource-bounded, not script-bounded:** in *every* script the high-resource language works and the
  low-resource one craters (Amharic/Georgian ≈ 0). Performance tracks fastText quality.
- **Monolingual target quality:** SIB-200 recovers **85–95%** of a trained multilingual model; STS
  **63–76%** → real-but-coarse in-language structure.
- **Controls:** random-init ≈ floor (CIs non-overlapping); a LaBSE-bridge variant rescues weak languages
  (uk +77%, fa +74%) → the *bridge/representation* is the limiter, the graft mechanism is sound.
- **Honest framing:** GRAFT ≈ a bitext-free, embedding-space *translate-test* (closest prior: GiBERT-
  style input injection). Not algorithmically novel; the value is the training-free recipe and the
  resource-bound characterization.

---

## Part II — ByteEmbed (tokenizer-free distillation), re-scoped

**The clean question.** Hold the teacher, recipe, data, and budget fixed and vary **only the student's
tokenizer**: byte (`byt5`) vs subword (`mt5`). ByT5 *is* mT5 with the 250k vocab table reallocated into
transformer layers, so this is the cleanest possible single-variable A/B for "is the subword vocabulary
worth its parameters for low-resource multilingual retrieval?"

**Setup (`run_lowresource.py`).**
- **Teacher:** SONAR (NLLB-200, 1024-d) — covers all 200 FLORES languages → **no teacher-ceiling** on any
  chosen language (the constraint that previously forced language choices). LaBSE fallback.
- **Students:** `byt5` / `mt5` × {small, base, large} (6) — a **quality-per-parameter curve**, distilled
  from cached SONAR targets (one teacher pass; identical supervision for both).
- **Languages (9, uniform coverage):** Telugu, Tamil, Marathi, Amharic, Hausa, Kinyarwanda (low-resource;
  5 families, 5 scripts) + English / Mandarin / Arabic anchors. **Every language is scored on every task.**
- **Eval:** SIB-200 (classification) · Belebele (retrieval) · FLORES-1012 (parallel bitext) · STS
  (SemRel / Indic / C-MTEB) · MIRACL deep retrieval (en/zh/ar/te). Bootstrap 95% CIs; mE5/LaBSE baselines.

**Hypotheses.**
- **H1 — parameter allocation:** at matched compute, **byte ≥ subword on multilingual retrieval**, with
  the gap *largest on high-fertility low-resource languages*. The cross-size control is sharp: byte-small
  (219M) vs subword-base (278M) — byte winning with *fewer* total params would refute "byte wins because
  it's bigger."
- **H2 — efficiency, honest:** byte is more *parameter*-efficient (no 128–256M vocab table) but **costs
  more compute** (longer sequences). The claim rests on quality-per-parameter, not cost.

**Status:** model results pending the A100 run. Prior feasibility (mE5 teacher, ≤24 mid/high-resource
langs; superseded) gave the motivating signal: contrastive distillation closes the retrieval gap
(Tatoeba 0.12→0.56), iso-recipe byte > subword on retrieval (subword wins STS), and — at equal batch —
byte scales cleanly small→base including MIRACL. The current study removes that work's confounds (mE5
teacher-ceiling, batch-shrink, mid-resource skew).

---

## Part III — the tokenization tax (measured; the motivation, reported honestly)

On the parallel FLORES-1012, tax = tokens(lang)/tokens(English) for the **same content**:

| | Telugu | Tamil | Marathi | Amharic | Hausa | Kinyarwanda | Arabic | Chinese |
|---|---|---|---|---|---|---|---|---|
| **subword token tax** | 1.41 | 1.26 | 1.52 | 1.71 | 1.36 | 1.50 | 1.34 | 0.91 |
| **byte UTF-8 tax** | 2.68 | 3.19 | 2.69 | 1.71 | 1.07 | 1.13 | 1.60 | 0.92 |
| **byte seq vs subword** | 7.4× | 9.9× | 6.9× | 3.9× | 3.1× | 2.9× | 4.7× | 4.0× |

**The honest finding — "byte removes the tokenization tax" is FALSE for non-Latin scripts.** UTF-8
multibyte encoding makes byte *more* expensive there (Tamil byte-tax 3.19 vs subword 1.26; byte sequences
7–10× longer for Indic). Byte wins the cost axis only on Latin low-resource (Hausa, Kinyarwanda). The
defensible motivation is therefore **parameter allocation**: a subword model spends ≈87% of a small
encoder on a 250k vocab table that low-resource languages barely use; byte spends it on the transformer.
Whether that buys quality is the empirical question Part II answers.

---

## Limitations & honest scope (state up front)
- **GRAFT** is a bitext-free translate-test (not a new algorithm), resource-bounded (fails on the
  low-resource tail where fastText collapses).
- **ByteEmbed novelty is empirical, not methodological** — distillation (Reimers & Gurevych 2020),
  byte-level encoders (ByT5, CANINE), and noisy→clean robustness distillation (RoLASER) are all prior;
  the contribution is a *rigorous, uniform-coverage, low-resource, parameter-controlled* study and the
  first byte-level multilingual retriever evaluated this way.
- **Byte is not cheaper** — Part III shows a higher sequence/compute cost, worst for the non-Latin
  low-resource scripts we target. A fully rigorous comparison also needs a **matched-FLOPs** curve, not
  just matched recipe/token-views.
- **STS is byte's weak task and the rarest data** (Tamil 256 / Arabic 627 pairs) — reported with CIs and
  treated as secondary; retrieval (Belebele / FLORES / MIRACL) is the primary, uniformly-covered axis.
- The teacher is a **subword** model (SONAR), so this measures *byte-imitating-subword* at matched
  recipe, not intrinsic byte superiority; a from-scratch (no-teacher) arm would be needed for that claim
  and is deferred.

## Full experimental setup (methods)
- **Teacher:** `facebook/SONAR` text encoder (frozen, 1024-d; `source_lang` per FLORES code) — LaBSE
  (768-d) fallback.
- **Students:** `google/byt5-{small,base,large}` (byte) / `google/mt5-{small,base,large}` (subword);
  encoder-only mean-pool + linear projection to the teacher dim + L2; `max_bytes=256`.
- **Objective:** in-batch InfoNCE (τ=0.05) + alignment + MoCo queue 8192 + relational STS term; AdamW
  (lr 2e-4); batch 64 (equal at all sizes); steps small 10k / base 13k / large 15k; bf16 + grad-checkpoint.
- **Data:** equal max-min balanced Wikipedia (~42k/lang; floor = Kinyarwanda 42,621), distilled against
  cached teacher targets. **Eval:** SIB-200, Belebele, FLORES-1012, SemRel24STS / IndicCrosslingualSTS /
  C-MTEB STS, MIRACL — all public; bootstrap 95% CIs on headline gaps; iso-recipe subword control +
  mE5/LaBSE baselines.
