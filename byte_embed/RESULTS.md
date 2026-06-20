# ByteEmbed — feasibility results

Byte-level (tokenizer-free) ByT5-small student, cosine-distilled from a frozen multilingual
subword teacher (`multilingual-e5-base`, 768-d). Full suite run via
`byte_embed/run_feasibility.py` (base + random-init control + 4 ablations). Numbers below are
the local run (RTX 5070 Ti): base 2000 steps / ablations 1000 steps, batch 16, 8 languages
(en/tr/sw/bn/ar/ru/hi/vi, 5 scripts), 80k Wikipedia training sentences, 200 monolingual eval
sentences per language. Raw JSON gitignored under `results/`.

## Headline
- **The byte student reproduces the subword teacher** across all 5 scripts: teacher-alignment
  cosine **0.90–0.94 (mean 0.92)**.
- **It recovers ~66% of the teacher's cross-lingual P@1** on average (ru 0.89, ar/vi/tr
  0.72–0.78, bn/hi 0.41–0.43), up from 0.02–0.05 at 300 steps — so the gap closes with compute.
- **H2 holds decisively:** the byte student is MORE robust than the subword teacher on **all 8
  orthographic perturbations**, every gap's 95% CI excludes 0, and 7–8/8 languages win each.
  Largest gaps where it matters most: **romanize +0.063, diacritics +0.030** (transliteration /
  accent loss, which fragments a subword tokenizer on non-Latin scripts but not a byte encoder).
- **Random-init control is decisive:** an untrained byte student has align 0.010 and ~0
  robustness gap → the alignment is 100% from distillation and the robustness edge emerges with
  training (not a trivial architecture artifact).

## 1. Teacher alignment (student reproduces teacher) — base config
| en | tr | sw | bn | ar | ru | hi | vi | mean |
|----|----|----|----|----|----|----|----|------|
| 0.905 | 0.914 | 0.935 | 0.916 | 0.919 | 0.914 | 0.928 | 0.916 | **0.918** |

## 2. Cross-lingual retrieval P@1 (student / teacher)
| tr | sw | bn | ar | ru | hi | vi |
|----|----|----|----|----|----|----|
| 0.72/0.99 | 0.64/0.99 | 0.41/1.00 | 0.78/1.00 | 0.89/1.00 | 0.43/0.99 | 0.77/1.00 |

Student recovers ~66% of the teacher's cross-lingual P@1 on average; weakest on non-Latin bn/hi.

## 3. H2 — orthographic robustness (byte student vs subword teacher)
stability = cosine(clean, perturbed); higher = more robust. Mean over 8 langs, bootstrap 95% CI
on the pooled per-sentence gap, #langs where student > teacher.

| perturbation | student | teacher | gap | gap 95% CI | wins |
|--------------|--------:|--------:|------:|------------|-----:|
| romanize | 0.967 | 0.904 | +0.063 | [+0.060,+0.066] | 8/8 |
| diacritics | 0.996 | 0.967 | +0.030 | [+0.028,+0.032] | 8/8 |
| case | 0.998 | 0.983 | +0.015 | [+0.014,+0.016] | 7/8 |
| spelling | 0.998 | 0.984 | +0.014 | [+0.013,+0.015] | 8/8 |
| delete | 0.999 | 0.987 | +0.012 | [+0.011,+0.013] | 8/8 |
| punct | 0.996 | 0.984 | +0.011 | [+0.011,+0.012] | 8/8 |
| swap | 0.999 | 0.990 | +0.009 | [+0.009,+0.010] | 8/8 |
| keyboard | 0.999 | 0.993 | +0.006 | [+0.005,+0.007] | 7/8 |

All 8 gaps positive and significant. The byte student is near-invariant (~0.997) under noise
while staying discriminative (align 0.92, xP@1 up to 0.89) — so the stability is real, not
collapse.

## 4. Ablations (mean align · mean robustness gap over perts · #perts student>teacher)
| run | align | mean_gap | pos_perts |
|-----|------:|---------:|----------:|
| random_init (untrained control) | 0.010 | −0.000 | 6/8 |
| max_bytes=128 | 0.899 | +0.022 | 8/8 |
| objective=mse | 0.902 | +0.022 | 8/8 |
| pooling=max | 0.883 | +0.023 | 8/8 |
| student=byt5-base | 0.903 | +0.023 | 8/8 |

- **random_init**: align ≈ 0 and no robustness advantage → distillation is causal for both.
- **H2 gap robust to every ablation** (+0.022–0.023, 8/8) — not contingent on a config choice.
- **byt5-base ≈ byt5-small** — capacity is not the bottleneck at feasibility scale.

## Robustness note (the eval-harness fix)
An earlier Colab run distilled fine but returned an EMPTY eval: a single FLORES-parallel load
failed on the runtime and a bare `except` swallowed it, zeroing align/xP@1/robustness. The eval
was rebuilt to decouple the parts: the H2 robustness probe + teacher-alignment use per-language
MONOLINGUAL sentences (`load_mono`, Wikipedia→FLORES) that always load, so the eval can never
come back fully empty again; only cross-lingual P@1 needs parallel data (FLORES with
trust_remote_code+retries → OPUS-100 fallback) and degrades gracefully. Errors are surfaced.

## 5. Novelty (verified literature deep-dive)
- **Algorithmic novelty: LOW–MEDIUM.** A recombination of published parts — cosine embedding
  distillation (Reimers & Gurevych 2020), distill-into-a-different-tokenizer *embedder* (2026
  Turkish tokenizer-surgery paper), subword→byte distillation (Minixhofer ALM NeurIPS 2025;
  AI2 Bolmo) — but those byte-distillation works target *generative* LMs, not embedders.
- **Application novelty: MEDIUM–HIGH.** No raw-byte multilingual *retrieval embedder* exists;
  ByT5/CANINE are LMs, the T5 dual-encoder line (GTR) is subword. The intersection is open.
- **Empirical-finding novelty: HIGH**, and *contrarian*. The single closest prior work,
  **c-RoLASER (Nishimwe, Sagot & Bawden, LREC-COLING 2024, arXiv:2403.17220)** — a near-exact
  twin (sentence embeddings × orthographic robustness × frozen-teacher distillation × char-vs-
  subword student) — found the OPPOSITE: their character student "never outperforms" subword and
  failed to align to the teacher (cosine dist 0.05–0.13). **Our result overturns it:** byte >
  subword on 8/8 perturbations WITH tight 0.92 alignment, and the random-init control (0.01→0.92)
  directly answers their stated failure mode. ByteEmbed is best framed as a causally-controlled
  rebuttal of a published negative — which REQUIRES engaging c-RoLASER head-on.
- **Reportable:** workshop/short (MRL, RepL4NLP, Findings) as-is; full main-conference needs an
  explicit c-RoLASER head-to-head, standard benchmarks (MTEB/MIRACL/BEIR, not just P@1), more
  languages, and a downstream task where the robustness gain matters.
- Closest priors — overall: c-RoLASER (LREC-COLING 2024); method: 2026 Turkish tokenizer-surgery
  + cosine distillation; retrieval-robustness: CharacterBERT-DR (SIGIR 2022, arXiv:2204.00716).

## Caveats
- Feasibility scale (2000 steps, 80k sentences); NOT the iso-compute / fertility-curve study.
- The ~66% cross-lingual P@1 recovery is the weakness reviewers will probe first.
- A few 2026-stamped preprints cited above were confirmed to exist but need author-roster checks
  before citing in a manuscript.
- Absolute robustness gaps are modest outside romanize/diacritics (both encoders fairly stable).
- Non-Latin cross-lingual P@1 (bn/hi ~0.42) lags the Latin/Cyrillic langs.
- The random-init "robustness" baseline embeds near-randomly, so its stability is itself only
  weakly interpretable; the clean control claim is the alignment (0.01 → 0.92 from distillation).
