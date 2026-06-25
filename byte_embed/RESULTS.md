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

## 5. Standard benchmarks (byte student vs multilingual-e5 teacher) — the reality check
SIB-200 (classification), Tatoeba (bitext mining), STS22 (STS), scored clean and romanized.
The teacher's Tatoeba ~0.89 matches published m-e5, so the harness is sound.

| task | teacher | student | stu/tea |
|------|--------:|--------:|--------:|
| SIB-200 classification (clean) | 0.882 | 0.802 | 0.91 |
| SIB-200 classification (romanized) | 0.600 | **0.611** | 1.02 |
| Tatoeba bitext mining (clean) | 0.892 | **0.206** | **0.23** |
| Tatoeba bitext mining (rom src) | 0.295 | 0.059 | 0.20 |
| STS22 (Spearman) | 0.636 | 0.388 | 0.61 |

**Downstream robustness (clean → romanized):** SIB teacher 0.882→0.600 (drop 0.28) vs student
0.802→0.611 (drop 0.19) — student drops LESS *and ends up higher*. Tatoeba teacher 0.892→0.295
vs student 0.206→0.059 — student drops less in absolute terms but is near-floor either way.

**Revised, benchmark-grounded conclusion (this supersedes the rosy custom-eval read):**
- **ByteEmbed is a robust CLASSIFIER, not (yet) a competitive RETRIEVER.** SIB clean recovers 91%
  of the teacher and **beats it under romanization** — the H2 robustness translates downstream
  for classification.
- **Cross-lingual RETRIEVAL is weak: Tatoeba 0.206 vs 0.892 (23%).** The standard benchmark
  reveals this and CORRECTS the custom FLORES/OPUS P@1 (0.66), which was optimistic (Tatoeba is
  harder: 1000 candidates, out-of-domain). align-cosine 0.92 is enough for coarse classification
  but NOT for 1-in-1000 nearest-neighbor matching.
- **Nuances the c-RoLASER positioning:** we overturn them on classification robustness, but
  partially confirm their concern (the distilled byte student doesn't fully match the teacher on
  the hardest task, retrieval) at feasibility scale.

## 6. Novelty (verified literature deep-dive)
- **Algorithmic novelty: LOW–MEDIUM.** A recombination of published parts — cosine embedding
  distillation (Reimers & Gurevych 2020), distill-into-a-different-tokenizer *embedder* (2026
  Turkish tokenizer-surgery paper), subword→byte distillation (Minixhofer ALM NeurIPS 2025;
  AI2 Bolmo) — but those byte-distillation works target *generative* LMs, not embedders.
- **Application novelty: MEDIUM–HIGH.** No raw-byte multilingual *retrieval embedder* exists;
  ByT5/CANINE are LMs, the T5 dual-encoder line (GTR) is subword. The intersection is open.
- **Empirical-finding novelty: MEDIUM (corrected — was overstated).** There is NO single close
  prior; the work sits at an intersection of disjoint lines: multilingual sentence-embedding
  distillation (Reimers & Gurevych 2020, subword→subword), byte-level distillation (Bolmo / ALM,
  *generative* LMs), and char-level robustness (c-RoLASER). No one occupies the multilingual ×
  byte × retrieval intersection.
- **On c-RoLASER specifically (correcting an earlier overstatement).** Nishimwe, Sagot & Bawden,
  LREC-COLING 2024, arXiv:2403.17220, "Making Sentence Embeddings Robust to User-Generated Content"
  is **monolingual English**, distills **LASER2**, and its c-RoLASER student is a **Character-CNN
  (CharacterBERT-style), NOT byte-level**; it evaluates **UGC** robustness (MultiLexNorm, RoCS-MT,
  FLORES†) + a few MTEB tasks — **none of our multilingual benchmarks (Tatoeba/SIB-200/STS22)**.
  So it is NOT a "near-exact twin we overturn": there is no head-to-head (we never ran their UGC
  eval; they never ran ours). It IS relevant in two narrower ways: (a) its recipe (distill noisy→
  clean-teacher) is exactly our `byte-robust` augmentation objective → **prior art for that method**,
  making byte-robust *less* novel, not a rebuttal; (b) their char-level negative is a data point our
  byte-level multilingual result contrasts with, in a different setting. Do NOT frame the paper as
  "overturning a published negative."
- **Reportable:** workshop/short (MRL, RepL4NLP, Findings) as-is; full main-conference needs
  standard benchmarks (MTEB/MIRACL/BEIR), more languages, an iso-compute *curve*, and a real
  downstream task — plus careful positioning vs the distillation-for-robustness line (RoLASER) and
  the byte-LM-distillation line (Bolmo/ALM), since the method is a recombination of known parts.
- No single close prior (intersection of disjoint lines): multilingual sentence-embedding
  distillation (Reimers & Gurevych 2020, subword→subword); byte-LM distillation (Bolmo/ALM,
  generative); char/UGC robustness (c-RoLASER, English). RoLASER is the **method ancestor** of
  `byte-robust` (noisy→clean-teacher distillation); CharacterBERT-DR (SIGIR 2022) is the closest
  retrieval-robustness work.

## 7. FULL-PAPER RUN — objectives + iso-compute + nuanced robustness (supersedes §1–5 cosine-only)
Single orchestrated run (`run_paper.py`): byt5-small {cosine, contrastive, both} + mt5-small
subword baseline (iso-compute) + random control; 8 langs, 2000 steps, batch 16. Teacher m-e5:
Tatoeba 0.892, SIB 0.882, SIB-rom 0.600, STS 0.637.

| config | align | Tatoeba | Tat-rom | SIB | SIB-rom | STS | mean rob-gap |
|--------|------:|--------:|--------:|----:|--------:|----:|-------------:|
| byte-cosine | 0.917 | 0.121 | 0.041 | 0.791 | 0.600 | 0.387 | +0.021 |
| byte-contrastive | 0.305 | 0.558 | 0.202 | 0.836 | 0.643 | 0.410 | −0.028 |
| byte-both | 0.767 | 0.514 | 0.178 | 0.826 | 0.644 | 0.419 | −0.007 |
| subword-mt5-both | 0.707 | 0.240 | 0.058 | 0.782 | 0.536 | 0.520 | −0.021 |
| byte-random | 0.014 | 0.023 | 0.022 | 0.601 | 0.467 | 0.122 | −0.003 |

robustness gap by perturbation (byte-both vs teacher, 95% CI · langs won):
diacritics +0.015 [+.013,+.016] 7/8 · spelling +0.006 8/8 · swap +0.005 8/8 · delete +0.006 8/8 ·
case +0.007 7/8 · keyboard +0.002 6/8 · **romanize −0.087 [−.095,−.079] 2/8** · punct −0.009 0/8

Findings: (1) contrastive closes the retrieval gap (Tatoeba 0.121→0.558, ~63% of teacher) — §5's
weakness was an OBJECTIVE problem; (2) a robustness↔retrieval tradeoff (cosine robust/can't
retrieve; contrastive retrieves/loses robustness; both balances) — the §1–4 "byte more robust on
all 8 perts" was a COSINE-ONLY artifact; (3) iso-compute byte > subword on retrieval/classification/
romanized tasks (STS the exception); (4) byte robustness is WITHIN-SCRIPT (6/8); it LOSES on
romanization (script change) and punctuation; (5) the downstream win survives contrastive
(byte-both beats teacher on romanized SIB, 0.644 vs 0.600); (6) fertility does NOT predict the byte
gain (within-script r=−0.18); it predicts romanization vulnerability (r=−0.77). The fertility
"unifier" is rejected.

## 8. Breaking the tradeoff — augmentation-consistency + MoCo queue
Two new objectives (`run_paper --only byte-robust byte-robust-queue`), teacher unchanged:

| config | align | Tatoeba | Tat-rom | SIB | SIB-rom | STS | mean rob-gap |
|--------|------:|--------:|--------:|----:|--------:|----:|-------------:|
| byte-robust (augmentation) | 0.747 | 0.383 | 0.159 | 0.818 | 0.626 | 0.424 | +0.008 |
| byte-robust-queue (augment+MoCo) | 0.460 | 0.543 | 0.252 | 0.798 | 0.613 | 0.418 | −0.003 |

byte-robust robustness by perturbation (vs teacher; held-out = diacritics/swap/delete/punct):
diacritics +0.020 (6/8) · romanize +0.017 (4/8) · spelling +0.009 (8/8) · keyboard +0.004 (7/8) ·
swap +0.007 (8/8) · delete +0.007 (8/8) · case +0.014 (7/8) · punct −0.012 (0/8)

Findings: (1) augmentation FIXES romanization — byte-both romanize −0.087 → byte-robust **+0.017**;
positive on 7/8 perturbations incl. held-out diacritics/swap/delete (generalization). (2) byte-robust
is the only contrastive-trained config with positive mean robustness (+0.008) that also retrieves
(0.383) — it pushes the robustness↔retrieval frontier outward; best-balanced (uniform robustness +
best STS 0.424). (3) the MoCo queue scales retrieval (0.514 → **0.543**, best byte config) but trades
robustness back → the tradeoff is a frontier you re-balance, not escape. (4) methods are standard
(augmentation, MoCo) → contribution stays empirical (now covers the script-change case), not methodological.

## 9. Scale-up + iso-compute curve + strong baselines (12 langs, 3000 steps, objective=both)
`run_scale.py`: byte (byt5) vs subword (mt5) students at two sizes + mE5/LaBSE baselines.

| model | params | SIB | Tatoeba | STS |
|-------|------:|----:|--------:|----:|
| byte-small | 219M | 0.818 | **0.590** | 0.453 |
| subword-small (mt5) | 147M | 0.787 | 0.277 | 0.551 |
| byte-base | 416M | 0.811 | **0.583** | 0.480 |
| subword-base (mt5) | 278M | 0.842 | 0.362 | 0.559 |
| mE5 teacher (baseline) | 278M | 0.882 | 0.892 | 0.661 |
| LaBSE (baseline) | 472M | 0.839 | 0.937 | 0.636 |

Findings: (1) **byte ≫ subword on retrieval** at both sizes (+0.22–0.31 Tatoeba); byte-small (219M)
beats subword-**base** (278M), 0.590 vs 0.362 — the parameter-allocation argument (byte spends
params on the encoder, not a 250k vocab table). (2) **subword > byte on STS** (~0.55 vs ~0.47) —
byte's weakness is fine-grained similarity. (3) classification is a wash (byte better small,
subword better base). (4) both students are **below the SOTA baselines** (mE5/LaBSE Tatoeba
0.89–0.94) — expected, those are trained on billions of pairs; the apples-to-apples claim is
byte-student vs subword-student. (5) **scaling caveat:** base models used batch 8 vs small's 16
(half the token-views), so byte-base is undertrained — the retrieval win holds at both sizes but a
clean scaling claim needs matched-token training. **Sharper contribution = byte retrieval + robustness
via parameter allocation** (controlled iso-compute study + first byte-level multilingual retriever).

## 10. A100 fair-scaling re-run — equal batch, real MIRACL (PARTIAL: byte-small + byte-base only)
`run_cloud.py` (phase=scaling) on an 80 GB A100, addressing §9's batch-8-vs-16 confound directly:
byt5 {small, base, large} vs mt5 {small, base, large} at **equal batch 64** for every size, bigger
models get more steps (small 10k / base 13k / large 15k), `objective=both` + `rel_weight=1.0`, AdamW,
24 langs (SIB/Tatoeba/STS), mE5-base teacher. **Adds MIRACL** real passage retrieval
(`mteb/MIRACLRetrieval`, 8 langs, 250 queries, ~20k-distractor pool; nDCG@10). The runtime
disconnected three times and **died mid-byte-large (step 4500/15000)** before its first checkpoint,
so **only byte-small and byte-base completed** — byte-large, the subword (mt5) students, and the
mE5/LaBSE baselines are pending a re-run.

| model | steps×batch (views) | SIB-200 | Tatoeba | STS22 | **MIRACL nDCG@10** | peak |
|-------|--------------------:|--------:|--------:|------:|-------------------:|-----:|
| **byte-small** | 10000×64 (640k) | 0.8227 | 0.8068 | 0.4695 | **0.5835** | 14.2 GB |
| **byte-base**  | 13000×64 (832k) | 0.8391 | 0.8558 | 0.4886 | **0.6963** | 18.8 GB |
| byte-large | 15000×64 | — incomplete (died at step 4500/15000, no eval) | | | | 20.3 GB |

MIRACL nDCG@10 per language (real pool, 250q / ~20k distractors):

| | sw | bn | hi | te | th | ko | fi | ar | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byte-small | .522 | .502 | .453 | .428 | .601 | .566 | .842 | .754 | **.584** |
| byte-base  | .632 | .636 | .559 | .546 | .694 | .731 | .899 | .873 | **.696** |

Findings (partial — 2 of 6 students, no baselines):
1. **Clean monotonic scaling small→base on every axis once batch is equalized:** SIB +0.016,
   Tatoeba +0.049, STS +0.019, and **MIRACL +0.113 (+19% relative)**; byte-base beats byte-small on
   **all 8** MIRACL languages individually. This **resolves §9's confound** — the local run's batch-8
   byte-base looked flat/undertrained, so scaling appeared absent; at equal batch 64 it is clearly
   present. (§9 point 5 asked for "matched-token training"; this is it.)
2. **Real retrieval now reportable** — first MIRACL nDCG@10 for the byte student across 8 scripts
   (Swahili→Arabic); byte-base 0.696 mean, strong on fi/ar (0.87–0.90), weaker on Indic/Telugu
   (0.45–0.56). (The MIRACL pipeline was switched to the parquet-native `mteb/MIRACLRetrieval` after
   `datasets≥4.0` dropped script-based loaders, which is why §9 carried no MIRACL.)
3. **Much higher Tatoeba than §9** (0.81–0.86 vs 0.59) from 10k–13k steps + batch 64 + AdamW +
   relational loss — approaching the LaBSE/mE5 baselines (0.89–0.94), though those baselines were not
   re-run here for a same-pool comparison.
4. **Incomplete by design-failure:** no byte-large, no subword iso-compute pair, no baselines this
   run → the size curve stops at "base" and the byte-vs-subword iso-compute claim is **pending the
   re-run**. Numbers are a single fair run (no seed variance yet).

## Caveats
- §10 is a PARTIAL run (byte-small + byte-base only; byte-large/subword/baselines pending a re-run
  after 3 Colab disconnects killed the runtime mid-byte-large at step 4500/15000).
- Feasibility scale (≤3000 steps, ≤96k sentences); below-SOTA in absolute terms (training scale).
- §1–5 used cosine-only distillation; §7–9 are the authoritative, current results.
- The ~66% cross-lingual P@1 recovery is the weakness reviewers will probe first.
- A few 2026-stamped preprints cited above were confirmed to exist but need author-roster checks
  before citing in a manuscript.
- Absolute robustness gaps are modest outside romanize/diacritics (both encoders fairly stable).
- Non-Latin cross-lingual P@1 (bn/hi ~0.42) lags the Latin/Cyrillic langs.
- The random-init "robustness" baseline embeds near-randomly, so its stability is itself only
  weakly interpretable; the clean control claim is the alignment (0.01 → 0.92 from distillation).
