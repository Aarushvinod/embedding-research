# Finalized retrieval experiment — byte vs subword, BGE-M3 teacher

The locked design for the paper's retrieval/QA axis. vs the completed SONAR study, the go-forward run
changes the **teacher** (SONAR → BGE-M3), the **objective** (retrieval-only InfoNCE), and the
**language set** (deep-research-verified, below). The core comparison — byte vs subword — is entirely
WITHIN this run: both students share the teacher, targets, data, recipe, and evaluation, so the
tokenizer remains the only manipulated variable.

## Languages (finalized 2026-07 via adversarially-verified deep research)

5 low-resource (Joshi class 0–2) + 3 high-resource anchors; every language has a deep-retrieval
benchmark with real human queries and public relevance labels:

| Lang | Joshi | Deep retrieval | Axis | Notes |
|---|---|---|---|---|
| Telugu (te) | 1 | MIRACL dev (828q/518k psgs) + Mr.TyDi + IndicQA (secondary) | monolingual | |
| Swahili (sw) | 2 | MIRACL dev (482q/132k) + Mr.TyDi | monolingual | new |
| Yoruba (yo) | 2 | MIRACL dev (119q/49k, "surprise language") | monolingual | new; NOT in XLM-R (see caveat) |
| Amharic (am) | 2 | 2AIRTC (240 topics, peer-reviewed) + Amharic-PR (community) | monolingual | first-party verified |
| Hausa (ha) | 2 | CIRAL Test A (80q/1.4k judgments, 715k news psgs) | **cross-lingual** (en query → ha passage) — flagged | |
| English (en) | 5 | MIRACL (799q/32.9M) + Mr.TyDi | monolingual | anchor |
| Chinese (zh) | 5 | MIRACL (393q/4.9M) | monolingual | non-Latin anchor |
| Arabic (ar) | 5 | MIRACL (2,896q/2.1M) + Mr.TyDi | monolingual | non-Latin anchor |

**Dropped:** Kinyarwanda (no deep-retrieval benchmark exists anywhere — AfriQA's gold passages are
English/French pivot text, not Kinyarwanda), Tamil + Marathi (their only benchmark, IndicQA, has a
~250-doc pool — not deep). **AfriQA finding:** `masakhane/afriqa-gold-passages` DOES ship `context`
passages, but they are English/French Wikipedia text — usable only as reversed cross-lingual
(African query → pivot passage) coverage for bem/fon/ibo/kin/twi/wol/zul; not wired.

**Yoruba caveat (state in the paper):** yo is not in XLM-R/CC-100, the BGE-M3 backbone, so the teacher
signal is weakest there. Both students inherit the same weakened targets — the byte-vs-subword
comparison stays internally fair — and both ByT5/mT5 saw yo in mC4 pretraining. Training data is fine
(91k usable wiki paragraphs, measured).

## Design (locked)

| Axis | Setting |
|---|---|
| Students | `google/byt5-{small,base,large}` vs `google/mt5-{small,base,large}` (encoder-only + attn pool + 1024-d head) |
| Teacher | **BGE-M3** (`BAAI/bge-m3`, retrieval-trained, 1024-d); `me5-large` wired as the ablation teacher |
| Objective | **Retrieval-only: pure InfoNCE** (τ=0.05, MoCo queue 8192) — `objective='contrastive'`, `rel_weight=0` (the alignment add-on and the STS-motivated relational term are dropped) |
| Optimizer | AdamW lr 2e-4, batch 64, **50k steps for every model (iso-step)**, bf16 |
| Early stop | `patience` windows without window-avg loss improving > `min_delta`. **Default 0 = off** (exact iso-step). If enabled, `steps` is a cap; realized `steps_run` is saved and must be reported |
| Data | 8 languages, balanced max-min Wikipedia sentences (~42k/lang, ~336k total) |
| Targets | BGE-M3, precomputed once, cached as `teachertargets_bge-m3_8langs_*` |
| Pooling | `attn` for byte AND subword (fair) |
| Checkpoints | `{name}_attn_bge-m3.pt` — per-teacher namespace, never resumes from SONAR checkpoints |
| Baselines | mE5-base, LaBSE (same battery, same pools) |

## Evaluation (retrieval-only)

| Benchmark | Pool | Languages | Axis / metric |
|---|---|---|---|
| Belebele | 488 | all 8 | shallow passage retrieval, nDCG@10 |
| FLORES bitext | 1,012 | all 8 | cross-lingual sentence retrieval, P@1 |
| MIRACL (dev) | 20k rerank pools | en zh ar te sw yo | deep monolingual, nDCG@10 + R@100 |
| Mr.TyDi | 20k rerank pools | te sw | deep monolingual QA, nDCG@10 + R@100 |
| 2AIRTC | 12.6k (full) | am | deep ad-hoc IR (peer-reviewed), nDCG@10 + R@100 |
| Amharic-PR | 20k | am | deep monolingual (community), nDCG@10 + R@100 |
| CIRAL Test A | 715k stream → pool | ha | **cross-lingual, flagged**, nDCG@10 + R@100 |
| IndicQA | ~250 | te | small-pool secondary |

SIB and STS are dropped (`eval_battery` computes them only on request via `tasks=`). AfriCLIRMatrix
remains wired but off the default battery (superseded by CIRAL for ha; am has two monolingual sets).
MIRACL evaluation must use dev (test qrels are held out); report CIs for yo (119 queries) and te's
judgment-sparse dev (~2 judgments/query).

## Predictions (written before running)

1. Both students' deep-retrieval scores rise substantially vs the SONAR run (teacher ceiling lifted;
   objective purely discriminative).
2. Byte > subword persists at every size on the deep monolingual benchmarks.
3. byte-small vs subword-large: byte-small stays ahead on deep retrieval (the cross-size headline).
4. FLORES bitext DROPS vs the SONAR run (no bitext teacher, no alignment term) — report as the
   specialization trade-off, not a regression.
5. yo scores land low for both students (teacher coverage), with the byte−subword gap still positive.

## How to run

Colab: `notebooks/byteembed_retrieval_a100.ipynb` top-to-bottom (smoke → parallel run → baselines).
CLI: `python -m byte_embed.run_lowresource --teacher bge-m3 --pooling attn --steps 50000 --out results/retrieval_bgem3.json`

Results land in `results/retrieval_bgem3.json`. The first eval builds the retrieval pools (CIRAL
streams its 715k-passage corpus once — the big one-time download; everything caches to
`checkpoints/qa_*.json` and is reused by every later model).
