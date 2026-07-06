# Finalized retrieval experiment — byte vs subword, BGE-M3 teacher

The locked design for the paper's retrieval/QA axis. vs the completed SONAR study, the go-forward run
changes the **teacher** (SONAR → BGE-M3), the **objective** (retrieval-only InfoNCE), and the
**language set** (deep-research-verified, below). The core comparison — byte vs subword — is entirely
WITHIN this run: both students share the teacher, targets, data, recipe, and evaluation, so the
tokenizer remains the only manipulated variable.

## Languages (finalized 2026-07 via adversarially-verified deep research)

6 lower-resource languages + 3 high-resource anchors. **ONE deep benchmark per language** — the one
with the most passages available (simplest to track; the alternatives stay wired for optional
corroboration via `benchmarks=`):

| Lang | Joshi | Deep retrieval (the one) | Axis | Notes |
|---|---|---|---|---|
| Telugu (te) | 1 | MIRACL dev (828q / 518k psgs) | monolingual | Mr.TyDi + IndicQA off-default |
| Bengali (bn) | 3 | MIRACL dev (411q / 297k) | monolingual | the ONE class-rule relaxation (see below) |
| Swahili (sw) | 2 | MIRACL dev (482q / 132k) | monolingual | Mr.TyDi off-default |
| Yoruba (yo) | 2 | MIRACL dev (119q / 49k, "surprise language") | monolingual | NOT in XLM-R (see caveat) |
| Amharic (am) | 2 | Amharic-PR (68,301 psgs) | monolingual | > 2AIRTC's 12,587; 2AIRTC off-default |
| Hausa (ha) | 2 | CIRAL Test A (80q / 715k news psgs) | **cross-lingual** (en query → ha passage) — flagged | |
| English (en) | 5 | MIRACL (799q / 32.9M) | monolingual | anchor |
| Chinese (zh) | 5 | MIRACL (393q / 4.9M) | monolingual | non-Latin anchor |
| Arabic (ar) | 5 | MIRACL (2,896q / 2.1M) | monolingual | non-Latin anchor |

**Bengali (Joshi class 3) is the one deliberate relaxation of the class 0–2 rule**, and it is stated
as such in the paper: the adversarially-verified sweep found NO remaining class 0–2 language with any
usable deep-retrieval benchmark (Somali is CIRAL-only and its ~9k-article Wikipedia cannot meet the
training floor; Tigrinya's TiQuAD is not publicly distributed; Sinhala/Nepali/Khmer/Lao/Burmese have
nothing real). Among the verified backfill candidates (bn/th/id, all class 3), Bengali has the
strongest case: monolingual MIRACL with human judgments, a fifth script (Bengali–Assamese) for the
study, and 270M speakers served by thin per-capita NLP resources.

**Dropped:** Kinyarwanda (no deep-retrieval benchmark exists anywhere — AfriQA's gold passages are
English/French pivot text, not Kinyarwanda), Tamil + Marathi (their only benchmark, IndicQA, has a
~250-doc pool — not deep), Somali (CIRAL-only + untrainable wiki). **AfriQA finding:**
`masakhane/afriqa-gold-passages` DOES ship `context` passages, but they are English/French Wikipedia
text — usable only as reversed cross-lingual coverage for bem/fon/ibo/kin/twi/wol/zul; not wired.

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
| Data | 9 languages, balanced max-min Wikipedia sentences (~42k/lang, ~378k total) |
| Targets | BGE-M3, precomputed once, cached per language-LIST tag (`teachertargets_bge-m3_te-bn-sw-..._42000`) |
| Pooling | `attn` for byte AND subword (fair) |
| Checkpoints | `{name}_attn_bge-m3.pt` — per-teacher namespace (+ `_b-{arm}` for boundary arms) |
| Baselines | **The teacher only (BGE-M3)** — its scores are the per-benchmark ceiling; no third-party baselines are measured in this run (mE5/LaBSE reference numbers live in the SONAR-run results) |

## Evaluation (retrieval-only, one deep benchmark per language)

| Benchmark | Pool | Languages | Axis / metric |
|---|---|---|---|
| Belebele | 488 | all 9 | shallow passage retrieval, nDCG@10 |
| MIRACL (dev) | 20k rerank pools | en zh ar te bn sw yo | deep monolingual, nDCG@10 + R@100 |
| Amharic-PR | 20k | am | deep monolingual, nDCG@10 + R@100 |
| CIRAL Test A | full-corpus stream → pool | ha | **cross-lingual, flagged**, nDCG@10 + R@100 |

FLORES bitext, SIB, and STS are dropped from the default (`eval_battery` computes them only on
request via `tasks=`). Mr.TyDi, IndicQA, 2AIRTC, and AfriCLIRMatrix remain wired but off the default
battery (optional corroboration). MIRACL evaluation must use dev (test qrels are held out); report
CIs for yo (119 queries) and te's judgment-sparse dev (~2 judgments/query).

## Boundary-injection arms (the tokenization-mechanism probe; byte students only)

`--boundary teacher|random` (or `run(..., boundary=...)`), each arm in its own results file:

- **A (raw)** — the normal byte student (the main run).
- **B (`teacher`)** — a 1-byte marker (U+001E) inserted wherever BGE-M3's tokenizer (XLM-R spm)
  would split. Grants the byte model subword's segmentation INFORMATION with zero vocab table and
  zero parameter change.
- **C (`random`)** — the placebo: the SAME per-sentence marker count at random character positions
  (deterministic per sentence). Controls for sequence-length / delimiter / register-token artifacts.

Teacher targets stay clean; the transform applies to the student's training inputs AND all its eval
inputs. Reading: B > C ≈ A → segmentation info genuinely helps; B ≈ C > A → any markers help
(artifact — no credit to the tokenizer); B ≈ C ≈ A → byte needs nothing from segmentation.
**BOTH arms run** in the notebook (byte-small each, 50k, back to back, resumable); the notebook cell
prints the three-arm comparison table (A = the main run's byte-small).

## AfriQA cross-lingual viability probe (eval-only)

The REVERSE cross-lingual axis: **African-language question → English gold passage** — the direction
a low-resource speaker querying English Wikipedia actually needs. Benchmark `afriqa` (off the default
battery): native human questions from `masakhane-io/afriqa` gold passages (GitHub JSONL), pool = all
gold contexts + English distractors streamed from the MIRACL en corpus. Languages (trained only):
ha (300q), sw (295q), yo (254q). Backfilled additively onto every model via
`reeval(..., qa_only=True, benchmarks=('afriqa',))`; the notebook prints the per-language viability
table.

## Predictions (written before running)

1. Both students' deep-retrieval scores rise substantially vs the SONAR run (teacher ceiling lifted;
   objective purely discriminative).
2. Byte > subword persists at every size on the deep monolingual benchmarks.
3. byte-small vs subword-large: byte-small stays ahead on deep retrieval (the cross-size headline).
4. yo scores land low for both students (teacher coverage), with the byte−subword gap still positive.

## How to run

Colab: `notebooks/byteembed_retrieval_a100.ipynb` top-to-bottom (smoke → parallel run → baselines).
CLI: `python -m byte_embed.run_lowresource --teacher bge-m3 --pooling attn --steps 50000 --out results/retrieval_bgem3.json`

Results land in `results/retrieval_bgem3.json`. The first eval builds the retrieval pools (CIRAL
streams its 715k-passage corpus once — the big one-time download; everything caches to
`checkpoints/qa_*.json` and is reused by every later model).
