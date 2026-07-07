# Finalized retrieval experiment — byte vs subword, BGE-M3 teacher

The locked design for the paper's retrieval/QA axis. vs the completed SONAR study, the go-forward run
changes the **teacher** (SONAR → BGE-M3), the **objective** (retrieval-only InfoNCE), and the
**language set** (deep-research-verified, below). The core comparison — byte vs subword — is entirely
WITHIN this run: both students share the teacher, targets, data, recipe, and evaluation, so the
tokenizer remains the only manipulated variable.

## Languages (finalized 2026-07 via adversarially-verified deep research)

7 lower-resource languages + 3 high-resource anchors. **ONE deep benchmark per language** — the one
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
| Kinyarwanda (rw) | 1 | AfriQA (347 native questions → English passages) | **cross-lingual, REVERSE** (rw query → en passage) — flagged | trained (42,621-sentence wiki = the floor-setter); the mirrored counterpart of ha's standard |
| English (en) | 5 | MIRACL (799q / 32.9M) | monolingual | anchor |
| Chinese (zh) | 5 | MIRACL (393q / 4.9M) | monolingual | non-Latin anchor |
| Arabic (ar) | 5 | MIRACL (2,896q / 2.1M) | monolingual | non-Latin anchor |

The two flagged languages are symmetric: ha's benchmark crosses on the QUERY side (English query,
deep Hausa passage pool — tests trained-language passage embeddings), rw's crosses on the PASSAGE
side (native Kinyarwanda question, deep English pool — tests trained-language query embeddings, and
is the literal RAG use case: a low-resource speaker querying English Wikipedia).

**Bengali (Joshi class 3) is the one deliberate relaxation of the class 0–2 rule**, and it is stated
as such in the paper: the adversarially-verified sweep found NO remaining class 0–2 language with any
usable deep-retrieval benchmark (Somali is CIRAL-only and its ~9k-article Wikipedia cannot meet the
training floor; Tigrinya's TiQuAD is not publicly distributed; Sinhala/Nepali/Khmer/Lao/Burmese have
nothing real). Among the verified backfill candidates (bn/th/id, all class 3), Bengali has the
strongest case: monolingual MIRACL with human judgments, a fifth script (Bengali–Assamese) for the
study, and 270M speakers served by thin per-capita NLP resources.

**Dropped:** Tamil + Marathi (their only benchmark, IndicQA, has a ~250-doc pool — not deep),
Somali (CIRAL-only + untrainable ~9k-article wiki). Kinyarwanda was initially dropped ("no deep
benchmark"), then REINSTATED once AfriQA's gold passages were wired: accepting flagged cross-lingual
coverage for Hausa and not for Kinyarwanda was inconsistent, and rw trains normally (it clears the
42k floor).

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
| Optimizer | AdamW lr 2e-4, batch 64, **100k-step CAP for every model (iso-cap)**, bf16 |
| Early stop | **ENABLED: patience=10 × 1,000-step windows, min_delta=1e-3** — stop when 10k consecutive steps fail to improve the windowed loss; identical rule for every model. Realized `steps_run` is saved per model and MUST be reported alongside scores (models may stop at different steps — the cap and the stopping rule, not the step count, are held equal) |
| Data | 10 languages, balanced max-min Wikipedia sentences (~42k/lang, ~420k total) |
| Targets | BGE-M3, precomputed once, cached per language-LIST tag (`teachertargets_bge-m3_te-bn-sw-..._42000`) |
| Pooling | `attn` for byte AND subword (fair) |
| Checkpoints | `{name}_attn_bge-m3.pt` — per-teacher namespace (+ `_b-{arm}` for boundary arms) |
| Baselines | **The teacher only (BGE-M3)** — its scores are the per-benchmark ceiling; no third-party baselines are measured in this run (mE5/LaBSE reference numbers live in the SONAR-run results) |

## Evaluation (retrieval-only, one deep benchmark per language)

All benchmarks report the full statistic set — **nDCG@10, precision@10, recall@10, recall@100 (deep)
/ recall@1 (Belebele), and MRR@10**. On single-relevant benchmarks (Belebele, Amharic-PR, AfriQA)
precision@k is recall@k / k by construction — reported anyway for completeness.

| Benchmark | Pool | Languages | Axis |
|---|---|---|---|
| Belebele | 488 | all 10 | shallow passage retrieval |
| MIRACL (dev) | 20k rerank pools | en zh ar te bn sw yo | deep monolingual |
| Amharic-PR | 20k | am | deep monolingual |
| CIRAL Test A | full-corpus stream → pool | ha | **cross-lingual, flagged** |
| AfriQA | gold contexts + 20k en distractors | rw (deep) · ha/sw/yo (probe) | **cross-lingual REVERSE, flagged** |

FLORES bitext, SIB, and STS are dropped from the default (`eval_battery` computes them only on
request via `tasks=`). Mr.TyDi, IndicQA, 2AIRTC, and AfriCLIRMatrix remain wired but off the default
battery (optional corroboration). MIRACL evaluation must use dev (test qrels are held out); report
CIs for yo (119 queries) and te's judgment-sparse dev (~2 judgments/query).

## Full-corpus FINAL evaluation (post-training; the reported numbers)

The 20k rerank pools above are the cheap training-time signal; the paper's headline numbers come
from the FULL-CORPUS pass (`byte_embed/full_eval.py`, dispatched by `slurm/submit_full_eval.sh`
after all trainings finish — one job per model + the BGE-M3 baseline + a merge job):

- **MIRACL at TRUE full corpus for the thesis languages** — te 518k, bn 297k, sw 132k, yo 49k
  passages — and 500k-capped pools for the anchors (en 33M / zh 4.9M / ar 2.1M are evaluation-
  compute prohibitive at full size and carry no thesis claims). ALL dev queries. This matches the
  official MIRACL protocol for the full-corpus languages -> externally comparable numbers.
- **Deep QA at full corpus, all queries**: Amharic-PR 68k, 2AIRTC 12.6k, Mr.TyDi te/sw 548k/137k
  (public TEST labels — the held-out corroborator), IndicQA te, CIRAL-ha 715k. AfriQA at 100k
  English distractors (constructed pool; no native corpus).
- Full-corpus scores are LOWER than pool scores by construction — never compare across settings.

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
**BOTH arms run at ALL THREE byte sizes** (6 boundary models; 12 trained students in total across
the study, 100k steps each), each arm in its own results file + `_b-{arm}` checkpoint namespace; the
notebook prints the per-size three-arm comparison table (A = the main run's byte students).

## AfriQA reverse cross-lingual benchmark (on the default battery)

The REVERSE cross-lingual axis: **African-language question → English gold passage** — the direction
a low-resource speaker querying English Wikipedia actually needs. Native human questions from
`masakhane-io/afriqa` gold passages (GitHub JSONL); pool = all gold contexts + English distractors
streamed from the MIRACL en corpus. Roles: **rw (347q) — Kinyarwanda's deep benchmark** (rw is
trained; flagged cross-lingual, mirroring ha's CIRAL standard); ha (300q), sw (295q), yo (254q) ride
along as the reverse-axis probe. Runs in the main eval; for results files created before it was
added, backfill additively via `reeval(..., qa_only=True, benchmarks=('afriqa',))`. The notebook
prints the per-language table.

## Predictions (written before running)

1. Both students' deep-retrieval scores rise substantially vs the SONAR run (teacher ceiling lifted;
   objective purely discriminative).
2. Byte > subword persists at every size on the deep monolingual benchmarks.
3. byte-small vs subword-large: byte-small stays ahead on deep retrieval (the cross-size headline).
4. yo scores land low for both students (teacher coverage), with the byte−subword gap still positive.

## How to run (single session, multi-session, or SLURM)

**Colab, one session:** `notebooks/byteembed_retrieval_a100.ipynb` top-to-bottom. (CLI equivalent:
`python -m byte_embed.run_lowresource --teacher bge-m3 --pooling attn --steps 100000 --out results/retrieval_bgem3.json`)

**Colab, several sessions in parallel** (shared Drive folder): every training cell exposes
`SESSION_MODELS` / `ARM_MODELS` + `MAX_CONCURRENT` + `RUN_BASELINES` knobs — give each session a
DISJOINT model list (part-files and checkpoints are per-model, so they never collide). Two rules:
(1) start one session alone until its log prints `reusing cached targets` (the one-time teacher
pass), then start the rest; (2) `RUN_BASELINES=True` in exactly one session. Example 3-session split:
byte-large+subword-large / the four small+base / both boundary arms.

**SLURM:** `bash slurm/submit_all.sh` from the repo root — 1 precompute job → 12 dependency-gated
training jobs (one model each, `slurm/train_model.sbatch`) → 3 merge jobs (the main merge also scores
the teacher baseline). Same part-file convention as `run_parallel`, so Colab sessions and SLURM jobs
are interchangeable mid-study.

Results land in `results/retrieval_bgem3.json` (+ `_bteacher` / `_brandom` for the arms). The first
eval builds the retrieval pools (CIRAL streams its 715k-passage corpus once — the big one-time
download; everything caches to `checkpoints/qa_*.json` and is reused by every later model).
