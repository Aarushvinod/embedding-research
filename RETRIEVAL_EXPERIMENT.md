# Finalized retrieval experiment — byte vs subword, BGE-M3 teacher

The locked design for the paper's retrieval/QA axis. One variable changes vs the completed SONAR
study: **the teacher**. Everything else — students, data, recipe, evaluation — is identical, so the
two runs are directly comparable and any lift is attributable to the teacher's retrieval geometry.

## Why change the teacher

The SONAR study's deep-retrieval scores were bounded by the teacher, not the students: SONAR is an
NLLB bitext encoder trained for translation alignment (hence near-perfect FLORES bitext, ~0.17 MIRACL),
not for asymmetric query→passage matching. Distilling a **retrieval-trained** teacher hands both
students a retrieval-shaped embedding space while keeping training monolingual-text-only (no QA pairs
needed — viable for low-resource languages with no supervised retrieval data).

**Teacher: `BAAI/bge-m3`** — the strongest open multilingual dense retriever (XLM-R-large backbone,
100+ languages, 1024-d, SOTA-level MIRACL). No query/passage prefix needed; we distill its dense
vector. `intfloat/multilingual-e5-large` is wired as the ablation teacher (`teacher_name='me5-large'`,
`passage: ` prefix) to show the result does not depend on the teacher choice.

Known trade-off vs SONAR: XLM-R's ~100 languages cover am/ha but likely NOT Kinyarwanda (SONAR's 200
do). rw has no deep-retrieval benchmark anyway (Belebele-only), but watch its Belebele score; if it
craters, hybrid targets (BGE-M3 where covered, SONAR elsewhere) are a one-line mix at the cache layer.

## Design (locked)

| Axis | Setting |
|---|---|
| Students | `google/byt5-{small,base,large}` vs `google/mt5-{small,base,large}` (encoder-only + attn pool + 1024-d head) |
| Objective | InfoNCE (τ=0.05, MoCo queue 8192) + alignment + relational (`objective='both'`, `rel_weight=1.0`) |
| Optimizer | AdamW lr 2e-4, batch 64, **50k steps for every model (iso-step)**, bf16 |
| Data | 9 languages (te ta mr am ha rw + en zh ar), balanced ~42k sentences/lang (~378k), unchanged |
| Targets | BGE-M3, precomputed once, cached as `teachertargets_bge-m3_9langs_42000.npy` |
| Pooling | `attn` for byte AND subword (fair) |
| Checkpoints | `{name}_attn_bge-m3.pt` — per-teacher namespace, never resumes from SONAR checkpoints |
| Baselines | mE5-base, LaBSE (same battery, same pools) |

## Evaluation

Headline retrieval benchmarks (nDCG@10, plus recall@100 where pools are deep):

| Benchmark | Pool | Languages | Axis |
|---|---|---|---|
| Belebele | 488 | all 9 | shallow passage retrieval |
| MIRACL | ~20k | en zh ar te | deep monolingual |
| IndicQA | ~250 | mr ta te | small-pool QA |
| Amharic-PR | ~20k | am | deep monolingual (community) |
| 2AIRTC | 12.6k | am | deep ad-hoc IR (peer-reviewed) |

Mr.TyDi (te) and AfriCLIRMatrix (am/ha, cross-lingual) are still computed but reported as secondary
(coverage-capped pool / different task axis). The non-retrieval battery (SIB, FLORES, STS) still runs —
we expect FLORES/STS to DROP vs the SONAR run (the teacher is no longer a bitext specialist); report
that honestly as the retrieval-specialization trade-off.

## Predictions (write down before running)

1. Both students' deep-retrieval scores rise substantially vs the SONAR run (teacher ceiling lifted).
2. Byte > subword persists at every size on the deep benchmarks (the gap is representational, not
   teacher-specific).
3. byte-small vs subword-large: byte-small stays ahead on deep retrieval (the cross-size headline).
4. IndicQA (small pool) stays the closest benchmark; whether subword's lead survives the retrieval
   teacher is an open question the run answers.

## How to run

Colab: `notebooks/byteembed_retrieval_a100.ipynb` top-to-bottom (smoke → parallel run → baselines).
CLI: `python -m byte_embed.run_lowresource --teacher bge-m3 --pooling attn --steps 50000 --out results/retrieval_bgem3.json`

Results land in `results/retrieval_bgem3.json`; the SONAR results file is untouched, so the
teacher-comparison table (SONAR-taught vs BGE-M3-taught, same students) falls straight out of the two
JSONs.
