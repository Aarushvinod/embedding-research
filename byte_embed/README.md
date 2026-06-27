# ByteEmbed — tokenizer-free byte-level multilingual embeddings

Distill a frozen multilingual **SONAR** teacher into a **byte-level** (ByT5) student and ask: at
matched compute, is removing the subword vocabulary a better **parameter allocation** for
**low-resource multilingual retrieval**? The control is an identically-trained **subword** (mT5)
student — same teacher, recipe, data, and budget; only the tokenizer differs. That single-variable
A/B is the whole point (ByT5 *is* mT5 with the 250k vocab table reallocated to transformer layers).

## The study — `run_lowresource.py`
- **Teacher:** SONAR (NLLB-200 encoder, 1024-d) — covers all 200 FLORES languages, so there is **no
  teacher-ceiling** on any chosen language. Falls back to **LaBSE** (768-d) if `fairseq2` won't install.
- **Students (6):** `byt5` / `mt5` × {small, base, large}, encoder-only → mean-pool → linear → L2.
  Training all three sizes is what licenses the parameter-allocation claim (a quality-per-parameter
  frontier, not a single point).
- **Languages (9):** 6 low-resource — Telugu, Tamil, Marathi, Amharic, Hausa, Kinyarwanda (5 families,
  5 scripts) — + English / Mandarin / Arabic anchors. **Every language is scored on every task.**
- **Training:** equal **max-min balanced** Wikipedia (~42k/lang; Kinyarwanda is the floor), distilled
  against **cached teacher targets** — SONAR embeds the corpus once and both students reuse the
  identical vectors.
- **Objective:** in-batch InfoNCE + alignment (`both`) + MoCo queue 8192 + a relational STS term, AdamW,
  batch 64 (equal at all sizes), `max_bytes=256`, bf16 + grad-checkpoint; `*-large` is checkpointed.

## Eval battery — uniform (every language, every task)
| task | dataset | metric | langs |
|---|---|---|---|
| classification | SIB-200 | accuracy | all 9 |
| retrieval | Belebele | nDCG@10 | all 9 |
| bitext (parallel) | FLORES-1012 | P@1 (→English) | all 9 |
| STS | SemRel24STS · IndicCrosslingualSTS (ta) · C-MTEB (zh) | Spearman | all 9 |
| deep retrieval | MIRACL | nDCG@10 / recall@100 | en, zh, ar, te |

Plus a **tokenization-efficiency** table (subword token tax vs byte UTF-8 cost) and a **compute
profile** (FLOPs / throughput / latency / VRAM).

## Run it
- **Colab A100:** open `notebooks/byteembed_lowresource_a100.ipynb`, set runtime = A100, run
  top-to-bottom (do the smoke cell first). Resumable across sessions.
- **CLI:** `python -m byte_embed.run_lowresource --smoke` then `python -m byte_embed.run_lowresource`.

### UCD-vector variant (tokenizer-free, *not bytes*)
`ucd.py` / `run_ucd.py` explore a representation we have **not** tried before: instead of raw UTF-8
bytes, the student reads **Unicode Character Database property vectors** per codepoint
(`General_Category`, `Canonical_Combining_Class`, `Bidi_Class`, script/block, binary flags) plus a
CANINE-style hashed-codepoint identity channel — fed to the **same ByT5 body** via `inputs_embeds`, so
only the input representation changes. One position per character removes the UTF-8 multibyte tax, the
input table stays tiny (cross-lingual feature sharing), and unseen codepoints still arrive with valid
script/category structure. Reuses the **same cached SONAR targets** as the byte study (identical
supervision).
- **Colab A100:** `notebooks/ucdembed_lowresource_a100.ipynb` (sibling of the byte notebook).
- **CLI:** `python -m byte_embed.run_ucd --smoke` then `python -m byte_embed.run_ucd`
  (`--compare` also trains the byte + subword arms for a 3-way UCD/byte/subword table).

## Modules
- `run_lowresource.py` — orchestrator: teacher cache → 6 students → eval + efficiency, incremental + resumable.
- `teachers.py` — `SonarTeacher` (+ LaBSE fallback) and cached `precompute_targets`.
- `eval_mteb.py` — Belebele / FLORES-bitext / STS-dispatch uniform battery.
- `efficiency.py` — subword-tax vs byte-UTF8-tax table + FLOPs/throughput/latency profiler.
- `distill.py` — distillation loop (with cached-targets mode); `model.py` — `ByteStudent`;
  `data.py` — `load_balanced_sentences`; `miracl.py` — MIRACL; `config.py` — the 9-language code maps;
  `robustness.py` — orthographic perturbations (deferred from v1, kept for a follow-up robustness run).

## Honest framing
Byte is **not** cheaper. Its UTF-8 cost is *higher* than subword for non-Latin scripts — Indic byte
sequences run **7–10× longer** than subword, and the byte "tax" vs English exceeds the subword token
tax for Telugu/Tamil/Marathi. The claim is strictly **parameter allocation**: byt5 spends parameters
on the transformer, mt5 on a 250k vocab table (≈87% of mt5-small's encoder), and we measure
quality-per-parameter while reporting the compute cost openly.

## Status
**Implemented and locally validated** (all eval loaders, exact dataset sizes, the eval→summary→figures
contract). The 6-model A100 run is **pending**. See `RESULTS.md` for the design, the validated dataset
sizes, the measured tokenization-tax numbers, and the prior mE5-teacher feasibility work (superseded).
