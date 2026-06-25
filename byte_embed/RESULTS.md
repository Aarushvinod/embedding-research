# ByteEmbed — results & experimental audit

**Current study: the SONAR low-resource byte-vs-subword comparison (`run_lowresource.py`).** Model
results are **pending the A100 run**; this file records the design, the *validated* dataset sizes, and
the tokenization-efficiency numbers (which need no training). Prior mE5-teacher feasibility work is
condensed at the bottom and **superseded** by this study.

---

## 1. Design

| | |
|---|---|
| **Teacher** | SONAR (NLLB-200, 1024-d; LaBSE 768-d fallback) — covers all 200 FLORES langs → no teacher-ceiling |
| **Students** | byt5 / mt5 × {small, base, large} (6), encoder-only → mean-pool → linear → L2 |
| **Languages** | te, ta, mr, am, ha, rw (low-resource) + en, zh, ar (anchors) — 5 families, 5 scripts |
| **Training** | equal max-min Wikipedia (~42k/lang), cached SONAR targets (one pass), objective `both` + queue 8192 + relational, AdamW, batch 64, steps 10k/13k/15k |
| **Eval** | SIB-200 (class.) · Belebele (retrieval) · FLORES-1012 (bitext) · STS · MIRACL (en/zh/ar/te) — **uniform: every language on every task** |
| **Controls** | iso-recipe mt5 baseline · quality-vs-params curve · cross-size control · mE5/LaBSE reference baselines |

**Hypotheses.** **H1 (allocation):** at matched teacher/recipe/data/budget, byte ≥ subword on
multilingual retrieval, with the gap largest on high-fertility low-resource languages.
**H2 (efficiency, honest):** byte is more *parameter*-efficient but **not** cheaper in sequence/compute
(worst for non-Latin) — the case rests on quality-per-parameter, not cost.

## 2. Validated dataset sizes (measured, not estimated)

**Training (Wikipedia `20231101.*`, usable 20–300-char sentences):** Kinyarwanda **42,621 ← floor**,
Amharic 64,425, Hausa 231,309, te/ta/mr/en/zh/ar all > 60k. → `n_per_lang = 42,000` × 9 ≈ **378k**, no
supplementation needed.

**Eval (per language):**
- **SIB-200** 701 train / **204 test** (uniform). **FLORES** **1,012** parallel. **Belebele** **900** questions.
- **MIRACL** dev queries: te 828 · en 799 · zh 393 · ar 2,896 (scored on a 250-query × ~20k-pool sample).
- **STS** uses **all splits** (zero-shot → leakage-free): SemRel all-splits en 8,350 / ha 2,551 / mr 1,746 /
  te 1,573 / am 1,258 / rw 1,102 / ar 627; Tamil `IndicCrosslingualSTS en-ta` **256**; Chinese `C-MTEB/STSB`
  1,361. ⚠️ Tamil (256) and Arabic (627) are the STS floor → bootstrap CIs, flag small-n. (STS22 is too
  small — en 197/ar 193 — and is not used.)

Confirmed loadable IDs: `mteb/SemRel24STS`, `mteb/IndicCrosslingualSTS`, `C-MTEB/STSB`, `Davlan/sib200`,
`mteb/flores` (devtest, one column per FLORES language), `facebook/belebele`, `mteb/MIRACLRetrieval`.

## 3. Tokenization efficiency (measured on FLORES-1012; the motivation — reported honestly)

Tax = tokens(lang)/tokens(English) on the **same parallel** content.

| lang | subword token tax (mt5) | **byte UTF-8 tax** | byte seq length vs subword |
|---|---|---|---|
| Telugu | 1.41 | **2.68** | 7.4× |
| Tamil | 1.26 | **3.19** | 9.9× |
| Marathi | 1.52 | **2.69** | 6.9× |
| Amharic | 1.71 | 1.71 | 3.9× |
| Hausa | 1.36 | 1.07 | 3.1× |
| Kinyarwanda | 1.50 | 1.13 | 2.9× |
| Arabic | 1.34 | 1.60 | 4.7× |
| Chinese | 0.91 | 0.92 | 4.0× |

**Finding (honest):** "byte removes the tokenization tax" is **false for non-Latin scripts** — UTF-8
multibyte makes byte *more* expensive there (Tamil byte-tax 3.19 vs subword 1.26; byte sequences
7–10× longer for Indic). Byte wins the cost axis only on Latin low-resource (Hausa, Kinyarwanda).
→ The motivation is **parameter allocation** (mt5-small spends 128M/147M ≈ 87% of its encoder on the
250k vocab table; byt5 ≈ 0), not "byte is cheaper." Whether that allocation actually buys quality is
what the run answers.

## 4. Model results

**Pending the A100 run.** The orchestrator writes per-model SIB / Belebele / FLORES / STS / MIRACL +
params (total / transformer-only) + a compute profile to `results/byte_lowresource.json`, and
`gen_figures.main_lowres()` renders the tax plot, the quality-per-parameter Pareto, and the
byte−subword-vs-fertility figure.

---

## Prior feasibility work (mE5 teacher — SUPERSEDED by §1–4)

Earlier phases distilled `multilingual-e5-base/large` into byte/subword students at feasibility scale
(≤24 langs, mostly mid/high-resource; run via now-removed orchestrators — recoverable from git
history). Findings that motivated the current clean study:
- **A byte student reproduces a subword teacher** (alignment ~0.92 across 5 scripts).
- **Contrastive distillation is essential for retrieval** — cosine aligns but can't retrieve
  (Tatoeba 0.12 → 0.56 with the `both` objective).
- **Iso-recipe byte > subword on retrieval/classification/romanized tasks** (subword wins STS) — but
  the size axis was **confounded** (batch shrank with size). At equal batch 64, byte scaled cleanly
  small→base incl. MIRACL +0.11 — motivating this matched-budget re-run.
- **Robustness is within-script**; orthographic augmentation fixes the romanization failure
  (−0.087 → +0.017). *(Robustness is deferred from the current v1; `robustness.py` is retained.)*
- **The "fertility → byte gain" unifier was rejected** (within-script r ≈ −0.18) — consistent with
  §3's honest finding that byte's per-script cost is governed by UTF-8, not by a uniform advantage.

These results stand as preliminary signal; the **SONAR low-resource study** above is the clean,
uniform-coverage version that the paper will report.
