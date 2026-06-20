# GATE-GRAFT — feasibility results

Bitext-free extension of a frozen English text-embedder to new languages, via
Gromov–Wasserstein/anchored alignment of fastText + grafting, with a per-word reliability
gate. All numbers are from `gate_graft/*` on public data (fastText-157, MUSE, OPUS-100,
multilingual-e5 as topline). Raw JSON under `results/` (gitignored).

## Headline
- **The bitext-free graft works at the sentence level and recovers 41–78% of a multilingual
  model *trained on* the language** — frozen English encoder, no parallel data.
- **The reliability gate is a WORD-level tool** (lexicon-induction reliability / abstention),
  not a sentence-level embedding improver. Blending grafted embeddings by reliability *hurts*.

## 1. Word level — bilingual lexicon induction (the gate's home turf)
Anchored GW alignment of fastText; BLI via MUSE dicts (eval-only). k=2000.

| lang | tier | BLI p@1 | gate AUC | P@25% | P@100% |
|------|------|--------:|---------:|------:|-------:|
| ca | near (Romance) | 0.70 | 0.90 | 0.98 | 0.70 |
| tr | mid (Turkic) | 0.12 | 0.78 | 0.30 | 0.12 |
| fi | far (Uralic) | 0.075 | 0.69 | 0.17 | 0.075 |
| bn | non-Latin | 0.011 | 0.47 | — | 0.011 |

The gate's value tracks alignment quality: excellent where alignment is strong (ca AUC 0.90,
P@25% 0.98), degrading to chance where it collapses (bn). **Transliteration** (`--translit`)
roughly doubles Bengali BLI (0.011→0.026) and lifts gate AUC 0.47→0.53.

## 2. Sentence level — three grafting mechanisms (Catalan)
| mechanism | ca sent p@1 | note |
|-----------|------------:|------|
| bow → e5 sentence-space lift | 0.04 | lossy lift dominates |
| gate-weighted bow in fastText space | 0.28 | lift was the bottleneck |
| **token-graft into frozen e5 (transformer composes)** | **0.54** | the real thing |

## 3. Token-graft across 11 languages / 6 scripts (the main result)
Graft top-~10–14k target words into frozen e5 input embeddings (WECHSEL-style lift), extend
the tokenizer, frozen body composes. OPUS-100 target→en retrieval (n=400). Topline =
multilingual-e5 (trained on the language).

| lang | script | graft p@1 | topline | **recovery** | conf@25 |
|------|--------|----------:|--------:|-------------:|--------:|
| id | Latin | 0.605 | 0.873 | 69% | 0.61 |
| de | Latin | 0.600 | 0.873 | 69% | 0.72 |
| ca | Latin | 0.537 | 0.688 | **78%** | 0.65 |
| ru | Cyrillic | 0.477 | 0.912 | 52% | 0.52 |
| tr | Latin | 0.465 | 0.850 | 55% | 0.45 |
| fi | Latin | 0.445 | 0.713 | 62% | 0.38 |
| bn | Bengali | 0.410 | 0.897 | 46% | 0.38 |
| ar | Arabic | 0.403 | 0.907 | 44% | 0.56 |
| el | Greek | 0.400 | 0.870 | 46% | 0.42 |
| hi | Devanagari | 0.390 | 0.902 | 43% | 0.48 |
| he | Hebrew | 0.370 | 0.897 | 41% | 0.34 |

Absolute graft p@1 is strikingly flat (0.37–0.605) across very diverse languages; recovery
(41–78%) is lower for distant/non-Latin languages mostly because their toplines are higher.

## 4. Controls / ablations
- **Random-init baseline** (new rows = noise, no alignment): ca 0.098, tr 0.052, fi 0.068,
  bn 0.015 — vs graft 0.41–0.54. **The alignment is what makes the graft work (5–27× over
  random).** Decisive control.
- **Vocab-size sweep** (ca, grafted tokens): 5k→0.362, 15k→0.537, 30k→0.605. More coverage
  helps monotonically; scaling the grafted vocabulary further should help.
- **Bengali confound ruled out**: keep only Bengali-script words → 0.41 (unchanged); keep
  only shared Latin/digit tokens → 0.007 (random). The bn signal is real grafted Bengali.
- **Gate at sentence level**: embedding-blend *hurts* everywhere (ca 0.54→0.35). Gate-as-
  confidence (P@25% > overall) helps a *majority* of languages (ca/de/ru/el/ar/hi, big for
  ar/hi) but is inconsistent (worse for tr/fi/he/bn). Useful-but-noisy; not the headline.

## 5. Key findings & reframing
1. **GRAFT is the strong, general result.** Bitext-free lexical grafting into a frozen English
   encoder recovers a large fraction (41–78%) of a model trained on the language, across 6
   scripts — validated by the random-init control and the Bengali script ablation.
2. **Sentence aggregation is robust to noisy per-word alignment.** Bengali word-BLI 0.011 vs
   sentence retrieval 0.41 — coarse alignment useless for word translation suffices for
   sentence retrieval.
3. **The gate is a word-level reliability tool**, not a sentence-level embedding improver.
   The original "gate improves grafted output" coupling is only half-supported.
4. **Implication for the proposal**: lead with bitext-free grafting as a frozen-encoder
   language-extension method; position the gate as a lexical-reliability / abstention
   contribution (word-level).

## Caveats
- e5-base-v2 is uncased + accent-stripping; the graft uses whole-word `normalized=False`
  tokens (not proper subword surgery) — a feasibility shortcut.
- OPUS-100 test sets vary in difficulty (toplines 0.69–0.91); recovery ratio is the fairer
  cross-language metric than raw p@1.
- Swahili is absent from OPUS-100 (excluded). The gate-blend fallback (toward the mean token)
  is one naive mechanism; smarter gate uses at the token level are untested.

## Reproduce
```bash
pip install -r requirements-local.txt && pip install -e .          # word-level (CPU)
python -m gate_graft.run_feasibility --langs ca tr fi bn --k 2000  # BLI + gate
# token-graft + matrix need torch + transformers (requirements-cloud.txt):
python -m gate_graft.token_graft --lang ca                          # one language
python -m gate_graft.run_matrix                                     # full matrix + topline
```
