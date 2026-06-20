# GATE-GRAFT — feasibility results

Bitext-free extension of a frozen English text-embedder to new languages, via
Gromov–Wasserstein/anchored alignment of fastText + grafting, with a per-word reliability
gate. All numbers are from `gate_graft/*` on public data (fastText-157, MUSE, OPUS-100,
multilingual-e5 as topline). Raw JSON under `results/` (gitignored).

## Headline
- **The bitext-free graft works at the sentence level and recovers 43–85% of a multilingual
  model *trained on* the language** — frozen English encoder, no parallel data.
- **But it is RESOURCE-bounded, not script-bounded** (expanded 21-language study, §6): in
  every script the high-resource language works and the low-resource one craters; the genuine
  low-resource tail (Amharic, Georgian) fails. The bottleneck is monolingual (fastText)
  quality — and a 100-word seed dictionary does *not* fix it.
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

## 6. Expanded study — 21 languages, script-balanced (token-graft, align_k=20000)
graft p@1 [95% bootstrap CI] · multilingual-e5 topline · recovery. Non-Latin scripts use
translit-seeded alignment (neutral effect, see below).

| script | lang | graft [CI] | topline | recov |
|--------|------|-----------:|--------:|------:|
| Latin | de / id / ca | 0.64 / 0.63 / 0.59 | 0.87 / 0.87 / 0.69 | 73 / 72 / 85% |
| Latin | pl / vi | 0.43 / 0.31 | 0.82 / 0.80 | 52 / 39% |
| Cyrillic | bg / ru / uk | 0.52 / 0.49 / 0.23 | 0.85 / 0.91 / 0.87 | 61 / 53 / 26% |
| Arabic | ar / fa / ur | 0.44 / 0.30 / 0.11 | 0.91 / 0.71 / 0.81 | 49 / 42 / 13% |
| Devanagari | hi / mr / ne | 0.39 / 0.21 / 0.11 | 0.90 / 0.94 / 0.66 | 43 / 23 / 16% |
| Bengali/Greek/Hebrew | bn / el / he | 0.42 / 0.41 / 0.38 | 0.90 / 0.87 / 0.90 | 46 / 47 / 43% |
| **low-resource** | si / km / am / ka | 0.21 / 0.12 / 0.02 / 0.02 | 0.79 / 0.63 / 0.73 / **0.03** | — |

**Resource-bounded, not script-bounded.** In *every* script the high-resource language works
(ca/de/id ~0.6, ru/bg ~0.5, ar 0.44, bn/el/he ~0.40, hi 0.39) and the low-resource one craters
(uk 0.23, ur/ne 0.11, vi 0.31, mr 0.21). The genuine low-resource tier fails (am/ka ≈ 0); where
the topline itself fails (Georgian 0.03) so does the graft (0.02) — no regime where graft beats
a weak topline. Performance tracks per-language fastText quality, with script secondary.

### Controls & ablations (align_k=20000 unless noted)
- **Random-init baseline** (alignment discarded): ca 0.12, ru 0.06, ar 0.03, hi 0.03, bn 0.01.
  Every graft CI is non-overlapping with these → the alignment effect is **significant**; the
  within-script high-vs-low gaps (e.g. ru [.44–.54] vs uk [.19–.27]) are also significant.
- **Vocab-size sweep** (helps the *working* langs): ca 10k/20k/30k = 0.50 / 0.585 / 0.605;
  hi = 0.33 / 0.385 / 0.41. Monotonic; won't fix a broken alignment.
- **Transliteration-seeding**: neutral-to-negative at the graft level — ar 0.44 vs 0.448 (off),
  hi 0.385 vs 0.388, mr 0.29 (off) vs 0.21 (on). (It 2×'d *word-level* BLI for bn, but not the
  graft.)
- **Seed dictionary (weak supervision, 100 pairs)**: does NOT rescue weak langs — uk 0.23→0.20,
  fa 0.30→0.28 (slightly worse than the unsupervised identical-string anchors). ur/mr/ne have no
  MUSE dict (fell back). **The bottleneck is monolingual quality, not supervision quantity.**

### Bottom line
GRAFT is a viable bitext-free extension method for **mid-to-high-resource languages** (any of 6
scripts), recovering 43–85% of a bitext-trained multilingual model with zero parallel data. It
is **not** a long-tail solution: the genuine low-resource tail lacks the monolingual (fastText)
quality the alignment needs, and neither bigger vocab, transliteration, nor a small seed
dictionary closes that gap. The realistic lever for the tail is *better monolingual
representations*, not more cross-lingual supervision.
