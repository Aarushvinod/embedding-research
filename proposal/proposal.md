# Tokenizer-Free Embeddings for Low-Resource Question Answering: Byte-Level versus Subword Multilingual Retrieval

*Aarush Vinod — Sorva Health — avinod@sorvahealth.com*

*(Editable full-text version. The compiled proposal lives in `main.tex` with citations in `references.bib`; this file is the same content in raw text for easy editing. Inline `[key]` markers correspond to the bibliography keys.)*

---

## Abstract

Answering questions in low-resource languages increasingly relies on retrieval-augmented generation (RAG), where a retriever finds the passage containing the answer. Yet the multilingual embeddings these retrievers use rely on English-centric subword tokenizers. These tokenizers segment low-resource languages into far more tokens than they do for high-resource languages. We ask whether tokenizer-free, byte-level embeddings yield better retrieval. To test this, we distill SONAR embeddings into byte-level (ByT5) and subword (mT5) students across three model sizes, with fixed training objectives and parameters. We evaluate both embeddings on passage retrieval (Belebele) and deep retrieval tasks (MIRACL, IndicQA, 2AIRTC, and AfriCLIRMatrix), for six low-resource languages (Telugu, Tamil, Marathi, Amharic, Kinyarwanda, Hausa). Preliminary results show that byte embeddings perform better on passage retrieval, although the advantage diminishes as parameter count increases. For deep retrieval, byte embeddings perform significantly better, with the smallest byte-level model surpassing the largest subword model. To check that this is a property of the representation rather than of raw parameter count, we add a control in which byte and subword students share an identical transformer and differ only in the input tokenizer. This project aims to show that byte-level embeddings yield fairer multilingual embeddings well-suited for low-resource languages, and that a smaller tokenizer-free retriever can serve those languages better than a larger subword one.

---

## 1. Introduction and Motivation

Large language models are increasingly the interface through which people seek information, and for the many languages with little training data this is made workable by retrieval-augmented generation (RAG): rather than answer from parametric memory alone, the system first *retrieves* a passage likely to contain the answer and then conditions its response on that passage [lewis2020rag]. The quality of the final answer is therefore bounded by the quality of the retriever. For low-resource languages, however, the multilingual sentence embeddings that power retrieval are built on subword tokenizers trained on English-dominated corpora, which fragment low-resource and non-Latin-script languages into far more tokens than high-resource ones [petrov2023, ahia2023]. This "tokenization tax" degrades representation quality for exactly the languages that are least served today and that stand to benefit most from better retrieval, and it raises a concrete research question: is the subword tokenizer itself a bottleneck for low-resource retrieval, and would removing it help?

This proposal investigates that question directly. We compare *byte-level* (ByT5) and *subword* (mT5) multilingual sentence embeddings, distilled from a shared frozen SONAR teacher under a fixed recipe so that only the input representation varies, and evaluate them as retrievers for six low-resource languages spanning four scripts and five language families, with three high-resource anchors. We organize the work around four research questions:

- **RQ1.** Do byte-level embeddings retrieve better than subword embeddings for low-resource languages, on both shallow passage retrieval and deep open-domain (QA / ad-hoc) retrieval?
- **RQ2.** How does any advantage depend on model size — in particular, can the smallest byte-level model match or surpass the largest subword model?
- **RQ3.** Is a byte advantage attributable to the *representation* rather than to raw parameter count, when transformer capacity is held equal across the two tokenizers?
- **RQ4.** Can an intrinsic, downstream-free measure of tokenization (such as cross-lingual *parity*) characterize the inequity that motivates the approach, and does it predict the per-language retrieval gap between byte and subword?

Preliminary results suggest the answer to RQ1 is yes, and most strongly for deep retrieval, where the smallest byte model already surpasses the largest subword model. Section 2 situates the work in the literature and makes the research gap explicit; Section 3 details the proposed methodology and evaluation.

---

## 2. Literature Review

Modern NLP serves the world's languages unequally, and a growing body of work locates one root of that inequality in the tokenizer. Joshi et al. [joshi2020] document a steep resource hierarchy in which most languages, including the majority spoken across the Global South, receive little data, tooling, or evaluation. Subword tokenizers, trained on corpora dominated by high-resource languages, inherit and amplify this imbalance. Petrov et al. [petrov2023] show that identical content translated across languages is segmented into very different numbers of tokens, with disparities reaching 15x and non-Latin scripts among the worst affected. Ahia et al. [ahia2023] quantify the effect on parallel FLORES-200 text, where mid-resourced languages with their own scripts such as Telugu and Georgian require up to 5x more tokens than English. Because commercial models bill per token, this tokenization tax is also economic: speakers of those languages pay several times more for equivalent content.

The penalty is structural rather than incidental. Somide [somide2026] finds that every African language studied carries a per-language premium over English, and that even the best available subword tokenizer narrows but never closes the gap. Lundin et al. [lundin2025tokentax] report that higher subword fertility, the number of tokens per word, is associated with lower downstream accuracy on African-language QA, though the relationship is correlational and partly confounded with pretraining-data volume. Both are recent preprints, and their exact multipliers should be read with that caveat, and with the tokenizer version on which they were measured, since newer vocabularies have reduced the worst cases.

The natural response is to remove the tokenizer. Clark et al. [clark2022canine] argue that fixed subword vocabularies are not equally suited to all languages and operate directly on characters, discarding the vocabulary at inference; CANINE outperforms a larger subword mBERT by 2.8 F1 on multilingual TyDi QA while using 28% fewer parameters. ByT5 [xue2022byt5], the byte-level model we build on, matches or beats the equally sized subword mT5 [xue2021mt5] at small scale, attributed to reallocating the parameters that subword models lock in a large vocabulary-embedding matrix into the transformer itself. Once viewed as a small-model curiosity, byte-level modeling now scales: the Byte Latent Transformer [pagnoni2024blt] matches a FLOP-controlled Llama 3 up to 8B parameters. Together these establish both feasibility and a precedent for a smaller tokenizer-free model outperforming a larger subword one.

The tokenizer literature also offers an intrinsic, downstream-free way to characterize multilingual fairness. Tokenization parity, the token-length ratio of a language to English on parallel text [petrov2023, ahia2023], is computable for any tokenizer, including byte-level ones where it reduces to average bytes per sentence, making it the one number that compares byte and subword on equal terms; complements include the Gini coefficient of per-language cost [lee2026equity] and the information-theoretic Renyi efficiency [zouhar2023]. Two cautions apply, and both shape our claims. First, byte-level input reduces but does not eliminate cross-lingual disparity, which still exceeds 4x for some pairs because UTF-8 length varies by script [petrov2023]. Second, these intrinsic measures do not reliably predict downstream quality, which is task-dependent [lee2026equity, ali2023tokenizer]. Parity is therefore best reported as a descriptive measure of multilingual equity, not a validated forecaster of task performance.

**Research gap.** Three gaps remain, and this proposal targets all three. (i) The verified evidence for tokenizer-free gains concentrates on classification and extractive QA; *no prior study isolates a byte advantage for dense retrieval* — the step that actually bottlenecks multilingual RAG — in low-resource languages. (ii) Existing comparisons confound the input representation with parameter count, because byte and subword models of the "same size" allocate parameters very differently; a control that holds transformer capacity fixed is missing. (iii) Although intrinsic tokenization metrics are widely reported, *whether parity predicts the per-language byte-versus-subword retrieval gap has not been tested*. Filling these gaps requires a controlled distillation study with a retrieval-centric, broad low-resource evaluation, which we now describe.

---

## 3. Proposed Methodology

### 3.1 Theoretical framework

Our central hypothesis is one of *parameter allocation*. A subword model spends a large fraction of its parameters on a vocabulary embedding table (Table 1): roughly 87% for the smallest mT5 and still 45% at large scale. A byte-level model needs only a ~256-entry input table, so almost all of its parameters sit in the transformer that actually contextualizes the input. At matched *total* parameters, then, byte models devote more capacity to computation, and a tokenizer-free model can plausibly serve a low-resource language better than a larger subword one that has spent its budget memorizing a vocabulary skewed toward high-resource languages. We test the hypothesis through controlled distillation: rather than pretrain from scratch, we distill a single strong frozen multilingual teacher into byte and subword students under an identical recipe, so the *only* variable is the input representation. A matched-transformer ablation (Section 3.6) then separates the representation effect from raw capacity.

### 3.2 Teacher and cached targets

The teacher is the SONAR text encoder [duquenne2023sonar], a 1024-dimensional sentence-embedding space covering 200 languages, used frozen. Because SONAR spans every study language, it imposes no "teacher ceiling" on the low-resource targets. We embed each training sentence once with SONAR and cache the target vectors to disk; both students then train against identical cached targets, which guarantees identical supervision and removes the per-step teacher cost.

### 3.3 Student architectures

Students are encoder-only ByT5 [xue2022byt5] (byte) and mT5 [xue2021mt5] (subword) at three sizes (small, base, large). Each applies pooling over the encoder outputs (mean or multi-head attentive, fixed within a comparison), a linear projection to the teacher's 1024 dimensions, and L2 normalization. We report both total and transformer-only parameter counts (Table 1), since the latter is the fair axis for the byte-versus-subword comparison.

**Table 1. Parameter allocation.**

| Size  | Subword total | Subword transformer | Vocab share | Byte total | Byte transformer |
|-------|---------------|---------------------|-------------|------------|------------------|
| Small | 147M          | 19M                 | 87%         | 219M       | ~219M            |
| Base  | 278M          | 86M                 | 69%         | 416M       | ~416M            |
| Large | 565M          | 309M                | 45%         | 866M       | ~866M            |

Subword models lock 45–87% of their parameters in the vocabulary embedding table; byte models put nearly all parameters in the transformer (the byte input table is <1M). This motivates reporting transformer-only parameters and the matched-transformer control.

### 3.4 Distillation objective

Training combines three terms against the cached SONAR targets, identical for byte and subword and across sizes: (i) a contrastive InfoNCE loss (temperature 0.05) with a MoCo-style negative queue of 8,192 to sharpen retrieval geometry; (ii) an alignment loss pulling each student embedding toward its teacher target; and (iii) a relational loss that preserves the teacher's pairwise similarity structure within a batch. We optimize with AdamW (learning rate 2e-4, batch size 64), holding all hyperparameters fixed across the byte/subword arms so that the tokenizer is the only manipulated variable.

### 3.5 Training data

We train on nine languages: six low-resource targets (Telugu, Tamil, Marathi, Amharic, Hausa, Kinyarwanda) and three high-resource anchors (English, Mandarin, Arabic). Sentences are drawn from Wikipedia with a balanced max–min sampler that caps every language at the size of the smallest (~42k sentences each, ~378k total), so no language dominates the distillation signal.

### 3.6 Controlling for capacity: the matched-transformer ablation

To answer RQ3, we add a control in which byte and subword students at each size share an *identical* transformer configuration (the mT5-{size} dimensions, randomly initialized) and differ *only* in the input embedding and tokenizer (a ~256-entry byte table versus the full subword vocabulary). Here the transformer parameter counts are equal by construction (19.4M / 85.7M / 309.4M at small/base/large), so any remaining byte advantage is attributable to the representation, not to capacity. Because these models are trained from scratch, we read them as an isolating ablation rather than as headline numbers, and pair it with a cross-size comparison (RQ2): whether the smallest byte model outperforms the largest subword model.

### 3.7 Evaluation protocol

The primary axis is retrieval, scored as a rerank pool in which each query's gold passage competes against a large distractor set; we report nDCG@10 and recall@100. We complement retrieval with semantic textual similarity (STS), classification, and bitext mining for a rounded picture, and with an intrinsic tokenization metric (Section 3.8). Table 2 lists the benchmarks and their language coverage. Belebele provides uniform shallow passage retrieval for all nine languages; deep retrieval is covered by MIRACL (monolingual), IndicQA and Mr. TyDi (QA), the peer-reviewed Amharic ad-hoc collection 2AIRTC [yeshambel2020twoairtc], and the cross-lingual AfriCLIRMatrix [ogundepo2022africlirmatrix], which supplies the only public deep-retrieval signal for Hausa.

**Table 2. Evaluation matrix.**

| Task | Benchmark | Metric | Languages |
|------|-----------|--------|-----------|
| Retrieval — shallow        | Belebele [bandarkar2023belebele]            | nDCG@10            | all 9 |
| Retrieval — deep, mono     | MIRACL [zhang2023miracl]                    | nDCG@10, R@100     | en, zh, ar, te |
| Retrieval — deep, QA       | IndicQA [doddapaneni2023indicxtreme]        | nDCG@10, R@100     | mr, ta, te |
|                            | Mr. TyDi [zhang2021mrtydi]                  | nDCG@10, R@100     | te |
|                            | 2AIRTC [yeshambel2020twoairtc]              | nDCG@10, R@100     | am |
| Retrieval — deep, x-ling   | AfriCLIRMatrix [ogundepo2022africlirmatrix] | nDCG@10, R@100     | am, ha |
| Semantic similarity        | SemRel-2024 [ousidhoum2024semrel] (+ Indic/C-MTEB) | Spearman    | all 9 |
| Classification             | SIB-200                                     | accuracy          | all 9 |
| Bitext mining              | FLORES-200                                  | P@1 (→ en)        | all 9 |
| Intrinsic (no model)       | Tokenization parity / fertility             | ratio vs. English | all 9 |

Retrieval is the primary axis; every language has at least shallow retrieval (Belebele), and five of six low-resource languages have a deep-retrieval benchmark. Kinyarwanda currently has only shallow coverage (Section 3.9).

### 3.8 Intrinsic tokenization metric

To answer RQ4 we compute tokenization parity — the ratio of a language's token count to English's on parallel FLORES-200 text — for the subword tokenizer, and its byte-level analogue (bytes per sentence) for the byte model, alongside fertility and a Gini coefficient of per-language cost. These are downstream-free descriptors of multilingual equity. We then test, per language, whether parity correlates with the measured byte-minus-subword retrieval gap. We report parity as a descriptive fairness number and treat any correlation as exploratory, consistent with evidence that intrinsic metrics do not cleanly predict downstream quality [lee2026equity, ali2023tokenizer].

### 3.9 Statistical treatment, reproducibility, and scope

We report bootstrap 95% confidence intervals on all benchmark means and, budget permitting, at least two random seeds for the small and base models; comparisons are made at matched optimizer steps. All teacher targets, retrieval pools, and checkpoints are cached so runs are resumable and results are reproducible. Two scope limitations are stated up front. First, AfriCLIRMatrix is cross-lingual (English query to African passage), a different axis from the monolingual sets, and is labeled as such. Second, *Kinyarwanda has no public deep-retrieval corpus* — AfriQA [ogundepo2023afriqa], the obvious candidate, ships no passage collection — so it is evaluated on shallow retrieval (Belebele) only; extending deep coverage to Kinyarwanda is an explicit goal of ongoing data work.
