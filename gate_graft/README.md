# GATE-GRAFT — feasibility

**Question this answers:** is a per-word *reliability gate* worth adding to bitext-free
cross-lingual alignment? I.e., can a cheap, bitext-free signal tell us *where* the
structural map is trustworthy, so we trust the graft there and fall back elsewhere?

**What it does (no training, runs on a laptop):**
1. Load top-k fastText vectors for a target language and English.
2. Align them bitext-free: entropic Gromov-Wasserstein -> pseudo-dictionary -> Procrustes.
3. Compute a per-word gate `r` from bitext-free signals (coupling peakiness, mutual-kNN
   neighbourhood preservation, retrieval cycle-consistency).
4. Evaluate Bilingual Lexicon Induction (MUSE dictionaries, **eval-only**) and report:
   - `p@1` (ungated) per language tier — does alignment work and decay with distance?
   - `gate AUC` — does `r` separate correctly- from incorrectly-aligned words?
   - `P@25%` vs `P@100%` — does answering only on high-gate words give high precision?

**Run:**
```bash
pip install -r ../requirements-local.txt && pip install -e ..
python -m gate_graft.run_feasibility --langs ca tr bn eu --k 2000
```
First run downloads fastText vectors (~1.2 GB/language, cached) + small MUSE dicts.

**Reading the result.** Encouraging = `p@1` decays `ca`>`tr`>`bn`>`eu`, `gate AUC` > ~0.65,
and `P@25%` clearly above `P@100%`. That means the gate predicts reliability and buys
precision-on-a-subset — justifying the full GATE-GRAFT proposal. Discouraging = AUC ~0.5
(gate is noise) or P@25% ~ P@100% (gate buys nothing).

**What this does NOT show.** It tests the gate at the *word/BLI* level, not the full
graft-into-a-frozen-sentence-encoder + sentence-retrieval pipeline. That (and the
monolingual-fallback blend, and per-token sentence confidence) is the next step if the
signals here are positive.
