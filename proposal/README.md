# Research proposal (LaTeX)

Byte-level vs. subword multilingual embeddings for low-resource question answering.

## Files
- `main.tex` — the proposal (abstract, introduction + research questions, related work, proposed work).
- `references.bib` — bibliography (natbib + bibtex).

## Compile
**Overleaf (recommended):** import this folder, then either
- *New Project → Upload Project* with a zip of `proposal/`, or
- *New Project → Import from GitHub* and point at this repo (set `proposal/main.tex` as the main file).

Overleaf auto-detects pdfLaTeX + bibtex. Build order is the standard
`pdflatex → bibtex → pdflatex → pdflatex`.

**Locally** (if you have a TeX distribution):
```bash
cd proposal
latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submitting
The `.bib` entries are best-effort: **verify author lists, venues, and arXiv IDs**, especially the
2025–2026 preprints (`lundin2025tokentax`, `lee2026equity`, `somide2026`) and the dataset cites.
The `\emph{[Expand ...]}` note in §3 (Proposed Work) marks where to add the detailed protocol,
baselines, statistical treatment, and timeline.
