# embedding-research — ByteEmbed

**Byte-level vs subword tokenizer-free multilingual sentence embeddings**, distilled from a frozen
retrieval teacher (**BGE-M3**) and evaluated as retrievers for low-resource-language question
answering. The clean question: at matched compute, is removing the ~250k subword vocabulary table a
better **parameter allocation** for low-resource retrieval? The byte student (ByT5 encoder) and the
subword student (mT5 encoder) share the teacher, targets, data, recipe, and evaluation — the
tokenizer is the only manipulated variable.

## Start here
- **`RETRIEVAL_EXPERIMENT.md`** — the locked experimental design: the 10-language set (7 lower-resource
  + 3 anchors, deep-research-verified), the retrieval-only InfoNCE objective, the 12-model grid
  (byte/subword × small/base/large + boundary arms), the eval battery, and how to run it.
- **`byte_embed/README.md`** — module map and the code-level how-to.
- **`byte_embed/RESULTS.md`** — study design, validated dataset sizes, measured tokenization-tax.
- **`byte_embed/NOVELTY.md`** — citation-grounded positioning vs prior work.
- **`figures/`** — figures (PNG); regenerate with `python gen_figures.py`. Raw run JSON is under
  `results/` (gitignored).

## Layout
```
RETRIEVAL_EXPERIMENT.md   locked design for the byte-vs-subword retrieval study — START HERE
gen_figures.py            builds figures/ from results/*.json
common/                   shared eval helpers (l2norm + optional SIB/STS MTEB probes)
byte_embed/               the study (+ README.md, RESULTS.md, NOVELTY.md)
  run_lowresource.py  orchestrator      teachers.py  BGE-M3 / mE5 retrieval teachers
  distill.py          InfoNCE + patience distillation      model.py  byte/subword students
  data.py  balanced Wikipedia sampler   config.py  language sets
  miracl.py / qa_retrieval.py  deep retrieval evals (pool + full-corpus)
  eval_mteb.py  Belebele battery        boundaries.py  boundary-injection arms
  full_eval.py  post-training full-corpus final evaluation
  reeval.py  additive re-evaluation of finished checkpoints
notebooks/                byteembed_retrieval_a100.ipynb (the runner) + specialization notebooks
slurm/                    submit_all.sh (training) · submit_full_eval.sh (final eval) · train_model.sbatch
results/                  raw run JSON (gitignored)
```

## Quick start (Colab A100 / cloud GPU)
Open `notebooks/byteembed_retrieval_a100.ipynb`, set runtime = A100, run top-to-bottom (do the smoke
cell first — it validates the BGE-M3 teacher and the whole pipeline). Resumable across sessions.
CLI equivalent:
```bash
pip install -r requirements-cloud.txt && pip install -e .
python -m byte_embed.run_lowresource --smoke      # ~5-min self-test
python -m byte_embed.run_lowresource --teacher bge-m3 --pooling attn --steps 100000 \
    --out results/retrieval_bgem3.json            # the full study (resumable)
```

## SLURM
```bash
bash slurm/submit_all.sh          # 1 precompute -> 12 dependency-gated trainings -> merges
bash slurm/submit_full_eval.sh    # after training: full-corpus final eval, one job per model
```
Same per-model part-file convention as the notebook, so Colab sessions and SLURM jobs are
interchangeable mid-study. See `RETRIEVAL_EXPERIMENT.md` → "How to run" for multi-session
partitioning and the cluster-flag env vars (partition/account/qos/gres).

## Status
The 12-model study runs on the UMD Nexus (CLIP) cluster / Colab A100. Training uses a retrieval-only
InfoNCE objective distilled from BGE-M3; the reported numbers come from the post-training full-corpus
evaluation (`full_eval.py`). Predictions are pre-registered in `RETRIEVAL_EXPERIMENT.md`.
