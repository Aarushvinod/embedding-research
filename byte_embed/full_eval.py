"""FULL-CORPUS final evaluation — run after all training finishes; no retraining.

Per model (loaded from its checkpoint), scores the literature-comparable battery:
  * MIRACL, FULL corpus for the thesis languages (te 518k / bn 297k / sw 132k / yo 49k passages),
    capped at 500k passages for the anchors (en/zh/ar), ALL dev queries.
  * Deep QA at FULL corpus, all queries: Amharic-PR (68k), 2AIRTC (12.6k), Mr.TyDi te/sw
    (548k/137k — the held-out-TEST corroborator), IndicQA te, CIRAL-ha (715k).
  * AfriQA at 100k English distractors, all queries (it has no native corpus — pool is constructed).

Full-corpus numbers are LOWER than the training-time 20k-pool numbers by construction (harder task);
never compare across the two settings. Pools cache once (shared by all models); each model pays only
its own encoding time (the en/zh/ar 500k pools dominate: hours for byte-large, minutes for subword).

One model per invocation (SLURM-friendly; see slurm/submit_full_eval.sh):
  python -m byte_embed.full_eval --results results/retrieval_bgem3.json --label main --only byte-small
  python -m byte_embed.full_eval --teacher-baseline                     # BGE-M3 itself, once
  python -m byte_embed.full_eval --merge                                # combine parts + print table
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

FULL_MIRACL_LANGS = ["te", "bn", "sw", "yo", "en", "zh", "ar"]
ANCHOR_CAPS = {"en": 500_000, "zh": 500_000, "ar": 500_000}   # thesis langs: no cap = full corpus
QA_FULL = ("amharicpr", "2airtc", "mrtydi", "indicqa", "ciral")
ALL_Q, ALL_D = 1_000_000, 10_000_000                          # "no limit" (cache-name-friendly)


def _models_in(results_path):
    """Union of models across the merged results file and any part-files (teacher, teacher_dim)."""
    p = Path(results_path)
    models, teacher, tdim = {}, "bge-m3", 1024
    for f in [p] + sorted(p.parent.glob(f"{p.stem}_part_*.json")):
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            models.update(d.get("models", {}))
            teacher = d.get("teacher", teacher)
            tdim = d.get("teacher_dim", tdim)
    return models, teacher, tdim


def _battery(enc, ckpt_dir):
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    out = {}
    out["miracl_full"] = eval_miracl_langs(enc, FULL_MIRACL_LANGS, n_queries=None,
                                           full=True, corpus_caps=ANCHOR_CAPS, cache_dir=ckpt_dir)
    out["qa_full"] = eval_qa_retrieval(enc, benchmarks=QA_FULL, n_queries=ALL_Q, distractors=ALL_D,
                                       max_stream=1_500_000, cache_dir=ckpt_dir)
    out["afriqa_100k"] = eval_qa_retrieval(enc, benchmarks=("afriqa",), n_queries=ALL_Q,
                                           distractors=100_000, cache_dir=ckpt_dir)["afriqa"]
    return out


def _part_path(label, name):
    return f"results/full_eval_part_{label}_{name}.json"


def run_one(results_path, label, name, pooling="attn", ckpt_dir="checkpoints", device="cuda"):
    from byte_embed.boundaries import make_transform
    from byte_embed.reeval import _load_enc

    outp = Path(_part_path(label, name))
    if outp.exists():
        print(f"=== {label}/{name}: already evaluated -> skip ===")
        return
    models, teacher, tdim = _models_in(results_path)
    if name not in models:
        raise SystemExit(f"{name!r} not found in {results_path} (finished models: {sorted(models)})")
    bm = models[name]
    tsuf = ("" if teacher == "sonar" else f"_{teacher}") + (f"_b-{bm['boundary']}" if bm.get("boundary") else "")
    enc = _load_enc(name, bm, pooling, ckpt_dir, tdim, device, tsuf=tsuf)
    if enc is None:
        raise SystemExit(f"checkpoint for {name} (suffix {tsuf!r}) not found in {ckpt_dir}")
    if bm.get("boundary"):
        _t, _e = make_transform(bm["boundary"]), enc
        enc = lambda xs: _e([_t(x) for x in xs])  # noqa: E731 — the arm evals with its own transform

    print(f"=== FULL EVAL {label}/{name} (steps_run={bm.get('steps_run')}) ===")
    res = _battery(enc, ckpt_dir)
    res.update(label=label, model=name, steps_run=bm.get("steps_run"), params=bm.get("params"),
               boundary=bm.get("boundary"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    mf = res["miracl_full"]
    print(f"  saved -> {outp} | MIRACL-full mean nDCG@10 = {mf.get('ndcg@10_mean')}")


def run_teacher_baseline(results_path="results/retrieval_bgem3.json", ckpt_dir="checkpoints",
                         device="cuda"):
    # Use the teacher RECORDED in the results, not a hardcoded model — an me5-large run must be scored
    # against mE5, not BGE-M3. load_teacher gives the same model + prefix + char-budget as the training
    # targets, so the ceiling is encoded identically to what the students distilled.
    _, teacher_name, _ = _models_in(results_path)
    label = {"bge-m3": "BGE-M3", "me5-large": "mE5-large"}.get(teacher_name, teacher_name)
    outp = Path(_part_path("baseline", label))
    if outp.exists():
        print(f"=== baseline {label}: already evaluated -> skip ===")
        return
    from byte_embed.teachers import load_teacher
    teacher = load_teacher(teacher_name, device=device)

    def enc(xs):
        return teacher.encode(xs)

    print(f"=== FULL EVAL baseline/{label} (the teacher ceiling) ===")
    res = _battery(enc, ckpt_dir)
    res.update(label="baseline", model=label)
    outp.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {outp}")


def merge(out="results/full_eval.json"):
    parts = sorted(glob.glob("results/full_eval_part_*.json"))
    merged = {"models": {}}
    for f in parts:
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        merged["models"][f"{r.get('label')}/{r.get('model')}"] = r
    Path(out).write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(parts)} parts -> {out}\n")
    print(f"{'model':28}{'steps':>8}" + "".join(f"{l:>8}" for l in FULL_MIRACL_LANGS)
          + f"{'mean':>8}{'AmhPR':>8}{'CIRAL':>8}{'MrTyDi':>8}{'AfriQA':>8}")
    for key, r in merged["models"].items():
        mf = (r.get("miracl_full") or {})
        per = mf.get("per_lang") or {}
        qa = r.get("qa_full") or {}
        af = r.get("afriqa_100k") or {}
        def g(x, k="ndcg@10"):
            return f"{x[k]:>8.3f}" if x and x.get(k) is not None else f"{'-':>8}"
        row = "".join(g(per.get(lang)) for lang in FULL_MIRACL_LANGS)
        print(f"{key:28}{str(r.get('steps_run') or '-'):>8}" + row
              + (f"{mf.get('ndcg@10_mean'):>8.3f}" if mf.get("ndcg@10_mean") is not None else f"{'-':>8}")
              + g((qa.get("amharicpr") or {}).get("per_lang", {}).get("am"))
              + g((qa.get("ciral") or {}).get("per_lang", {}).get("ha"))
              + g((qa.get("mrtydi") or {}).get("per_lang", {}).get("te"))
              + g((af.get("per_lang") or {}).get("rw")))

    # paired-bootstrap significance: byte vs subword at each size (+ boundary arms) from the parts
    try:
        from byte_embed.stats import report_arms, report_bytevssub
        print("\n" + "=" * 78 + "\nPAIRED-BOOTSTRAP SIGNIFICANCE (per-query nDCG@10; * = 95% CI excludes 0)")
        report_bytevssub()
        report_arms()
    except Exception as e:  # noqa: BLE001 — the table already printed; stats is a bonus
        print(f"[stats] paired bootstrap skipped: {type(e).__name__}: {e}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--label", default="main", help="main | bteacher | brandom | baseline")
    ap.add_argument("--only", default=None, help="model name to evaluate (one per job)")
    ap.add_argument("--teacher-baseline", action="store_true", help="evaluate BGE-M3 itself")
    ap.add_argument("--merge", action="store_true", help="combine part files + print the table")
    ap.add_argument("--pooling", default="attn")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.teacher_baseline:
        run_teacher_baseline(a.results, ckpt_dir=a.ckpt_dir, device=a.device)
    elif a.only:
        run_one(a.results, a.label, a.only, pooling=a.pooling, ckpt_dir=a.ckpt_dir, device=a.device)
    else:
        raise SystemExit("pass --only <model> (with --results/--label), --teacher-baseline, or --merge")


if __name__ == "__main__":
    main()
