"""Low-resource byte-vs-subword RETRIEVAL study — the A100 orchestrator.

Trains 6 students — byt5 / mt5 × {small, base, large} — distilled from a frozen teacher (default
SONAR; `--teacher bge-m3` for the finalized retrieval run) on 8 languages (5 low-resource:
te/sw/yo/am/ha + 3 anchors: en/zh/ar). EQUAL max-min balanced training data (~42k/lang) and CACHED
teacher targets (one teacher pass; both students train against the identical vectors). Retrieval-only
scoring: Belebele + FLORES bitext (eval_battery), MIRACL deep retrieval (en/zh/ar/te/sw/yo), the QA
benchmarks (qa_retrieval), and a compute profile. A one-time tokenization-efficiency table quantifies
the subword tax vs the byte UTF-8 cost.

INCREMENTAL + RESUMABLE: the results JSON is rewritten after every model, a model already present is
skipped, and `*-large` checkpoints model+optimizer (a Colab disconnect just means re-running the cell).

  python -m byte_embed.run_lowresource --smoke      # ~5-min self-test of the whole pipeline
  python -m byte_embed.run_lowresource              # the full study (resumable)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# RETRIEVAL-ONLY objective: pure InfoNCE contrastive distillation (student_i must pick teacher_i out
# of in-batch + queued negatives) — the discriminative geometry retrieval needs, nothing else. The
# alignment add-on ("both") and the relational term (rel_weight — added to target STS) were dropped
# when the paper narrowed to QA/retrieval.
OBJ = dict(objective="contrastive", queue_size=8192, rel_weight=0.0, optimizer="adamw")

# EQUAL batch 64. Per-size step schedule — bigger models train MORE: small 50k / base 75k / large 100k.
# byte and subword are matched WITHIN each size (the primary byte-vs-subword comparison stays iso-step);
# across sizes this is "bigger trained more" (addresses the large-model undertraining seen earlier). All
# models checkpointed. Override via run(steps=...) / parallel(steps=...) with a {size: steps} dict or an int.
_DEFAULT_STEPS = {"small": 50000, "base": 75000, "large": 100000}


def _grid(steps=None):
    sch = (_DEFAULT_STEPS if steps is None
           else steps if isinstance(steps, dict)
           else {s: int(steps) for s in _DEFAULT_STEPS})
    sizes = [(s, sch[s], True) for s in ("small", "base", "large")]
    return ([(f"byte-{s}", f"google/byt5-{s}", st, 64, ck) for s, st, ck in sizes]
            + [(f"subword-{s}", f"google/mt5-{s}", st, 64, ck) for s, st, ck in sizes])


def _f(x, w=8):
    return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def _save(results, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))


def run(out="results/byte_lowresource.json", smoke=False, device="cuda", n_per_lang=42000,
        ckpt_dir="checkpoints", teacher_name="sonar", miracl_q=250, miracl_extra=20000,
        only=None, with_baselines=True, with_efficiency=True, pooling="mean", steps=None,
        patience=0, min_delta=1e-3, boundary=None):
    """Train (a subset of) the 6 students. `only` = list of model names to train (None = all,
    [] = precompute-only). `with_baselines`/`with_efficiency` are turned OFF for parallel workers
    (the orchestrator does those once). Workers SKIP loading the teacher when targets are cached.
    `patience` (windows of no window-avg loss improvement > `min_delta` before stopping; 0 = off)
    makes `steps` a cap — the realized step count lands in the results as `steps_run`.
    `boundary` ('teacher'|'random') runs the boundary-injection arms: byte students only, markers
    inserted into all student inputs (train AND eval); teacher targets stay clean. Use a separate
    `out` file per arm. Zero-shot: ZEROSHOT_LANGS (Somali) are EVALUATED but never trained on."""
    import torch

    from byte_embed.boundaries import make_transform
    from byte_embed.config import STUDY_LANGS, ZEROSHOT_LANGS
    from byte_embed.data import load_balanced_sentences
    from byte_embed.distill import distill
    from byte_embed.efficiency import fertility_table, profile_model
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.model import ByteStudent
    from byte_embed.qa_retrieval import eval_qa_retrieval
    from byte_embed.teachers import (load_cached_targets, load_teacher, precompute_targets,
                                     targets_exist)

    langs = ["am", "sw", "en"] if smoke else STUDY_LANGS
    eval_langs = langs + ([] if smoke else ZEROSHOT_LANGS)   # Somali: eval-only (wiki too small to train)
    miracl_langs = (["te"] if smoke else ["en", "zh", "ar", "te", "sw", "yo"])   # the 6 of our 8 in MIRACL
    grid = _grid(steps)
    if smoke:
        n_per_lang = 1200
        grid = [("byte-small", "google/byt5-small", 80, 16, True),
                ("subword-small", "google/mt5-small", 80, 16, True)]
    if only is not None:
        grid = [g for g in grid if g[0] in only]
    transform = make_transform(boundary)                     # None for the raw arm
    if transform is not None:
        grid = [g for g in grid if "byt5" in g[1]]           # boundary arms are byte-only by design
        print(f"=== boundary-injection arm '{boundary}': byte students only -> {[g[0] for g in grid]} ===")

    results = (json.loads(Path(out).read_text(encoding="utf-8"))
               if Path(out).exists() else {"langs": langs, "teacher": teacher_name})
    results.setdefault("models", {})

    # 1) balanced training data + 2) teacher targets (cached: one pass, all students reuse) --------
    balanced = load_balanced_sentences(langs, n_per_lang=n_per_lang, cache_dir=ckpt_dir)
    floor = min(len(v) for v in balanced.values())
    tag = f"{len(langs)}langs_{floor}"
    if targets_exist(ckpt_dir, teacher_name, tag):            # parallel workers hit this -> no teacher
        sentences, sent_langs, targets = load_cached_targets(ckpt_dir, teacher_name, tag)
    else:
        teacher = load_teacher(teacher_name, device=device)
        sentences, sent_langs, targets = precompute_targets(teacher, balanced, langs, ckpt_dir, tag)
        del teacher
        torch.cuda.empty_cache()
    teacher_dim = int(targets.shape[1])
    results["teacher_dim"] = teacher_dim
    results["n_train"] = len(sentences)

    # 3) efficiency / tokenization table (once; no training) --------------------------------------
    if with_efficiency and "efficiency" not in results:
        print("\n=== tokenization-efficiency table (FLORES-1012) ===")
        results["efficiency"] = fertility_table(eval_langs)
        _save(results, out)

    # 4) the 6 students ---------------------------------------------------------------------------
    for name, backbone, steps, batch, ckpt in grid:
        if name in results["models"]:
            print(f"=== {name}: already in results -> skip ===")
            continue
        print(f"\n=== {name}  ({backbone}, {steps}x{batch}, teacher={teacher_name}) ===")
        try:
            student = ByteStudent(backbone, out_dim=teacher_dim, pooling=pooling).to(device)
            params = sum(p.numel() for p in student.parameters())
            vocab_params = student.enc.get_input_embeddings().weight.numel()
            torch.cuda.reset_peak_memory_stats()
            # checkpoint namespace includes the teacher for non-SONAR runs — otherwise the resume
            # logic would load the SONAR run's finished checkpoint and silently skip training.
            tsuf = "" if teacher_name == "sonar" else f"_{teacher_name}"
            if boundary:
                tsuf += f"_b-{boundary}"                     # per-arm checkpoint namespace
            cpath = str(Path(ckpt_dir) / f"{name}_{pooling}{tsuf}.pt") if ckpt else None
            hist = distill(student, None, sentences, device=device, steps=steps, batch=batch,
                           log_every=max(200, steps // 100), ckpt_path=cpath, targets=targets,
                           patience=patience, min_delta=min_delta, input_transform=transform, **OBJ)
            steps_run = hist[-1]["step"] if hist else steps   # < steps when patience stopped early

            def enc(xs, _s=student, _t=transform):           # each arm evals with its own transform
                return _s.encode([_t(x) for x in xs] if _t else xs, device=device)

            bm = eval_battery(enc, eval_langs)
            bm["miracl"] = eval_miracl_langs(enc, miracl_langs, n_queries=miracl_q,
                                             distractors=miracl_extra, cache_dir=ckpt_dir)
            bm["qa_retrieval"] = eval_qa_retrieval(enc, n_queries=(20 if smoke else miracl_q),
                                                   distractors=(500 if smoke else miracl_extra),
                                                   cache_dir=ckpt_dir)
            bm["profile"] = profile_model(student, student.tok, sentences[:256], device=device)
            bm.update(params=params, vocab_params=vocab_params,
                      transformer_params=params - vocab_params,
                      kind=("byte" if "byt5" in backbone else "subword"),
                      backbone=backbone, steps=steps, steps_run=steps_run,
                      boundary=boundary,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            results["models"][name] = bm
            _save(results, out)
            m = bm["means"]
            mir = (bm["miracl"] or {}).get("ndcg@10_mean")
            print(f"  saved {name}: params={params/1e6:.0f}M steps={steps_run}/{steps} "
                  f"Belebele={m.get('belebele_ndcg@10')} FLORES={m.get('flores_p@1')} "
                  f"MIRACL={mir} peak={bm['peak_vram_gb']}GB")
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — one model erroring shouldn't lose the others
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    # 5) reference baselines — incl. BGE-M3 itself (the teacher ceiling per benchmark) -------------
    from sentence_transformers import SentenceTransformer
    baselines = [("BGE-M3", "BAAI/bge-m3", ""),              # the teacher: measures the ceiling
                 ("mE5-base", "intfloat/multilingual-e5-base", "query: "),
                 ("LaBSE", "sentence-transformers/LaBSE", "")]
    if smoke or not with_baselines:        # parallel workers skip baselines; orchestrator runs them once
        baselines = []
    for bname, mid, prefix in baselines:
        if bname in results["models"]:
            continue
        print(f"\n=== baseline {bname} ({mid}) ===")
        try:
            mdl = SentenceTransformer(mid, device=device)

            def benc(xs, _m=mdl, _p=prefix):
                return _m.encode([_p + x for x in xs], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

            bm = eval_battery(benc, eval_langs)
            bm["miracl"] = eval_miracl_langs(benc, miracl_langs, n_queries=miracl_q,
                                             distractors=miracl_extra, cache_dir=ckpt_dir)
            bm["qa_retrieval"] = eval_qa_retrieval(benc, n_queries=miracl_q,
                                                   distractors=miracl_extra, cache_dir=ckpt_dir)
            bm.update(params=sum(p.numel() for p in mdl.parameters()), kind="baseline", backbone=mid)
            results["models"][bname] = bm
            _save(results, out)
            del mdl
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [{bname}] FAILED: {type(e).__name__}: {e}")

    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    M = results["models"]
    print("\n" + "=" * 104)
    print(f"LOW-RESOURCE BYTE vs SUBWORD — teacher={results.get('teacher', 'sonar')} "
          "(Belebele nDCG@10 | FLORES P@1 | MIRACL; SIB/STS shown when present in older results)")
    print("=" * 104)
    print(f"{'model':20}{'params(M)':>10}{'xfmr(M)':>9}{'SIB':>7}{'Belebele':>9}"
          f"{'FLORES':>8}{'STS':>7}{'MIRACL':>8}")
    for name, r in M.items():
        m = r.get("means", {})
        mir = (r.get("miracl") or {}).get("ndcg@10_mean")
        print(f"{name:20}{r.get('params', 0) / 1e6:>10.0f}"
              f"{r.get('transformer_params', 0) / 1e6:>9.0f}"
              f"{_f(m.get('sib'), 7)}{_f(m.get('belebele_ndcg@10'), 9)}{_f(m.get('flores_p@1'), 8)}"
              f"{_f(m.get('sts_spearman'), 7)}{_f(mir, 8)}")
    print("\nISO-COMPUTE byte − subword at matched size (Belebele | FLORES | STS):")
    for size in ("small", "base", "large"):
        b, s = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        if not (b and s):
            continue
        bm, sm = b.get("means", {}), s.get("means", {})
        def d(k):
            x, y = bm.get(k), sm.get(k)
            return f"{x - y:+.3f}" if (x is not None and y is not None) else "-"
        print(f"  {size:6} Belebele {d('belebele_ndcg@10')}  FLORES {d('flores_p@1')}  STS {d('sts_spearman')}")
    if any(r.get("qa_retrieval") for r in M.values()):
        print("\nDEEP QA-RETRIEVAL — nDCG@10 (Amharic-PR am · CIRAL ha + zero-shot so [cross-lingual]):")
        for name, r in M.items():
            qa = r.get("qa_retrieval")
            if qa:
                am = (qa.get("amharicpr") or {}).get("ndcg@10_mean")
                cl = (qa.get("ciral") or {}).get("per_lang") or {}
                ci_ha = (cl.get("ha") or {}).get("ndcg@10")
                ci_so = (cl.get("so") or {}).get("ndcg@10")
                print(f"  {name:20} Amharic-PR {_f(am, 7)}  CIRAL-ha {_f(ci_ha, 7)}  "
                      f"CIRAL-so {_f(ci_so, 7)}")
    eff = results.get("efficiency")
    if eff:
        print("\nTOKENIZATION (subword tax vs byte UTF-8 tax, vs English, on FLORES-1012):")
        for lang, e in eff.items():
            if e:
                print(f"  {lang:6} subword_tax {e['subword_tax']:>5}  byte_tax {e['byte_tax']:>5}  "
                      f"byte_seq_x {e['byte_seq_ratio']:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_lowresource.json")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=42000)
    ap.add_argument("--teacher", dest="teacher_name", default="sonar")
    ap.add_argument("--only", default=None,
                    help="comma-separated model names to train (e.g. 'byte-small,subword-large'); "
                         "empty string = precompute only")
    ap.add_argument("--no-baselines", dest="with_baselines", action="store_false")
    ap.add_argument("--no-efficiency", dest="with_efficiency", action="store_false")
    ap.add_argument("--pooling", default="mean", choices=["mean", "max", "attn"],
                    help="student sentence pooling ('attn' = lightweight multi-head attentive pool)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override steps for the model(s) trained in this call (int = same for all)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N log-windows without loss improvement (0 = off, iso-step)")
    ap.add_argument("--min-delta", type=float, dest="min_delta", default=1e-3,
                    help="minimum window-avg loss improvement that resets the patience counter")
    ap.add_argument("--boundary", default=None, choices=["teacher", "random"],
                    help="boundary-injection arm (byte students only; use a separate --out per arm)")
    a = ap.parse_args()
    only = None if a.only is None else [x for x in a.only.split(",") if x]
    run(out=a.out, smoke=a.smoke, device=a.device, n_per_lang=a.n_per_lang,
        teacher_name=a.teacher_name, only=only, with_baselines=a.with_baselines,
        with_efficiency=a.with_efficiency, pooling=a.pooling, steps=a.steps,
        patience=a.patience, min_delta=a.min_delta, boundary=a.boundary)


if __name__ == "__main__":
    main()
