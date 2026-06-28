"""Low-resource byte-vs-subword distillation study (SONAR teacher) — the A100 orchestrator.

Trains 6 students — byt5 / mt5 × {small, base, large} — distilled from a frozen **SONAR** teacher
(1024-d; LaBSE fallback) on 9 languages (6 low-resource: te/ta/mr/am/ha/rw + 3 high-resource
anchors: en/zh/ar). EQUAL max-min balanced training data (~42k/lang) and CACHED teacher targets
(one teacher pass; both students train against the identical vectors). Each student is scored on the
uniform battery — SIB-200 (classification), Belebele (retrieval), FLORES-1012 (bitext), STS — plus
MIRACL deep retrieval (en/zh/ar/te) and a compute profile. A one-time tokenization-efficiency table
quantifies the subword tax vs the byte UTF-8 cost.

INCREMENTAL + RESUMABLE: the results JSON is rewritten after every model, a model already present is
skipped, and `*-large` checkpoints model+optimizer (a Colab disconnect just means re-running the cell).

  python -m byte_embed.run_lowresource --smoke      # ~5-min self-test of the whole pipeline
  python -m byte_embed.run_lowresource              # the full study (resumable)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# balanced objective + big negative queue + relational (STS) term; AdamW on the A100.
OBJ = dict(objective="both", queue_size=8192, rel_weight=1.0, optimizer="adamw")

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
        only=None, with_baselines=True, with_efficiency=True, pooling="mean", steps=None):
    """Train (a subset of) the 6 students. `only` = list of model names to train (None = all,
    [] = precompute-only). `with_baselines`/`with_efficiency` are turned OFF for parallel workers
    (the orchestrator does those once). Workers SKIP loading the teacher when targets are cached."""
    import torch

    from byte_embed.config import STUDY_LANGS
    from byte_embed.data import load_balanced_sentences
    from byte_embed.distill import distill
    from byte_embed.efficiency import fertility_table, profile_model
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.model import ByteStudent
    from byte_embed.qa_retrieval import eval_qa_retrieval
    from byte_embed.teachers import (load_cached_targets, load_teacher, precompute_targets,
                                     targets_exist)

    langs = ["am", "rw", "en"] if smoke else STUDY_LANGS
    miracl_langs = (["te"] if smoke else ["en", "zh", "ar", "te"])   # the 4 of our 9 in MIRACL
    grid = _grid(steps)
    if smoke:
        n_per_lang = 1200
        grid = [("byte-small", "google/byt5-small", 80, 16, True),
                ("subword-small", "google/mt5-small", 80, 16, True)]
    if only is not None:
        grid = [g for g in grid if g[0] in only]

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
        results["efficiency"] = fertility_table(langs)
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
            cpath = str(Path(ckpt_dir) / f"{name}_{pooling}.pt") if ckpt else None  # tag by config
            distill(student, None, sentences, device=device, steps=steps, batch=batch,
                    log_every=max(200, steps // 100), ckpt_path=cpath, targets=targets, **OBJ)

            def enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = eval_battery(enc, langs)
            bm["miracl"] = eval_miracl_langs(enc, miracl_langs, n_queries=miracl_q,
                                             distractors=miracl_extra, cache_dir=ckpt_dir)
            bm["qa_retrieval"] = eval_qa_retrieval(enc, n_queries=(20 if smoke else miracl_q),
                                                   distractors=(500 if smoke else miracl_extra),
                                                   cache_dir=ckpt_dir)
            bm["profile"] = profile_model(student, student.tok, sentences[:256], device=device)
            bm.update(params=params, vocab_params=vocab_params,
                      transformer_params=params - vocab_params,
                      kind=("byte" if "byt5" in backbone else "subword"),
                      backbone=backbone, steps=steps,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            results["models"][name] = bm
            _save(results, out)
            m = bm["means"]
            mir = (bm["miracl"] or {}).get("ndcg@10_mean")
            print(f"  saved {name}: params={params/1e6:.0f}M SIB={m['sib']} "
                  f"Belebele={m['belebele_ndcg@10']} FLORES={m['flores_p@1']} "
                  f"STS={m['sts_spearman']} MIRACL={mir} peak={bm['peak_vram_gb']}GB")
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — one model erroring shouldn't lose the others
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    # 5) reference baselines (lang-agnostic encoders that fit the eval interface) ------------------
    from sentence_transformers import SentenceTransformer
    baselines = [("mE5-base", "intfloat/multilingual-e5-base", "query: "),
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

            bm = eval_battery(benc, langs)
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
    print("LOW-RESOURCE BYTE vs SUBWORD — SONAR teacher (SIB | Belebele nDCG@10 | FLORES P@1 | STS | MIRACL)")
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
        print("\nRAG-RETRIEVAL — QA open-retrieval nDCG@10 (IndicQA mr/ta/te · Mr.TyDi te · Amharic-PR am):")
        for name, r in M.items():
            qa = r.get("qa_retrieval")
            if qa:
                iq = (qa.get("indicqa") or {}).get("ndcg@10_mean")
                mt = (qa.get("mrtydi") or {}).get("ndcg@10_mean")
                am = (qa.get("amharicpr") or {}).get("ndcg@10_mean")
                print(f"  {name:20} IndicQA {_f(iq, 7)}  Mr.TyDi {_f(mt, 7)}  Amharic-PR {_f(am, 7)}")
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
    a = ap.parse_args()
    only = None if a.only is None else [x for x in a.only.split(",") if x]
    run(out=a.out, smoke=a.smoke, device=a.device, n_per_lang=a.n_per_lang,
        teacher_name=a.teacher_name, only=only, with_baselines=a.with_baselines,
        with_efficiency=a.with_efficiency, pooling=a.pooling, steps=a.steps)


if __name__ == "__main__":
    main()
