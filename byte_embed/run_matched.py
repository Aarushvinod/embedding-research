"""Matched-transformer control run: byte vs subword at IDENTICAL transformer params.

Both variants at each size use mT5-{size}'s transformer dims (random-init); only the input
embedding/tokenizer differs (byte 384-vocab vs subword 250k-vocab). Same SONAR distillation as the
main runs (reuses the cached targets), same eval battery. Answers "is byte's advantage just more
parameters?" — here the transformer is equal, so any byte win is representational.

From scratch (no pretraining) => absolute scores are lower than the pretrained main runs by design;
report it as an isolating ablation, not headline numbers.

  python -m byte_embed.run_matched --sizes small,base --steps 50000
  python -m byte_embed.run_matched --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OBJ = dict(objective="contrastive", queue_size=8192, rel_weight=0.0, optimizer="adamw")  # same as main runs (retrieval-only InfoNCE)


def _save(results, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def run(out="results/matched.json", sizes=("small", "base"), steps=50000, batch=64, pooling="mean",
        device="cuda", n_per_lang=42000, teacher_name="sonar", ckpt_dir="checkpoints",
        only=None, smoke=False, patience=0, min_delta=1e-3):
    import torch

    from byte_embed.config import STUDY_LANGS
    from byte_embed.data import load_balanced_sentences
    from byte_embed.distill import distill
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.matched import MatchedStudent
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval
    from byte_embed.teachers import (load_cached_targets, load_teacher, precompute_targets,
                                     targets_exist)

    langs = ["am", "rw", "en"] if smoke else STUDY_LANGS
    miracl_langs = ["te"] if smoke else ["en", "zh", "ar", "te"]
    if smoke:
        n_per_lang, steps, sizes = 1200, 80, ("small",)
    grid = [(f"{v}-{s}", v, s) for s in sizes for v in ("byte", "subword")]
    if only is not None:
        grid = [g for g in grid if g[0] in only]

    results = (json.loads(Path(out).read_text(encoding="utf-8")) if Path(out).exists()
               else {"langs": langs, "teacher": teacher_name,
                     "note": "matched-transformer control: mt5-{size} transformer, random init, byte vs subword input"})
    results.setdefault("models", {})

    # balanced data + cached SONAR targets (reuses the main runs' cache; no teacher reload) -----------
    balanced = load_balanced_sentences(langs, n_per_lang=n_per_lang, cache_dir=ckpt_dir)
    floor = min(len(v) for v in balanced.values())
    tag = f"{len(langs)}langs_{floor}"
    if targets_exist(ckpt_dir, teacher_name, tag):
        sentences, sent_langs, targets = load_cached_targets(ckpt_dir, teacher_name, tag)
    else:
        teacher = load_teacher(teacher_name, device=device)
        sentences, sent_langs, targets = precompute_targets(teacher, balanced, langs, ckpt_dir, tag)
        del teacher
        torch.cuda.empty_cache()
    teacher_dim = int(targets.shape[1])
    results["teacher_dim"] = teacher_dim

    for name, variant, size in grid:
        if name in results["models"]:
            print(f"=== {name}: already in results -> skip ===")
            continue
        try:
            student = MatchedStudent(variant, size=size, out_dim=teacher_dim, pooling=pooling).to(device)
            total, emb, xfmr = student.param_split()
            print(f"\n=== {name}  total={total / 1e6:.0f}M  input={emb / 1e6:.1f}M  "
                  f"TRANSFORMER={xfmr / 1e6:.1f}M  (random init, {steps}x{batch}) ===")
            torch.cuda.reset_peak_memory_stats()
            tsuf = "" if teacher_name == "sonar" else f"_{teacher_name}"   # per-teacher ckpt namespace
            cpath = str(Path(ckpt_dir) / f"matched_{name}_{pooling}{tsuf}.pt")
            hist = distill(student, None, sentences, device=device, steps=steps, batch=batch,
                           log_every=max(200, steps // 100), ckpt_path=cpath, targets=targets,
                           patience=patience, min_delta=min_delta, **OBJ)
            steps_run = hist[-1]["step"] if hist else steps

            def enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = eval_battery(enc, langs)
            bm["miracl"] = eval_miracl_langs(enc, miracl_langs, n_queries=250, distractors=20000,
                                             cache_dir=ckpt_dir)
            bm["qa_retrieval"] = eval_qa_retrieval(enc, n_queries=(20 if smoke else 250),
                                                   distractors=(500 if smoke else 20000),
                                                   cache_dir=ckpt_dir)
            bm.update(variant=variant, size=size, params=total, input_params=emb,
                      transformer_params=xfmr, steps=steps, steps_run=steps_run,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            results["models"][name] = bm
            _save(results, out)
            m = bm["means"]
            mir = (bm["miracl"] or {}).get("ndcg@10_mean")
            print(f"  saved {name}: xfmr={xfmr / 1e6:.0f}M steps={steps_run}/{steps} "
                  f"Belebele={m.get('belebele_ndcg@10')} FLORES={m.get('flores_p@1')} MIRACL={mir}")
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    M = results["models"]
    print("\n" + "=" * 96)
    print("MATCHED-TRANSFORMER CONTROL — byte vs subword at EQUAL transformer params (random init)")
    print("=" * 96)
    print(f"{'model':16}{'total(M)':>9}{'input(M)':>9}{'xfmr(M)':>9}{'SIB':>7}{'Belebele':>9}"
          f"{'FLORES':>8}{'STS':>7}{'MIRACL':>8}")
    for name, r in M.items():
        m = r.get("means", {})
        mir = (r.get("miracl") or {}).get("ndcg@10_mean")
        f = lambda x, w=7: (f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}}")  # noqa: E731
        print(f"{name:16}{r.get('params', 0) / 1e6:>9.0f}{r.get('input_params', 0) / 1e6:>9.1f}"
              f"{r.get('transformer_params', 0) / 1e6:>9.1f}"
              f"{f(m.get('sib'))}{f(m.get('belebele_ndcg@10'), 9)}{f(m.get('flores_p@1'), 8)}"
              f"{f(m.get('sts_spearman'))}{f(mir, 8)}")
    # byte - subword deltas at matched transformer
    for size in ("small", "base", "large"):
        b, s = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        if not (b and s):
            continue
        bx, sx = b.get("transformer_params", 0) / 1e6, s.get("transformer_params", 0) / 1e6
        bm, sm = b.get("means", {}), s.get("means", {})
        d = lambda k: (f"{bm[k] - sm[k]:+.3f}" if bm.get(k) is not None and sm.get(k) is not None else "-")  # noqa: E731
        bmir = (b.get("miracl") or {}).get("ndcg@10_mean")
        smir = (s.get("miracl") or {}).get("ndcg@10_mean")
        dmir = f"{bmir - smir:+.3f}" if bmir is not None and smir is not None else "-"
        print(f"\n{size}: byte transformer {bx:.0f}M vs subword {sx:.0f}M (matched) | "
              f"byte-subword: SIB {d('sib')} Belebele {d('belebele_ndcg@10')} "
              f"FLORES {d('flores_p@1')} STS {d('sts_spearman')} MIRACL {dmir}")
    if any(r.get("qa_retrieval") for r in M.values()):
        g = lambda x, w=7: (f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}}")  # noqa: E731
        print("\nDEEP QA-RETRIEVAL — nDCG@10 (Amharic-PR am · CIRAL ha + zero-shot so [cross-lingual]):")
        for name, r in M.items():
            qa = r.get("qa_retrieval")
            if qa:
                am = (qa.get("amharicpr") or {}).get("ndcg@10_mean")
                cl = (qa.get("ciral") or {}).get("per_lang") or {}
                print(f"  {name:16} Amharic-PR {g(am)}  CIRAL-ha {g((cl.get('ha') or {}).get('ndcg@10'))}  "
                      f"CIRAL-so {g((cl.get('so') or {}).get('ndcg@10'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/matched.json")
    ap.add_argument("--sizes", default="small,base")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pooling", default="mean", choices=["mean", "max", "attn"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=42000)
    ap.add_argument("--only", default=None, help="comma-separated model names (e.g. 'byte-small,subword-small')")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N log-windows without loss improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, dest="min_delta", default=1e-3)
    a = ap.parse_args()
    only = None if a.only is None else [x for x in a.only.split(",") if x]
    run(out=a.out, sizes=tuple(a.sizes.split(",")), steps=a.steps, batch=a.batch, pooling=a.pooling,
        device=a.device, n_per_lang=a.n_per_lang, only=only, smoke=a.smoke,
        patience=a.patience, min_delta=a.min_delta)


if __name__ == "__main__":
    main()
