"""Low-resource **UCD-vector** distillation study (SONAR teacher) — the A100 orchestrator.

This is the sibling of `run_lowresource.py`, exploring a representation we have NOT tried before:
instead of feeding raw UTF-8 bytes (ByteStudent), the student reads **Unicode Character Database
property vectors** per codepoint (`ucd.UCDStudent`) — General_Category, Combining_Class, Bidi_Class,
script/block, binary flags, plus a hashed-codepoint identity channel. The transformer body is the
SAME ByT5 encoder as the byte student, so only the *input representation* changes.

It reuses the byte study's machinery verbatim — the SAME cached SONAR targets (one teacher pass; same
tag => byte / subword / UCD all train against identical supervision), the SAME uniform eval battery
(SIB-200, Belebele, FLORES-1012, STS, MIRACL), and the SAME incremental/resumable save contract. So a
direct UCD-vs-byte-vs-subword table is a single `compare=True` run.

  python -m byte_embed.run_ucd --smoke         # ~5-min self-test of the whole UCD pipeline
  python -m byte_embed.run_ucd                 # the UCD study (ucd-{small,base,large}), resumable
  python -m byte_embed.run_ucd --compare       # also train byte/subword for the 3-way table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# same objective as the byte study (balanced contrastive + alignment + MoCo queue + relational STS).
OBJ = dict(objective="both", queue_size=8192, rel_weight=1.0, optimizer="adamw")

# per-size step schedule (bigger models train more), matched to run_lowresource so UCD is iso-step.
_DEFAULT_STEPS = {"small": 50000, "base": 75000, "large": 100000}


def _grid(steps=None, compare=False):
    sch = (_DEFAULT_STEPS if steps is None
           else steps if isinstance(steps, dict)
           else {s: int(steps) for s in _DEFAULT_STEPS})
    sizes = [(s, sch[s], True) for s in ("small", "base", "large")]
    # the UCD body reuses the ByT5 encoder at each size (same transformer as the byte student).
    grid = [(f"ucd-{s}", f"google/byt5-{s}", st, 64, ck, "ucd") for s, st, ck in sizes]
    if compare:                                              # optional: the byte + subword arms too
        grid += [(f"byte-{s}", f"google/byt5-{s}", st, 64, ck, "byte") for s, st, ck in sizes]
        grid += [(f"subword-{s}", f"google/mt5-{s}", st, 64, ck, "subword") for s, st, ck in sizes]
    return grid


def _f(x, w=8):
    return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def _save(results, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))


def _build_student(kind, backbone, teacher_dim, pooling):
    """UCD students use ucd.UCDStudent (property-vector front end); byte/subword reuse ByteStudent."""
    if kind == "ucd":
        from byte_embed.ucd import UCDStudent
        return UCDStudent(backbone, out_dim=teacher_dim, pooling=pooling)
    from byte_embed.model import ByteStudent
    return ByteStudent(backbone, out_dim=teacher_dim, pooling=pooling)


def _input_params(student, kind):
    """Input-representation params (the UCD analogue of a subword vocab table) + transformer params."""
    if kind == "ucd":
        ip = student.feat.input_params
    else:
        ip = student.enc.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in student.parameters())
    return total, ip, total - ip


def run(out="results/ucd_lowresource.json", smoke=False, device="cuda", n_per_lang=42000,
        ckpt_dir="checkpoints", teacher_name="sonar", miracl_q=250, miracl_extra=20000,
        only=None, with_baselines=True, with_efficiency=True, pooling="mean", steps=None,
        compare=False):
    """Train the UCD students (and, with compare=True, the byte/subword arms). Reuses the byte study's
    cached SONAR targets when present (same tag), so this never re-embeds the teacher if you already
    ran run_lowresource with the same langs/n_per_lang/teacher."""
    import torch

    from byte_embed.config import STUDY_LANGS
    from byte_embed.data import load_balanced_sentences
    from byte_embed.distill import distill
    from byte_embed.efficiency import fertility_table, profile_model
    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.teachers import (load_cached_targets, load_teacher, precompute_targets,
                                     targets_exist)

    langs = ["am", "rw", "en"] if smoke else STUDY_LANGS
    miracl_langs = (["te"] if smoke else ["en", "zh", "ar", "te"])
    grid = _grid(steps, compare=compare)
    if smoke:
        n_per_lang = 1200
        grid = [("ucd-small", "google/byt5-small", 80, 16, True, "ucd")]
        if compare:
            grid += [("byte-small", "google/byt5-small", 80, 16, True, "byte"),
                     ("subword-small", "google/mt5-small", 80, 16, True, "subword")]
    if only is not None:
        grid = [g for g in grid if g[0] in only]

    results = (json.loads(Path(out).read_text(encoding="utf-8"))
               if Path(out).exists() else {"langs": langs, "teacher": teacher_name})
    results.setdefault("models", {})

    # 1) balanced data + 2) cached teacher targets (reuses run_lowresource's cache if present) --------
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
    results["n_train"] = len(sentences)

    # 3) efficiency / tokenization table (once; no training) -----------------------------------------
    if with_efficiency and "efficiency" not in results:
        print("\n=== tokenization-efficiency table (FLORES-1012) ===")
        results["efficiency"] = fertility_table(langs)
        _save(results, out)

    # 4) students ------------------------------------------------------------------------------------
    for name, backbone, nsteps, batch, ckpt, kind in grid:
        if name in results["models"]:
            print(f"=== {name}: already in results -> skip ===")
            continue
        print(f"\n=== {name}  ({kind}: {backbone}, {nsteps}x{batch}, teacher={teacher_name}) ===")
        try:
            student = _build_student(kind, backbone, teacher_dim, pooling).to(device)
            params, in_params, xfmr_params = _input_params(student, kind)
            torch.cuda.reset_peak_memory_stats()
            cpath = str(Path(ckpt_dir) / f"{name}_{pooling}.pt") if ckpt else None
            distill(student, None, sentences, device=device, steps=nsteps, batch=batch,
                    log_every=max(200, nsteps // 100), ckpt_path=cpath, targets=targets, **OBJ)

            def enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = eval_battery(enc, langs)
            bm["miracl"] = eval_miracl_langs(enc, miracl_langs, n_queries=miracl_q,
                                             distractors=miracl_extra, cache_dir=ckpt_dir)
            bm["profile"] = profile_model(student, getattr(student, "tok", None),
                                          sentences[:256], device=device)
            bm.update(params=params, input_params=in_params, transformer_params=xfmr_params,
                      vocab_params=in_params,            # alias so shared tooling/figures still work
                      kind=kind, backbone=backbone, steps=nsteps,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            results["models"][name] = bm
            _save(results, out)
            m = bm["means"]
            mir = (bm["miracl"] or {}).get("ndcg@10_mean")
            print(f"  saved {name}: params={params/1e6:.0f}M input={in_params/1e6:.1f}M "
                  f"SIB={m['sib']} Belebele={m['belebele_ndcg@10']} FLORES={m['flores_p@1']} "
                  f"STS={m['sts_spearman']} MIRACL={mir} peak={bm['peak_vram_gb']}GB")
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — one model erroring shouldn't lose the others
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    # 5) reference baselines -------------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    baselines = [("mE5-base", "intfloat/multilingual-e5-base", "query: "),
                 ("LaBSE", "sentence-transformers/LaBSE", "")]
    if smoke or not with_baselines:
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
    print("LOW-RESOURCE UCD-vector study — SONAR teacher (SIB | Belebele nDCG@10 | FLORES P@1 | STS | MIRACL)")
    print("=" * 104)
    print(f"{'model':20}{'params(M)':>10}{'input(M)':>9}{'xfmr(M)':>9}{'SIB':>7}{'Belebele':>9}"
          f"{'FLORES':>8}{'STS':>7}{'MIRACL':>8}")
    for name, r in M.items():
        m = r.get("means", {})
        mir = (r.get("miracl") or {}).get("ndcg@10_mean")
        print(f"{name:20}{r.get('params', 0) / 1e6:>10.0f}"
              f"{r.get('input_params', 0) / 1e6:>9.1f}"
              f"{r.get('transformer_params', 0) / 1e6:>9.0f}"
              f"{_f(m.get('sib'), 7)}{_f(m.get('belebele_ndcg@10'), 9)}{_f(m.get('flores_p@1'), 8)}"
              f"{_f(m.get('sts_spearman'), 7)}{_f(mir, 8)}")
    # pairwise deltas vs byte / subword at matched size, when those arms are present (compare=True).
    for ref in ("byte", "subword"):
        rows = []
        for size in ("small", "base", "large"):
            u, b = M.get(f"ucd-{size}"), M.get(f"{ref}-{size}")
            if not (u and b):
                continue
            um, bm = u.get("means", {}), b.get("means", {})
            def d(k):
                x, y = um.get(k), bm.get(k)
                return f"{x - y:+.3f}" if (x is not None and y is not None) else "-"
            rows.append(f"  {size:6} Belebele {d('belebele_ndcg@10')}  FLORES {d('flores_p@1')}  "
                        f"STS {d('sts_spearman')}")
        if rows:
            print(f"\nUCD - {ref} at matched size (Belebele | FLORES | STS):")
            print("\n".join(rows))
    eff = results.get("efficiency")
    if eff:
        print("\nTOKENIZATION (subword tax vs byte UTF-8 tax, vs English, on FLORES-1012):")
        for lang, e in eff.items():
            if e:
                print(f"  {lang:6} subword_tax {e['subword_tax']:>5}  byte_tax {e['byte_tax']:>5}  "
                      f"byte_seq_x {e['byte_seq_ratio']:>5}")
        print("  (UCD is 1 position/character => it removes the byte UTF-8 multibyte penalty above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/ucd_lowresource.json")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=42000)
    ap.add_argument("--teacher", dest="teacher_name", default="sonar")
    ap.add_argument("--only", default=None,
                    help="comma-separated model names to train (e.g. 'ucd-small,ucd-large')")
    ap.add_argument("--no-baselines", dest="with_baselines", action="store_false")
    ap.add_argument("--no-efficiency", dest="with_efficiency", action="store_false")
    ap.add_argument("--pooling", default="mean", choices=["mean", "max", "attn"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--compare", action="store_true",
                    help="also train the byte + subword arms for a 3-way UCD/byte/subword table")
    a = ap.parse_args()
    only = None if a.only is None else [x for x in a.only.split(",") if x]
    run(out=a.out, smoke=a.smoke, device=a.device, n_per_lang=a.n_per_lang,
        teacher_name=a.teacher_name, only=only, with_baselines=a.with_baselines,
        with_efficiency=a.with_efficiency, pooling=a.pooling, steps=a.steps, compare=a.compare)


if __name__ == "__main__":
    main()
