"""Scale-up + iso-compute curve + strong baselines (ACL/EMNLP-level eval) — runs locally.

Trains an ISO-COMPUTE grid — {byt5-small, byt5-base} (byte) vs {mt5-small, mt5-base} (subword) —
distilled from the frozen multilingual-e5 teacher with the balanced `both` objective, and
evaluates them alongside STRONG BASELINES (mE5 teacher, LaBSE) on the multilingual MTEB suite:
SIB-200 (classification), Tatoeba (bitext mining), STS22 (STS), across 12 languages / 7 scripts.

This answers the three ACL-path items: (1) scale (byt5-base, more steps, 12 langs);
(2) full benchmarks vs strong baselines; (3) iso-compute curve (byte vs subword at matched sizes).
Document retrieval (MIRACL/BEIR) needs corpus indexing and is left for cloud.

  python -m byte_embed.run_scale --steps 3000            # full local run (~overnight on a 5070 Ti)
  python -m byte_embed.run_scale --smoke                 # quick validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from byte_embed.benchmark import _tatoeba
from common.eval import sib_probe, sts_probe

LANGS = ["en", "tr", "sw", "bn", "ar", "ru", "hi", "vi", "de", "es", "fr", "zh"]


def _mean(d):
    v = [x for x in d.values() if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def _benchmark(enc, langs):
    out = {"sib": {}, "tatoeba": {}, "sts": {}}
    for lang in langs:
        out["sib"][lang] = sib_probe(enc, lang)
        out["tatoeba"][lang] = _tatoeba(enc, lang)
        out["sts"][lang] = sts_probe(enc, lang)
        print(f"    {lang}: sib={out['sib'][lang]} tat={out['tatoeba'][lang]} sts={out['sts'][lang]}")
    return {"sib": out["sib"], "sib_mean": _mean(out["sib"]),
            "tatoeba": out["tatoeba"], "tatoeba_mean": _mean(out["tatoeba"]),
            "sts": out["sts"], "sts_mean": _mean(out["sts"])}


def run(device="cuda", out="results/byte_scale.json", smoke=False, steps=3000, n_per_lang=8000):
    import torch

    from byte_embed import config as C
    from byte_embed.data import load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent, TeacherEncoder

    langs = ["en", "sw", "tr", "ar"] if smoke else LANGS
    if smoke:
        steps, n_per_lang = 150, 1200
    # iso-compute grid: (name, backbone, batch). base models use a smaller batch to fit 12 GB.
    grid = [("byte-small", "google/byt5-small", 16),
            ("subword-small", "google/mt5-small", 16),
            ("byte-base", "google/byt5-base", 8),
            ("subword-base", "google/mt5-base", 8)]
    if smoke:
        grid = [("byte-small", "google/byt5-small", 12),
                ("subword-small", "google/mt5-small", 12)]

    teacher = TeacherEncoder(C.TEACHER, device=device)

    def t_enc(xs):
        return teacher.encode(xs, as_tensor=False, device=device)

    print(f"loading ~{n_per_lang}/lang sentences for {langs} ...")
    sents = load_sentences(langs, n_per_lang=n_per_lang)
    print(f"  {len(sents)} training sentences")

    results = {"langs": langs, "steps": steps, "models": {}}

    # ---- iso-compute grid (trained students) -------------------------------------------------
    for name, backbone, batch in grid:
        print(f"\n=== train {name}  ({backbone}, batch {batch}) ===")
        try:
            student = ByteStudent(backbone, out_dim=teacher.dim,
                                  max_bytes=(160 if smoke else C.MAX_BYTES)).to(device)
            params = sum(p.numel() for p in student.parameters())
            distill(student, teacher, sents, device=device, steps=steps, batch=batch, lr=C.LR,
                    objective="both", log_every=max(100, steps // 4))

            def s_enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = _benchmark(s_enc, langs)
            bm.update(params=params, kind=("byte" if "byt5" in backbone else "subword"),
                      backbone=backbone)
            results["models"][name] = bm
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — one OOM/error shouldn't kill the run
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    # ---- strong baselines (eval-only, no training) -------------------------------------------
    from sentence_transformers import SentenceTransformer
    baselines = [("mE5-teacher", "intfloat/multilingual-e5-base", "query: "),
                 ("LaBSE", "sentence-transformers/LaBSE", "")]
    for name, model_id, prefix in baselines:
        print(f"\n=== baseline {name} ({model_id}) ===")
        try:
            m = SentenceTransformer(model_id, device=device)

            def b_enc(xs, _m=m, _p=prefix):
                return _m.encode([_p + x for x in xs], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

            bm = _benchmark(b_enc, langs)
            bm.update(params=sum(p.numel() for p in m.parameters()), kind="baseline", backbone=model_id)
            results["models"][name] = bm
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    M = results["models"]
    print("\n" + "=" * 78)
    print("SCALE-UP + ISO-COMPUTE + BASELINES")
    print("=" * 78)
    print(f"{'model':16}{'kind':9}{'params(M)':>10}{'SIB':>8}{'Tatoeba':>9}{'STS':>7}")
    for name, r in M.items():
        pm = f"{r.get('params', 0) / 1e6:.0f}"
        print(f"{name:16}{r.get('kind', '?'):9}{pm:>10}"
              f"{_f(r.get('sib_mean'))}{_f(r.get('tatoeba_mean'), 9)}{_f(r.get('sts_mean'), 7)}")
    # iso-compute byte vs subword at each size
    print("\niso-compute (byte − subword) at matched size:")
    for size in ("small", "base"):
        b, s = M.get(f"byte-{size}"), M.get(f"subword-{size}")
        if b and s:
            for metric in ("sib_mean", "tatoeba_mean", "sts_mean"):
                if b.get(metric) is not None and s.get(metric) is not None:
                    print(f"  {size:5} {metric:12} byte {b[metric]:.3f} − subword {s[metric]:.3f} "
                          f"= {b[metric] - s[metric]:+.3f}")


def _f(x, w=8):
    return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_scale.json")
    a = ap.parse_args()
    run(device=a.device, out=a.out, smoke=a.smoke, steps=a.steps, n_per_lang=a.n_per_lang)


if __name__ == "__main__":
    main()
