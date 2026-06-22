"""Matched-token scaling + relational STS + SOTA-gap push for ByteEmbed (local).

Targets the three gaps from the scale run:
  (1) MATCHED-TOKEN scaling — byte-small (5000 steps x batch 16) vs byte-base (10000 x 8): both see
      80k sentence-views, so a size comparison is no longer confounded by training budget.
  (2) fine-grained STS — a RELATIONAL (similarity-preserving) loss matches the student's pairwise
      cosine matrix to the teacher's; ABLATED (byte-base with vs without rel) to isolate its effect.
  (3) SOTA gap — more steps + a MoCo negative queue, read against the mE5 / LaBSE baselines.

12 languages / 7 scripts; SIB-200, Tatoeba, STS22. Reuses run_scale._benchmark; baselines are
reused from results/byte_scale.json (no re-eval).

  python -m byte_embed.run_sota          # full (local, ~overnight on a 5070 Ti)
  python -m byte_embed.run_sota --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from byte_embed.run_scale import LANGS, _benchmark, _f


def run(device="cuda", out="results/byte_sota.json", smoke=False, n_per_lang=12000):
    import torch

    from byte_embed import config as C
    from byte_embed.data import load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent, TeacherEncoder

    langs = ["en", "sw", "tr", "ar"] if smoke else LANGS
    # name, backbone, steps, batch, queue_size, rel_weight  (matched views = steps * batch)
    grid = [("byte-small-rel", "google/byt5-small", 5000, 16, 4096, 1.0),
            ("byte-base-rel", "google/byt5-base", 10000, 8, 4096, 1.0),
            ("byte-base-norel", "google/byt5-base", 10000, 8, 4096, 0.0)]
    if smoke:
        grid = [("byte-small-rel", "google/byt5-small", 120, 12, 256, 1.0),
                ("byte-small-norel", "google/byt5-small", 120, 12, 256, 0.0)]
        n_per_lang = 1200

    teacher = TeacherEncoder(C.TEACHER, device=device)
    print(f"loading ~{n_per_lang}/lang sentences for {langs} ...")
    sents = load_sentences(langs, n_per_lang=n_per_lang)
    print(f"  {len(sents)} training sentences")

    results = {"langs": langs, "models": {}}
    for name, backbone, steps, batch, queue, rel in grid:
        print(f"\n=== {name}  ({backbone}, {steps}x{batch}={steps * batch} views, "
              f"queue {queue}, rel {rel}) ===")
        try:
            student = ByteStudent(backbone, out_dim=teacher.dim,
                                  max_bytes=(160 if smoke else C.MAX_BYTES)).to(device)
            params = sum(p.numel() for p in student.parameters())
            distill(student, teacher, sents, device=device, steps=steps, batch=batch, lr=C.LR,
                    objective="both", queue_size=queue, rel_weight=rel,
                    log_every=max(100, steps // 4))

            def s_enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = _benchmark(s_enc, langs)
            bm.update(params=params, steps=steps, batch=batch, views=steps * batch, rel=rel)
            results["models"][name] = bm
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    # reuse baselines (mE5 teacher, LaBSE) from the scale run if present
    sp = Path("results/byte_scale.json")
    if sp.exists():
        prev = json.loads(sp.read_text(encoding="utf-8")).get("models", {})
        for b in ("mE5-teacher", "LaBSE"):
            if b in prev:
                results["models"][b] = prev[b]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    M = results["models"]
    print("\n" + "=" * 78)
    print("MATCHED-TOKEN SCALING + RELATIONAL STS + SOTA GAP")
    print("=" * 78)
    print(f"{'model':18}{'params(M)':>10}{'SIB':>8}{'Tatoeba':>9}{'STS':>7}")
    for name, r in M.items():
        pm = f"{r.get('params', 0) / 1e6:.0f}"
        print(f"{name:18}{pm:>10}{_f(r.get('sib_mean'))}{_f(r.get('tatoeba_mean'), 9)}"
              f"{_f(r.get('sts_mean'), 7)}")
    a, b = M.get("byte-base-rel"), M.get("byte-base-norel")
    if a and b and a.get("sts_mean") is not None and b.get("sts_mean") is not None:
        print(f"\nRELATIONAL STS effect (byte-base): rel {a['sts_mean']:.3f} − norel {b['sts_mean']:.3f}"
              f" = {a['sts_mean'] - b['sts_mean']:+.3f}  (also tatoeba {a.get('tatoeba_mean')} vs "
              f"{b.get('tatoeba_mean')})")
    s, bb = M.get("byte-small-rel"), M.get("byte-base-rel")
    if s and bb:
        print("MATCHED-TOKEN scaling (small -> base, both 80k views, +rel):")
        for m in ("sib_mean", "tatoeba_mean", "sts_mean"):
            if s.get(m) is not None and bb.get(m) is not None:
                print(f"  {m:12} {s[m]:.3f} -> {bb[m]:.3f} = {bb[m] - s[m]:+.3f}")
    te = M.get("mE5-teacher")
    if te and bb and bb.get("tatoeba_mean") and te.get("tatoeba_mean"):
        print(f"SOTA gap (byte-base-rel vs mE5 teacher): Tatoeba {bb['tatoeba_mean']:.3f} / "
              f"{te['tatoeba_mean']:.3f} = {100 * bb['tatoeba_mean'] / te['tatoeba_mean']:.0f}% of teacher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=12000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_sota.json")
    a = ap.parse_args()
    run(device=a.device, out=a.out, smoke=a.smoke, n_per_lang=a.n_per_lang)


if __name__ == "__main__":
    main()
