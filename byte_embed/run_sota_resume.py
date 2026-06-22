"""Memory-SAFE resume of the SOTA run after the byt5-base VRAM crash.

The original run_sota froze Windows when byt5-base (batch 8) + MoCo queue + relational loss
exceeded the 12 GB laptop GPU and hung the display driver (a driver hang crashes the machine
before Python's try/except can catch the OOM). This resume makes a re-crash near-impossible:

  * the real culprit was AdamW's optimizer state: byt5-base's ~400M params cost ~6.4 GB (params +
    grads + two Adam moments) before activations, so even batch 4 peaked at 9.70 GB. Switching to
    ADAFACTOR (factored 2nd moments, native to T5/ByT5; ~50 MB state) drops it to ~6.5 GB peak;
  * caps PyTorch to 80% of VRAM (set_per_process_memory_fraction) so any residual blowup raises a
    CLEAN OutOfMemoryError that the try/except catches + skips -> worst case is a skipped config,
    never a system crash; + expandable_segments + gradient checkpointing (default-on);
  * byt5-base at BATCH 4, MATCHED 80k token-views (20000 steps) vs byte-small (5000x16);
  * re-runs ALL THREE configs (byte-small-rel, byte-small-norel, byte-base-rel) with Adafactor so
    the rel ablation (small rel vs norel) and matched-token scaling (small vs base) share ONE
    optimizer -> no mixed-optimizer confound. The crashed run's AdamW byte-small-rel (SIB 0.80 /
    Tatoeba 0.671 / STS 0.469) is kept only as an informal cross-check;
  * reports peak PyTorch VRAM per config for transparency.

  python -m byte_embed.run_sota_resume --probe   # ~9-min byt5-base batch-4 VRAM probe (cap-protected)
  python -m byte_embed.run_sota_resume           # full memory-safe resume
"""
from __future__ import annotations

import os

# must precede any torch import (run_scale import below may pull torch in transitively)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from byte_embed.run_scale import LANGS, _benchmark, _f  # noqa: E402

_CAP = 0.80
_LR = 1e-3  # Adafactor fixed LR (relative_step off); same for all configs -> clean comparison


def _setup_mem(torch):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_CAP)
        torch.cuda.empty_cache()
        print(f"[mem-safe] PyTorch VRAM capped at {_CAP:.0%} (~{_CAP * 12.2:.1f}/12.2 GB); "
              f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")


def probe(device="cuda", steps=1100):
    """Fill the batch-4 queue (maxlen 1024) and report peak VRAM, BEFORE committing to 20k steps.
    Cap-protected: if it would blow VRAM it raises a clean OOM here instead of crashing Windows."""
    import torch

    from byte_embed import config as C
    from byte_embed.data import load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent, TeacherEncoder

    _setup_mem(torch)
    teacher = TeacherEncoder(C.TEACHER, device=device)
    sents = load_sentences(["en", "tr", "ar", "ru"], n_per_lang=2000)
    print(f"PROBE byt5-base batch 4 + Adafactor + queue 4096 + rel, {steps} steps "
          f"(fills the 1024 queue; verify loss falls + peak VRAM) ...")
    torch.cuda.reset_peak_memory_stats()
    student = ByteStudent("google/byt5-base", out_dim=teacher.dim, max_bytes=C.MAX_BYTES).to(device)
    distill(student, teacher, sents, device=device, steps=steps, batch=4, lr=_LR,
            objective="both", queue_size=4096, rel_weight=1.0, optimizer="adafactor", log_every=200)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"PROBE OK. peak PyTorch VRAM = {peak:.2f} GB (cap {_CAP:.0%} = {_CAP * 12.2:.1f} GB; "
          f"+~0.6 GB CUDA ctx). Headroom looks {'GOOD' if peak < _CAP * 12.2 - 1 else 'TIGHT'}.")


def run(device="cuda", out="results/byte_sota.json", n_per_lang=12000):
    import torch

    from byte_embed import config as C
    from byte_embed.data import load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent, TeacherEncoder

    _setup_mem(torch)
    langs = LANGS
    # name, backbone, steps, batch, queue, rel  (all matched at 80k token-views, all Adafactor).
    # All three re-run with the SAME optimizer (Adafactor) so the rel ablation (small rel vs norel)
    # and the matched-token scaling (small vs base) share one optimizer = no mixed-optimizer confound.
    grid = [("byte-small-rel", "google/byt5-small", 5000, 16, 4096, 1.0),
            ("byte-small-norel", "google/byt5-small", 5000, 16, 4096, 0.0),
            ("byte-base-rel", "google/byt5-base", 20000, 4, 4096, 1.0)]

    teacher = TeacherEncoder(C.TEACHER, device=device)
    print(f"loading ~{n_per_lang}/lang sentences for {langs} ...")
    sents = load_sentences(langs, n_per_lang=n_per_lang)
    print(f"  {len(sents)} training sentences")

    results = {"langs": langs, "optimizer": "adafactor", "lr": _LR, "models": {}}
    for name, backbone, steps, batch, queue, rel in grid:
        print(f"\n=== {name}  ({backbone}, {steps}x{batch}={steps * batch} views, "
              f"queue {queue}, rel {rel}) ===")
        try:
            torch.cuda.reset_peak_memory_stats()
            student = ByteStudent(backbone, out_dim=teacher.dim, max_bytes=C.MAX_BYTES).to(device)
            params = sum(p.numel() for p in student.parameters())
            distill(student, teacher, sents, device=device, steps=steps, batch=batch, lr=_LR,
                    objective="both", queue_size=queue, rel_weight=rel, optimizer="adafactor",
                    log_every=max(200, steps // 5))

            def s_enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = _benchmark(s_enc, langs)
            bm.update(params=params, steps=steps, batch=batch, views=steps * batch, rel=rel,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            results["models"][name] = bm
            print(f"  peak PyTorch VRAM = {bm['peak_vram_gb']} GB")
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — clean OOM (or any error) skips, never crashes
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

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


def _summary(results):
    M = results["models"]
    print("\n" + "=" * 80)
    print("MEMORY-SAFE RESUME: matched-token scaling + relational STS + SOTA gap")
    print("=" * 80)
    print(f"{'model':18}{'params(M)':>10}{'SIB':>8}{'Tatoeba':>9}{'STS':>7}{'peakGB':>8}")
    for name, r in M.items():
        pm = f"{r.get('params', 0) / 1e6:.0f}" if r.get("params") else "-"
        pk = r.get("peak_vram_gb")
        pk = f"{pk:.1f}" if isinstance(pk, (int, float)) else "-"
        print(f"{name:18}{pm:>10}{_f(r.get('sib_mean'))}{_f(r.get('tatoeba_mean'), 9)}"
              f"{_f(r.get('sts_mean'), 7)}{pk:>8}")
    sr, sn = M.get("byte-small-rel"), M.get("byte-small-norel")
    if sr and sn:
        print("\nRELATIONAL STS effect (byte-small: rel − norel):")
        for m in ("sts_mean", "tatoeba_mean", "sib_mean"):
            if sr.get(m) is not None and sn.get(m) is not None:
                print(f"  {m:12} {sr[m]:.3f} − {sn[m]:.3f} = {sr[m] - sn[m]:+.3f}")
    br = M.get("byte-base-rel")
    if sr and br:
        print("MATCHED-TOKEN scaling (small → base, both 80k views, +rel):")
        for m in ("sib_mean", "tatoeba_mean", "sts_mean"):
            if sr.get(m) is not None and br.get(m) is not None:
                print(f"  {m:12} {sr[m]:.3f} → {br[m]:.3f} = {br[m] - sr[m]:+.3f}")
    te = M.get("mE5-teacher")
    if te and te.get("tatoeba_mean"):
        for nm in ("byte-small-rel", "byte-base-rel"):
            x = M.get(nm)
            if x and x.get("tatoeba_mean"):
                print(f"SOTA gap {nm}: Tatoeba {x['tatoeba_mean']:.3f} = "
                      f"{100 * x['tatoeba_mean'] / te['tatoeba_mean']:.0f}% of mE5 teacher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=12000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_sota.json")
    a = ap.parse_args()
    if a.probe:
        probe(device=a.device)
    else:
        run(device=a.device, out=a.out, n_per_lang=a.n_per_lang)


if __name__ == "__main__":
    main()
