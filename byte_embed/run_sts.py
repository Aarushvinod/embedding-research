"""STS-focused study: fine-tune byte students on real graded STS and eval test-only.

Tests whether the byte representation can do STS *when trained for it* (the fair test — our main runs
use a retrieval-leaning distillation objective). 8 languages (6 low-resource + en/zh anchors), all with
native graded train. CoSENT objective; per-language Spearman on the held-out test split; optional
whitening.

By default each student is **warm-started from its SONAR-distilled checkpoint** (strict=False, so the
encoder transfers even though the projection + pooling differ) — this gives a strong multilingual base.
Set init_ckpt=None to fine-tune the raw byt5 base instead.

  python -m byte_embed.run_sts                          # byte STS run
  python -m byte_embed.run_sts --smoke                  # ~5-min self-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# default warm-start checkpoints (the SONAR-distilled runs). {size} is filled per size.
_DEFAULT_INIT = {"byte": "checkpoints/byte-{size}_attn.pt"}


def _build(family, size, pooling, out_dim=1024):
    from byte_embed.model import ByteStudent
    return ByteStudent(f"google/byt5-{size}", out_dim=out_dim, pooling=pooling)


def run(family="byte", sizes=("small", "base"), pooling="mean", steps=2000, out=None,
        device="cuda", ckpt_dir="checkpoints", whiten=True, init_ckpt="default", smoke=False):
    import torch

    from byte_embed.sts_data import STS_LANGS, load_sts_train
    from byte_embed.sts_train import eval_sts_testonly, train_sts

    if smoke:
        sizes, steps = ("small",), 80
    out = out or f"results/sts_{family}_{pooling}.json"
    if init_ckpt == "default":
        init_ckpt = _DEFAULT_INIT.get(family)

    results = (json.loads(Path(out).read_text(encoding="utf-8")) if Path(out).exists()
               else {"family": family, "pooling": pooling, "langs": STS_LANGS})
    results.setdefault("models", {})

    train_langs = ["am", "rw", "en"] if smoke else STS_LANGS
    s1, s2, score, _ = load_sts_train(train_langs)

    for size in sizes:
        name = f"{family}-{size}"
        if name in results["models"]:
            print(f"=== {name}: already in results -> skip ===")
            continue
        print(f"\n=== {name}  (STS / CoSENT, {pooling} pool, {steps} steps) ===")
        student = _build(family, size, pooling).to(device)
        if init_ckpt:
            p = init_ckpt.format(size=size)
            if Path(p).exists():
                sd = torch.load(p, map_location=device)["model"]
                missing, unexpected = student.load_state_dict(sd, strict=False)
                print(f"  [init] warm-started from {p} "
                      f"(skipped {len(unexpected)} ckpt keys, {len(missing)} reinit)")
            else:
                print(f"  [init] {p} not found -> training from the raw {family} base")
        cpath = str(Path(ckpt_dir) / f"sts_{name}_{pooling}.pt")
        train_sts(student, s1, s2, score, device=device, steps=steps,
                  log_every=max(50, steps // 20), ckpt_path=cpath)

        bm = {"raw": eval_sts_testonly(student, langs=train_langs, whiten=False, device=device)}
        if whiten:
            print("  -- with whitening --")
            bm["whitened"] = eval_sts_testonly(student, langs=train_langs, whiten=True, device=device)
        bm.update(pooling=pooling, steps=steps, params=sum(p.numel() for p in student.parameters()))
        results["models"][name] = bm
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        w = f" | whitened={bm['whitened']['mean']}" if whiten else ""
        print(f"  saved {name}: STS(raw)={bm['raw']['mean']}{w}")
        del student
        torch.cuda.empty_cache()

    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    from byte_embed.sts_data import STS_LANGS
    M = results["models"]
    print("\n" + "=" * 86)
    print(f"STS-FOCUSED STUDY ({results.get('family')}, {results.get('pooling')} pool) — Spearman, test-only")
    print("=" * 86)
    hdr = f"{'model':14}{'mean(raw)':>11}{'mean(whit)':>12}  " + "".join(f"{l:>7}" for l in STS_LANGS)
    print(hdr)
    for name, r in M.items():
        raw = r.get("raw", {})
        wht = r.get("whitened", {})
        per = (wht or raw).get("per_lang", {})
        row = f"{name:14}{raw.get('mean', 0) or 0:>11.3f}{(wht.get('mean') or 0):>12.3f}  "
        row += "".join(f"{(per.get(l) if per.get(l) is not None else 0):>7.3f}" for l in STS_LANGS)
        print(row)
    print("(per-lang shown for whitened if available, else raw)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="byte", choices=["byte"])
    ap.add_argument("--sizes", default="small,base")
    ap.add_argument("--pooling", default="mean", choices=["mean", "max", "attn"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-whiten", dest="whiten", action="store_false")
    ap.add_argument("--init-ckpt", dest="init_ckpt", default="default",
                    help="'default' = the distilled checkpoint; 'none' = raw base; or a path with {size}")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    init = None if a.init_ckpt == "none" else a.init_ckpt
    run(family=a.family, sizes=tuple(a.sizes.split(",")), pooling=a.pooling, steps=a.steps,
        out=a.out, device=a.device, whiten=a.whiten, init_ckpt=init, smoke=a.smoke)


if __name__ == "__main__":
    main()
