"""Concurrent multi-model trainer for a big-VRAM GPU (e.g. the 96 GB RTX PRO 6000 Blackwell).

Trains the 6 students (byt5/mt5 × small/base/large) as CONCURRENT SUBPROCESSES to use the GPU's
headroom instead of one-at-a-time. Flow:
  1. precompute ONCE (balanced data + cached SONAR targets + efficiency table) — sequential;
  2. launch the students as `--only` subprocesses, <= `max_concurrent` at a time (VRAM bound), each
     reading the CACHED targets (so no SONAR reload) and writing its own results part-file;
  3. merge the part-files, run the baselines once, print the summary.

Honest speedup note: on a SINGLE GPU, concurrency helps most for the small/base students (they
underutilize the device); the two large students are compute-bound, so concurrency overlaps rather
than multiplies. Resumable: models already in the merged results (or with a finished part-file) are
skipped, so a killed/disconnected run just re-launches the unfinished ones.

  python -m byte_embed.run_parallel                      # all 6, <=3 concurrent
  python -m byte_embed.run_parallel --max-concurrent 4   # push the 96 GB card harder
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from byte_embed.run_lowresource import _grid, run


def _part(out, name):
    return f"{Path(out).with_suffix('')}_part_{name}.json"


def _merge(out, names):
    base = json.loads(Path(out).read_text(encoding="utf-8"))
    base.setdefault("models", {})
    for n in names:
        pf = Path(_part(out, n))
        if pf.exists():
            base["models"].update(json.loads(pf.read_text(encoding="utf-8")).get("models", {}))
    Path(out).write_text(json.dumps(base, indent=2, ensure_ascii=False))
    return base


def parallel(out="results/byte_lowresource.json", device="cuda", max_concurrent=3,
             n_per_lang=42000, teacher_name="sonar", pooling="mean", steps=None,
             patience=0, min_delta=1e-3):
    rows = _grid(steps)                              # [(name, backbone, steps, batch, ckpt), ...]
    names = [r[0] for r in rows]
    model_steps = {r[0]: r[2] for r in rows}         # per-model step count (size schedule)

    # 1) precompute ONCE: balanced data + SONAR targets (cached) + efficiency — no models/baselines
    print("=== [parallel] precompute: balanced data + teacher targets + efficiency table ===")
    run(out=out, device=device, n_per_lang=n_per_lang, teacher_name=teacher_name,
        only=[], with_baselines=False, with_efficiency=True)

    done = set(_merge(out, names).get("models", {}))     # pick up any finished part-files on resume
    pending = [n for n in names if n not in done]
    print(f"=== [parallel] to train: {pending or '(all done)'}  (<= {max_concurrent} at a time) ===")

    procs, logdir = {}, Path(out).parent
    while pending or procs:
        while pending and len(procs) < max_concurrent:
            n = pending.pop(0)
            logf = open(logdir / f"_log_{n}.txt", "w", encoding="utf-8")
            cmd = [sys.executable, "-u", "-m", "byte_embed.run_lowresource", "--only", n,  # -u: unbuffered
                   "--out", _part(out, n), "--device", device, "--no-baselines", "--no-efficiency",
                   "--n-per-lang", str(n_per_lang), "--teacher", teacher_name, "--pooling", pooling,
                   "--steps", str(model_steps[n]),
                   "--patience", str(patience), "--min-delta", str(min_delta)]
            procs[n] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf)
            print(f"  launched {n}  (log -> {logf.name})")
        for n, (p, logf) in list(procs.items()):
            if p.poll() is not None:
                logf.close()
                print(f"  {n} finished (returncode {p.returncode})")
                del procs[n]
        time.sleep(15)

    # 3) merge all part-files -> baselines once -> summary
    _merge(out, names)
    print("=== [parallel] merged all students; running baselines + summary ===")
    run(out=out, device=device, n_per_lang=n_per_lang, teacher_name=teacher_name,
        only=[], with_baselines=True, with_efficiency=False)
    print(f"\nSaved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/byte_lowresource.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-concurrent", type=int, dest="max_concurrent", default=3)
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=42000)
    ap.add_argument("--teacher", dest="teacher_name", default="sonar")
    ap.add_argument("--pooling", default="mean", choices=["mean", "max", "attn"])
    ap.add_argument("--steps", type=int, default=None, help="int = same steps for all sizes "
                    "(default = the per-size schedule small 50k / base 75k / large 100k)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N log-windows without loss improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, dest="min_delta", default=1e-3)
    a = ap.parse_args()
    parallel(out=a.out, device=a.device, max_concurrent=a.max_concurrent,
             n_per_lang=a.n_per_lang, teacher_name=a.teacher_name, pooling=a.pooling, steps=a.steps,
             patience=a.patience, min_delta=a.min_delta)


if __name__ == "__main__":
    main()
