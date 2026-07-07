"""Concurrent multi-model trainer — within one machine AND across sessions/jobs.

Trains students as CONCURRENT SUBPROCESSES (<= `max_concurrent` at a time, VRAM-bound) with the flow:
  1. precompute ONCE (balanced data + cached teacher targets + efficiency table) — sequential;
  2. launch each pending student as an `--only` subprocess, each reading the CACHED targets and
     writing its own results part-file (`{out}_part_{model}.json`);
  3. merge the part-files; optionally run the teacher baseline + summary.

MULTI-SESSION / MULTI-JOB partitioning (Colab sessions or SLURM jobs sharing one Drive/filesystem):
  * `only=[...]`   — train exactly these models in THIS session/job (None = the full grid).
  * `with_baselines=False` — skip the final baseline pass; run it in exactly ONE session (the last),
    or via a final `parallel(only=[], with_baselines=True)` call once every part-file exists.
  * Part-files and checkpoints are per-model (and per-boundary-arm via the `_b-{arm}` checkpoint
    namespace + separate `out`), so sessions training DISJOINT model lists never collide.
  * Sequencing rule: start session 1 alone until it logs "reusing cached targets" (the teacher
    precompute), THEN start the others — this avoids two sessions racing the teacher pass.
  * Boundary arms: pass boundary='teacher'|'random' (byte models only; use a separate --out per arm).

Resumable: models already in the merged results (or with a finished part-file) are skipped, so a
killed/disconnected session just re-runs the same cell/command.

  python -m byte_embed.run_parallel --teacher bge-m3 --pooling attn --steps 50000 \
         --out results/retrieval_bgem3.json --only byte-large,subword-large --no-baselines
  python -m byte_embed.run_parallel --teacher bge-m3 --pooling attn --steps 50000 \
         --out results/retrieval_bgem3_bteacher.json --boundary teacher
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
             patience=0, min_delta=1e-3, boundary=None, only=None, with_baselines=True):
    """`only` = the models THIS session trains (None = full grid; [] = merge/baseline-only pass).
    `with_baselines=False` for all but one session when partitioning across sessions/jobs."""
    rows = _grid(steps)                              # [(name, backbone, steps, batch, ckpt), ...]
    if boundary:
        rows = [r for r in rows if "byt5" in r[1]]   # boundary arms are byte-only
    all_names = [r[0] for r in rows]                 # merge scope: the whole (arm-filtered) grid
    if only is not None:
        unknown = [n for n in only if n not in all_names]
        if unknown:
            raise ValueError(f"unknown model names {unknown}; valid: {all_names}")
        rows = [r for r in rows if r[0] in only]
    names = [r[0] for r in rows]
    model_steps = {r[0]: r[2] for r in rows}         # per-model step count (size schedule)

    # 1) precompute ONCE: balanced data + teacher targets (cached) + efficiency — no models/baselines
    print("=== [parallel] precompute: balanced data + teacher targets + efficiency table ===")
    run(out=out, device=device, n_per_lang=n_per_lang, teacher_name=teacher_name,
        only=[], with_baselines=False, with_efficiency=True)

    done = set(_merge(out, all_names).get("models", {}))  # pick up any finished part-files on resume
    pending = [n for n in names if n not in done]
    print(f"=== [parallel] this session trains: {pending or '(nothing pending)'}  "
          f"(<= {max_concurrent} at a time; grid = {all_names}) ===")

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
            if boundary:
                cmd += ["--boundary", boundary]
            procs[n] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf)
            print(f"  launched {n}  (log -> {logf.name})")
        for n, (p, logf) in list(procs.items()):
            if p.poll() is not None:
                logf.close()
                print(f"  {n} finished (returncode {p.returncode})")
                del procs[n]
        if pending or procs:
            time.sleep(15)

    # 3) merge all part-files -> (optionally) teacher baseline -> summary
    _merge(out, all_names)
    if with_baselines:
        print("=== [parallel] merged; running the teacher baseline + summary ===")
    else:
        print("=== [parallel] merged (baselines skipped — run them in exactly one session) ===")
    run(out=out, device=device, n_per_lang=n_per_lang, teacher_name=teacher_name,
        only=[], with_baselines=with_baselines, with_efficiency=False)
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
    ap.add_argument("--boundary", default=None, choices=["teacher", "random"],
                    help="boundary-injection arm (byte models only; separate --out per arm)")
    ap.add_argument("--only", default=None,
                    help="comma-separated model names THIS session/job trains (default: full grid; "
                         "empty string = merge/baseline-only pass)")
    ap.add_argument("--no-baselines", dest="with_baselines", action="store_false",
                    help="skip the final teacher-baseline pass (run it in exactly one session)")
    a = ap.parse_args()
    only = None if a.only is None else [x for x in a.only.split(",") if x]
    parallel(out=a.out, device=a.device, max_concurrent=a.max_concurrent,
             n_per_lang=a.n_per_lang, teacher_name=a.teacher_name, pooling=a.pooling, steps=a.steps,
             patience=a.patience, min_delta=a.min_delta, boundary=a.boundary,
             only=only, with_baselines=a.with_baselines)


if __name__ == "__main__":
    main()
