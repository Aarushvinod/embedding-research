# Neural translation metrics (COMET and MetricX) run in a separate Python environment.
# Both metrics need transformers below version 5, so they never load inside the main process.
# Instead this module writes the rows as JSONL under ROOT/tmp/<uuid>, launches the standalone
# worker script with the metrics interpreter in isolated mode and reads one score per row back.
# The interpreter comes from BYTERESEARCH_METRICS_PYTHON, else from the venv metrics/ in the metrics
# directory (--metrics-dir, else BYTERESEARCH_METRICS_DIR, else ROOT/envs), which also holds metricx/.
# Callers must free any GPU model before scoring, because the worker loads its own large model.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from ..core.config import PINS

ENV_VAR = "BYTERESEARCH_METRICS_PYTHON"
DIR_VAR = "BYTERESEARCH_METRICS_DIR"
WORKER = Path(__file__).with_name("worker.py")
METRICX_MAX_INPUT = 1536
DEFAULT_BATCH = {"comet": 64, "metricx": 8}


def env_python(env_dir) -> Path:
    # Input: a venv directory. Output: the path of its Python interpreter on this platform.
    env_dir = Path(env_dir)
    return env_dir / "Scripts" / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"


def envs_dir(ctx) -> Path:
    # Input: Ctx. Output: the directory holding the metrics venv and the MetricX checkout: --metrics-dir,
    # else $BYTERESEARCH_METRICS_DIR, else ROOT/envs (a local disk on Colab, where Drive has no symlinks).
    chosen = ctx.opt("metrics_dir") or os.environ.get(DIR_VAR)
    return Path(chosen).expanduser().resolve() if chosen else ctx.paths.metrics_env().parent


def metrics_env_dir(ctx) -> Path:
    # Input: Ctx. Output: the metrics venv directory, <metrics dir>/metrics.
    return envs_dir(ctx) / "metrics"


def metricx_dir(ctx) -> Path:
    # Input: Ctx. Output: <metrics dir>/metricx, where the pinned MetricX repository is cloned.
    return envs_dir(ctx) / "metricx"


def metrics_python(ctx) -> Path:
    # Input: Ctx. Output: the metrics interpreter ($BYTERESEARCH_METRICS_PYTHON, else the venv in the
    # metrics directory); raises FileNotFoundError with a setup hint when it is missing.
    py = Path(os.environ[ENV_VAR]) if os.environ.get(ENV_VAR) else env_python(metrics_env_dir(ctx))
    if not py.exists():
        raise FileNotFoundError(f"metrics python {py} not found; run `python -m ByteResearch.run setup-metrics` "
                                f"or set {ENV_VAR}")
    return py


def batch_size(ctx, metric: str) -> int:
    # Inputs: Ctx and a neural metric name. Output: its batch size from --comet-batch-size or
    # --metricx-batch-size, defaulting to 64 for COMET and 8 for MetricX.
    return int(ctx.opt(f"{metric}_batch_size", DEFAULT_BATCH[metric]))


def metricx_model(ctx) -> str:
    # Input: Ctx. Output: the MetricX checkpoint: --metricx-model when given (e.g. the bfloat16 copy
    # for a 12GB GPU), else the pinned google/metricx-24-hybrid-xl-v2p6.
    return ctx.opt("metricx_model", PINS["metricx"]["model"])


def signature(ctx, metric: str) -> str:
    # Inputs: Ctx and a neural metric name. Output: a signature string naming the model and settings.
    if metric == "comet":
        return f"comet|model:{PINS['comet']['model']}|ref:yes"
    mx = PINS["metricx"]
    return (f"metricx24|model:{metricx_model(ctx)}|tokenizer:{mx['tokenizer']}|commit:{mx['commit']}"
            f"|max_input_length:{METRICX_MAX_INPUT}|ref:yes")


def worker_command(ctx, metric: str, in_path, out_path, batch: int) -> list:
    # Inputs: Ctx, metric name, input/output JSONL paths and batch size. Output: the argv list that
    # runs the worker with the metrics interpreter in isolated mode (-I).
    mx = PINS["metricx"]
    return [str(metrics_python(ctx)), "-I", str(WORKER), "--metric", metric, "--input", str(in_path),
            "--output", str(out_path), "--batch-size", str(batch), "--comet-model", PINS["comet"]["model"],
            "--metricx-repo", str(metricx_dir(ctx)), "--metricx-model", metricx_model(ctx),
            "--metricx-tokenizer", mx["tokenizer"], "--max-input-length", str(METRICX_MAX_INPUT)]


def run_worker(ctx, metric: str, srcs, hyps, refs, batch: int | None = None) -> list:
    # Inputs: Ctx, "comet" or "metricx", aligned sources, hypotheses and references, and an optional
    # batch size. Writes {source, hypothesis, reference} JSONL to ROOT/tmp/<uuid>, runs the worker
    # and removes the folder afterwards. Output: one float score per row.
    if not len(hyps):
        return []
    work = ctx.paths.tmp(uuid.uuid4().hex)
    work.mkdir(parents=True, exist_ok=True)
    in_path, out_path = work / "in.jsonl", work / "out.jsonl"
    lines = [json.dumps({"source": s, "hypothesis": h, "reference": r}, ensure_ascii=False)
             for s, h, r in zip(srcs, hyps, refs)]
    in_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(worker_command(ctx, metric, in_path, out_path, batch or batch_size(ctx, metric)),
                       check=True)
        scores = [float(json.loads(x)["score"]) for x in out_path.read_text(encoding="utf-8").splitlines() if x]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if len(scores) != len(hyps):
        raise RuntimeError(f"{metric} worker returned {len(scores)} scores for {len(hyps)} rows")
    return scores
