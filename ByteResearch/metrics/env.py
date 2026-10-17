# Builds the separate metrics environment and registers the `setup-metrics` stage.
# The venv lives at <metrics dir>/metrics (--metrics-dir, else BYTERESEARCH_METRICS_DIR, else ROOT/envs)
# and is created with --system-site-packages, so the CUDA torch is reused; uv is tried before venv.
# requirements-metrics.txt is installed with pip, constrained to the base torch version so the
# GPU torch is never replaced, and MetricX is cloned at its pinned commit into <metrics dir>/metricx.
# A self-check then scores identical and gibberish pairs: an identical hypothesis must get COMET
# above 0.9 and MetricX near 0, and gibberish must score worse on both.
# Each step is skipped when already done, and a marker JSON records the finished setup.

from __future__ import annotations

import shutil
import subprocess
import sys

from ..core.config import PINS
from ..core.io import read_json, sha256_file, write_json_atomic
from ..core.paths import PACKAGE_DIR
from ..core.registry import opt, register_stage
from ..core.results import now_iso
from .batching import METRIC_OPTIONS
from .neural import env_python, metrics_env_dir, metricx_dir, run_worker

REQUIREMENTS = PACKAGE_DIR / "requirements-metrics.txt"
METRICX_URL = "https://github.com/google-research/metricx.git"
MARKER = "byteresearch_env.json"
VERSIONS_SNIPPET = ("from importlib.metadata import version as v; import torch; print(v('unbabel-comet'), "
                    "v('transformers'), torch.__version__, torch.cuda.is_available(), v('numpy'))")
SELFCHECK_SRC = ["The weather is beautiful today.", "I am going to the market to buy fruit."]
SELFCHECK_REF = ["Hali ya hewa ni nzuri leo.", "Ninaenda sokoni kununua matunda."]
SELFCHECK_BAD = ["qzx vbn plorf tkk wqq", "zzzz kkkk wwww jjjj"]


def _run(cmd, cwd=None) -> None:
    # Inputs: an argv list and an optional working directory. Prints and runs it, raising on failure.
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def _output(cmd):
    # Input: an argv list (e.g. an interpreter with -c and a snippet that prints one line). Output:
    # the stripped standard output, or None when the command fails or cannot start.
    try:
        res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    except OSError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def base_python(ctx) -> str:
    # Input: Ctx. Output: the interpreter to base the venv on: --metrics-base-python, else the base
    # interpreter behind the current venv, else the current interpreter.
    return ctx.opt("metrics_base_python") or getattr(sys, "_base_executable", None) or sys.executable


def create_venv(env_dir, base: str) -> None:
    # Inputs: the venv directory and the base interpreter. Creates a --system-site-packages venv with
    # pip, using `uv venv --seed` when uv is available and `python -m venv` otherwise.
    uv = shutil.which("uv")
    if uv:
        try:
            _run([uv, "venv", "--seed", "--clear", "--system-site-packages", "--python", base, env_dir])
            return
        except subprocess.CalledProcessError as exc:
            print(f"uv venv failed ({exc}); falling back to python -m venv", flush=True)
    _run([base, "-m", "venv", "--clear", "--system-site-packages", env_dir])


def install_requirements(ctx, py, base: str) -> None:
    # Inputs: Ctx, the venv interpreter and the base interpreter. Installs requirements-metrics.txt
    # with pip, pinning torch to the base interpreter's version when it has one.
    cons = ctx.paths.tmp("metrics-constraints.txt")
    cons.parent.mkdir(parents=True, exist_ok=True)
    torch_version = _output([base, "-c", "import torch; print(torch.__version__)"])
    cons.write_text(f"torch=={torch_version}\n" if torch_version else "", encoding="utf-8")
    _run([py, "-m", "pip", "install", "-r", REQUIREMENTS, "-c", cons])


def clone_metricx(ctx):
    # Input: Ctx. Clones the MetricX repository into <metrics dir>/metricx (if absent) and checks out the
    # pinned commit. Output: the repository directory.
    repo, commit = metricx_dir(ctx), PINS["metricx"]["commit"]
    if not (repo / ".git").exists():
        _run(["git", "clone", METRICX_URL, repo])
    if _output(["git", "-C", repo, "rev-parse", "HEAD"]) != commit:
        _run(["git", "-C", repo, "checkout", "--quiet", commit])
    return repo


def self_check(ctx, metrics) -> dict:
    # Inputs: Ctx and the neural metrics to check. Scores identical and gibberish hypotheses and
    # asserts identical > 0.9 for COMET, identical < 2 for MetricX and gibberish worse for both.
    # Output: {metric: {"identical": [...], "gibberish": [...]}}.
    out = {}
    for metric in metrics:
        scores = run_worker(ctx, metric, SELFCHECK_SRC * 2, SELFCHECK_REF + SELFCHECK_BAD, SELFCHECK_REF * 2)
        same, bad = scores[:2], scores[2:]
        out[metric] = {"identical": same, "gibberish": bad}
        print(f"self-check {metric}: identical={same} gibberish={bad}", flush=True)
        ok = min(same) > 0.9 and max(bad) < min(same) if metric == "comet" else max(same) < 2.0 and min(bad) > max(same)
        if not ok:
            raise AssertionError(f"metrics self-check failed for {metric}: {out[metric]}")
    return out


def _fingerprint() -> dict:
    # Output: what a finished setup must match: the requirements hash and the MetricX commit.
    return {"requirements_sha256": sha256_file(REQUIREMENTS), "metricx_commit": PINS["metricx"]["commit"]}


def env_ready(ctx, unit=None) -> bool:
    # Inputs: Ctx and an unused unit. Output: whether the marker exists and matches the current
    # requirements and MetricX commit (the is_done of setup-metrics).
    marker = read_json(metrics_env_dir(ctx) / MARKER) or {}
    return all(marker.get(k) == v for k, v in _fingerprint().items())


def _selfcheck_metrics(ctx) -> list:
    # Input: Ctx. Output: the metrics named by --selfcheck-metrics ("none" for no self-check).
    spec = str(ctx.opt("selfcheck_metrics", "comet,metricx"))
    return [] if spec == "none" else [m for m in spec.split(",") if m]


@register_stage("setup-metrics", grid=(), is_done=env_ready, order=50, options=(
        opt("--metrics-base-python", default=None, help="base interpreter for the metrics venv"),
        opt("--selfcheck-metrics", default="comet,metricx", help="metrics to self-check, or none"),
        *METRIC_OPTIONS))
def setup_metrics(ctx, unit) -> None:
    # Inputs: Ctx and the empty unit. Creates the metrics venv, installs its requirements, clones
    # MetricX and runs the self-check, then writes the marker JSON. No output.
    env_dir, base = metrics_env_dir(ctx), base_python(ctx)
    py = env_python(env_dir)
    if ctx.force or not py.exists():
        create_venv(env_dir, base)
    install_requirements(ctx, py, base)
    clone_metricx(ctx)
    versions = _output([py, "-c", VERSIONS_SNIPPET])
    print(f"metrics env: comet/transformers/torch/cuda/numpy = {versions}", flush=True)
    checks = self_check(ctx, _selfcheck_metrics(ctx))
    write_json_atomic(env_dir / MARKER, {**_fingerprint(), "base_python": base, "versions": versions,
                                         "selfcheck": checks, "created": now_iso()})
