# Artifact locations for every stage of the ByteResearch pipeline.
# The artifacts root comes from --artifacts, then the BYTERESEARCH_ARTIFACTS environment
# variable, then ByteResearch/artifacts; smoke runs live in a separate ROOT/smoke subtree.
# The Paths class is the only place that spells out the on-disk layout of data, checkpoints,
# results, large arrays, logs, the metrics environment and scratch space.
# Run identifiers are also defined here: model__condition for trained models (with an lr suffix
# for sweep runs) and size__condition for ByT5-versus-mT5 comparisons.
# Nothing here creates directories except the explicit ensure() helper.

from __future__ import annotations

import os
import re
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ENV_VAR = "BYTERESEARCH_ARTIFACTS"
COMPLETE_MARKER = "COMPLETE"
META_FILE = "byteresearch_meta.json"
MANIFEST_FILE = "manifest.json"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def artifacts_root(override=None, smoke: bool = False) -> Path:
    # Inputs: an optional explicit root and the smoke flag. Output: the resolved artifacts root
    # (override, else $BYTERESEARCH_ARTIFACTS, else ByteResearch/artifacts), with /smoke appended
    # in smoke mode.
    base = override or os.environ.get(ENV_VAR) or (PACKAGE_DIR / "artifacts")
    root = Path(base).expanduser().resolve()
    return root / "smoke" if smoke else root


def run_id(model: str, condition: str, lr=None) -> str:
    # Inputs: model key, condition key and an optional learning rate. Output: the run id,
    # "model__condition" or "model__condition__lr<g>" for LR-sweep runs.
    base = f"{model}__{condition}"
    return base if lr is None else f"{base}__lr{float(lr):g}"


def parse_run_id(rid: str) -> dict:
    # Input: a run id produced by run_id. Output: {"model", "condition", "lr"} with lr a float or None.
    parts = rid.split("__")
    if len(parts) not in (2, 3):
        raise ValueError(f"bad run id {rid!r}")
    lr = float(parts[2][2:]) if len(parts) == 3 and parts[2].startswith("lr") else None
    return {"model": parts[0], "condition": parts[1], "lr": lr}


def compare_id(size: str, condition: str) -> str:
    # Inputs: model size and condition key. Output: the comparison run id "size__condition".
    return f"{size}__{condition}"


def ensure(path: Path) -> Path:
    # Input: a directory path. Output: the same path, created (with parents) if missing.
    path.mkdir(parents=True, exist_ok=True)
    return path


class Paths:
    # Every artifact location under one root. Construct with the resolved root; each method
    # returns a Path without touching the filesystem (except complete_checkpoints, which lists).

    def __init__(self, root):
        # Input: the artifacts root (already including /smoke when relevant).
        self.root = Path(root)

    def data(self, *parts) -> Path:
        # Inputs: path parts below ROOT/data. Output: the joined path.
        return self.root.joinpath("data", *parts)

    def flores_dir(self) -> Path:
        # Output: ROOT/data/flores.
        return self.data("flores")

    def ntrex_dir(self) -> Path:
        # Output: ROOT/data/ntrex.
        return self.data("ntrex")

    def nllb_dir(self, hub: str, x: str) -> Path:
        # Inputs: hub and target codes. Output: ROOT/data/nllb/{hub}__{x}.
        return self.data("nllb", f"{hub}__{x}")

    def nllb_candidate_dir(self, hub: str, x: str) -> Path:
        # Inputs: hub and candidate codes. Output: ROOT/data/nllb_candidates/{hub}__{x}.
        return self.data("nllb_candidates", f"{hub}__{x}")

    def wiki_dir(self, lang: str) -> Path:
        # Input: language code. Output: ROOT/data/wiki/{lang}.
        return self.data("wiki", lang)

    def sacrebleu_dir(self) -> Path:
        # Output: ROOT/data/sacrebleu (sacreBLEU SPM model cache).
        return self.data("sacrebleu")

    def models_dir(self) -> Path:
        # Output: ROOT/data/models (downloaded auxiliary models such as GlotLID and LaBSE).
        return self.data("models")

    def manifest(self, directory: Path) -> Path:
        # Input: a data directory. Output: its manifest.json path.
        return Path(directory) / MANIFEST_FILE

    def data_manifest(self) -> Path:
        # Output: ROOT/data/DATA_MANIFEST.json, the combined manifest hashed into provenance.
        return self.data("DATA_MANIFEST.json")

    def checkpoints_dir(self, rid: str) -> Path:
        # Input: run id. Output: ROOT/checkpoints/{rid}.
        return self.root / "checkpoints" / rid

    def checkpoint(self, rid: str, step: int) -> Path:
        # Inputs: run id and global step. Output: ROOT/checkpoints/{rid}/checkpoint-{step}.
        return self.checkpoints_dir(rid) / f"checkpoint-{int(step)}"

    def best_checkpoint(self, rid: str) -> Path:
        # Input: run id. Output: ROOT/checkpoints/{rid}/best.
        return self.checkpoints_dir(rid) / "best"

    def complete_checkpoints(self, rid: str) -> list:
        # Input: run id. Output: [(step, path)] of checkpoint-* directories holding a COMPLETE
        # marker, sorted by step ascending (incomplete directories are ignored).
        base = self.checkpoints_dir(rid)
        if not base.is_dir():
            return []
        found = [(int(m.group(1)), p) for p in base.iterdir() if (m := _CKPT_RE.match(p.name))]
        return sorted((s, p) for s, p in found if (p / COMPLETE_MARKER).exists())

    def checkpoint_for(self, rid: str, which="best") -> Path:
        # Inputs: run id and which checkpoint ("best", "latest" = newest complete, or a step int).
        # Output: the checkpoint directory; raises FileNotFoundError when "latest" has none.
        if which == "best":
            return self.best_checkpoint(rid)
        if which == "latest":
            done = self.complete_checkpoints(rid)
            if not done:
                raise FileNotFoundError(f"no complete checkpoint under {self.checkpoints_dir(rid)}")
            return done[-1][1]
        return self.checkpoint(rid, int(which))

    def results(self, experiment: str, rid: str, measurement: str, variant=None) -> Path:
        # Inputs: experiment, run id, measurement and optional variant. Output:
        # ROOT/results/{experiment}/{rid}/{measurement}[__{variant}].json.
        name = measurement if variant in (None, "") else f"{measurement}__{variant}"
        return self.root / "results" / experiment / rid / f"{name}.json"

    def arrays(self, experiment: str, rid: str, name: str) -> Path:
        # Inputs: experiment, run id and a file name. Output: ROOT/arrays/{experiment}/{rid}/{name}.
        return self.root / "arrays" / experiment / rid / name

    def log(self, stage: str, rid: str) -> Path:
        # Inputs: stage name and run id. Output: ROOT/logs/{stage}/{rid}.log.
        return self.root / "logs" / stage / f"{rid}.log"

    def metrics_env(self) -> Path:
        # Output: ROOT/envs/metrics, the separate transformers<5 venv for COMET and MetricX.
        return self.root / "envs" / "metrics"

    def tmp(self, *parts) -> Path:
        # Inputs: path parts below ROOT/tmp. Output: the joined scratch path.
        return self.root.joinpath("tmp", *parts)

    def relative(self, path) -> str:
        # Input: a path under the root. Output: its POSIX path relative to the root, for storing
        # references inside result JSONs.
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def absolute(self, ref: str) -> Path:
        # Input: a root-relative reference produced by relative. Output: the absolute path.
        return self.root / ref
