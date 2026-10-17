# Result files: one JSON per (experiment, run_id, measurement[, variant]) with a fixed envelope.
# The envelope holds schema/meta fields, a status block (complete flag, finished units and a
# config fingerprint), the config, provenance, granular per_item data and computed aggregates.
# Provenance records the git commit, library versions, the data-manifest hash, host and GPU,
# plus creation and update times and anything the caller adds (e.g. the checkpoint used).
# ResultFile performs every write atomically under a file lock, treats an unreadable file as
# absent, and offers a locked read-modify-write for incremental updates.
# Large arrays are stored as .npy/.npz under ROOT/arrays and referenced by relative path, and locate
# is the one place that maps a measurement to its result and array paths, keeping --pilot runs apart.

from __future__ import annotations

import datetime as _dt
import functools
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

from .io import locked, read_json, sha256_file, write_json_atomic, write_npy_atomic, write_npz_atomic
from .paths import PACKAGE_DIR

SCHEMA = "byteresearch/result/v1"
_LIBS = ("torch", "transformers", "accelerate", "datasets", "numpy", "sacrebleu", "tokenizers")


def now_iso() -> str:
    # Output: the current UTC time as an ISO-8601 string with seconds precision.
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    # Output: the repository HEAD commit (with "-dirty" when the work tree has changes), or
    # "unknown" when git or the repository is unavailable. Cached per process.
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE_DIR, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PACKAGE_DIR,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@functools.lru_cache(maxsize=1)
def library_versions() -> dict:
    # Output: {"python": version, library: version-or-None} for the core libraries. Cached.
    from importlib import metadata
    out = {"python": platform.python_version()}
    for lib in _LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def gpu_name():
    # Output: the name of CUDA device 0, or None without CUDA.
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def data_manifest_hash(ctx):
    # Input: Ctx (or None). Output: SHA-256 of ROOT/data/DATA_MANIFEST.json, or None if absent.
    if ctx is None:
        return None
    path = ctx.paths.data_manifest()
    return sha256_file(path) if path.exists() else None


def provenance(ctx, **extra) -> dict:
    # Inputs: Ctx (may be None) and extra fields (e.g. checkpoint={...}). Output: the provenance
    # block: git commit, library versions, data-manifest hash, host, GPU, created/updated times,
    # world size, smoke/pilot flags and the extra fields.
    stamp = now_iso()
    prov = {
        "git_commit": git_commit(),
        "versions": library_versions(),
        "data_manifest_hash": data_manifest_hash(ctx),
        "host": socket.gethostname(),
        "gpu": gpu_name(),
        "created": stamp,
        "updated": stamp,
    }
    if ctx is not None:
        prov.update({"world_size": ctx.world.size, "smoke": ctx.smoke, "pilot": ctx.pilot, "seed": ctx.seed})
    prov.update(extra)
    return prov


def envelope(*, experiment, run_id, measurement, variant=None, unit=None, config=None, complete=False,
             units_done=(), fingerprint=None, provenance=None, per_item=None, aggregate=None) -> dict:
    # Inputs: all envelope fields as keywords. Output: the result dict in the canonical layout
    # {schema, experiment, run_id, measurement, variant, unit, status, config, provenance,
    # per_item, aggregate}.
    return {
        "schema": SCHEMA,
        "experiment": experiment,
        "run_id": run_id,
        "measurement": measurement,
        "variant": variant,
        "unit": unit or {},
        "status": {"complete": bool(complete), "units_done": list(units_done), "fingerprint": fingerprint},
        "config": config or {},
        "provenance": provenance or {},
        "per_item": per_item if per_item is not None else {},
        "aggregate": aggregate if aggregate is not None else {},
    }


class ResultFile:
    # One result JSON. Construct with its path, the envelope meta (experiment, run_id, measurement,
    # variant, unit) and an optional zero-argument provenance factory; result_file() does this.

    def __init__(self, path, meta: dict | None = None, provenance_fn=None):
        # Inputs: JSON path, envelope meta dict and provenance factory (defaults: empty meta, none).
        self.path = Path(path)
        self.meta = dict(meta or {})
        self.provenance_fn = provenance_fn

    def load(self):
        # Output: the parsed result dict, or None if missing or unreadable.
        data = read_json(self.path)
        return data if isinstance(data, dict) else None

    def exists(self) -> bool:
        # Output: whether a readable result file exists.
        return self.load() is not None

    def is_complete(self) -> bool:
        # Output: whether the file exists and its status.complete flag is set.
        data = self.load()
        return bool(data and data.get("status", {}).get("complete"))

    def _build(self, old, per_item, aggregate, config, complete, extra) -> dict:
        # Inputs: the previous contents (or None) and the new fields. Output: the full envelope,
        # keeping the original creation time; extra["status"] and extra["provenance"] merge into
        # those blocks and other extra keys are set at the top level.
        extra = dict(extra or {})
        prov = self.provenance_fn() if self.provenance_fn else {}
        prov.update(extra.pop("provenance", {}))
        created = (old or {}).get("provenance", {}).get("created")
        prov.update({"created": created or prov.get("created") or now_iso(), "updated": now_iso()})
        status = extra.pop("status", {})
        env = envelope(**{k: self.meta.get(k) for k in ("experiment", "run_id", "measurement", "variant", "unit")},
                       config=config, complete=complete, provenance=prov, per_item=per_item, aggregate=aggregate)
        env["status"].update(status)
        env.update(extra)
        return env

    def write(self, per_item, aggregate, *, config, complete, extra=None) -> dict:
        # Inputs: per_item and aggregate data, the config dict, the complete flag and optional
        # extra fields. Writes the envelope atomically under the file lock. Output: the envelope.
        with locked(self.path):
            env = self._build(self.load(), per_item, aggregate, config, complete, extra)
            write_json_atomic(self.path, env)
        return env

    def update(self, fn) -> dict:
        # Input: fn(data) -> new data (may mutate and return None). Performs a locked
        # read-modify-write (data is {} when the file is absent) and writes atomically.
        # Output: the stored dict.
        with locked(self.path):
            data = self.load() or {}
            new = fn(data)
            data = data if new is None else new
            write_json_atomic(self.path, data)
        return data


def result_file(ctx, experiment: str, run_id: str, measurement: str, variant=None, unit=None) -> ResultFile:
    # Inputs: Ctx, experiment, run id, measurement, optional variant and unit (Unit or dict).
    # Output: the ResultFile at ROOT/results/{experiment}/{run_id}/{measurement}[__{variant}].json
    # with its envelope meta and a provenance factory bound to ctx.
    unit_dict = unit.as_dict() if hasattr(unit, "as_dict") else (unit or {})
    meta = {"experiment": experiment, "run_id": run_id, "measurement": measurement, "variant": variant,
            "unit": unit_dict}
    return ResultFile(ctx.paths.results(experiment, run_id, measurement, variant), meta,
                      functools.partial(provenance, ctx))


def result_done(ctx, experiment: str, run_id: str, measurement: str, variant=None) -> bool:
    # Inputs: as result_file. Output: whether that result exists and is complete (an is_done helper).
    return ResultFile(ctx.paths.results(experiment, run_id, measurement, variant)).is_complete()


def locate(ctx, experiment: str, run_id: str, measurement: str = "", variant=None) -> dict:
    # Inputs: Ctx, experiment, run id, measurement and the measurement's own variant (or None).
    # Output: {"variant", "result", "arrays"}: under --pilot the variant gains "__pilot" (or becomes
    # "pilot") and arrays live in ROOT/arrays/{experiment}/{run_id}/pilot/, so pilot runs never mix
    # with, or count as, full runs.
    base = variant or None
    var = (f"{base}__pilot" if base else "pilot") if ctx.pilot else base
    arrays = ctx.paths.arrays(experiment, run_id, "pilot") if ctx.pilot else ctx.paths.arrays(experiment, run_id, "")
    return {"variant": var, "result": ctx.paths.results(experiment, run_id, measurement, var), "arrays": arrays}


def run_result(ctx, experiment: str, run_id: str, measurement: str, variant=None, unit=None) -> ResultFile:
    # Inputs: as result_file. Output: the pilot-aware ResultFile at locate()'s result path.
    return result_file(ctx, experiment, run_id, measurement, locate(ctx, experiment, run_id, measurement, variant)[
        "variant"], unit)


def run_done(ctx, experiment: str, run_id: str, measurement: str, variant=None) -> bool:
    # Inputs: as result_file. Output: whether the pilot-aware result is complete.
    return ResultFile(locate(ctx, experiment, run_id, measurement, variant)["result"]).is_complete()


def run_array_path(ctx, experiment: str, run_id: str, name: str):
    # Inputs: Ctx, experiment, run id and an array file name (may contain subfolders). Output: its
    # pilot-aware path under locate()'s arrays folder.
    return locate(ctx, experiment, run_id)["arrays"] / name


def save_run_array(ctx, experiment: str, run_id: str, name: str, array=None, **arrays) -> str:
    # Inputs: as save_array. Writes the array(s) at the pilot-aware path. Output: the root-relative ref.
    return _write_arrays(ctx, run_array_path(ctx, experiment, run_id, name), array, arrays)


def _write_arrays(ctx, path, array, arrays: dict) -> str:
    # Inputs: Ctx, the destination, one array (saved as .npy) or named arrays (saved as .npz).
    # Writes atomically. Output: the root-relative reference.
    if arrays:
        write_npz_atomic(path, **arrays)
    else:
        write_npy_atomic(path, array)
    return ctx.paths.relative(path)


def save_array(ctx, experiment: str, run_id: str, name: str, array=None, **arrays) -> str:
    # Inputs: Ctx, experiment, run id, file name, and either one array (saved as .npy) or named
    # arrays (saved as .npz). Writes atomically under ROOT/arrays (ignoring --pilot). Output: the
    # root-relative reference to store in a result JSON.
    return _write_arrays(ctx, ctx.paths.arrays(experiment, run_id, name), array, arrays)


def load_array(ctx, ref: str, mmap: bool = False):
    # Inputs: Ctx, a reference from save_array and whether to memory-map. Output: the ndarray
    # (.npy) or NpzFile (.npz).
    return np.load(ctx.paths.absolute(ref), mmap_mode="r" if mmap else None, allow_pickle=False)
