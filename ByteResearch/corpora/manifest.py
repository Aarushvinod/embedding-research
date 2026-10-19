# Immutable dataset caches and their manifests under ROOT/data.
# Every dataset directory is built once: begin_build refuses to touch a directory whose manifest is
# complete unless --rebuild-data was passed, in which case the old directory is moved to
# ROOT/data/_superseded/<timestamp>/ first, and an incomplete leftover is simply cleared.
# finish_build hashes the outputs, writes manifest.json (sources, config, outputs, stats, code commit
# and timestamp), marks the outputs read-only and refreshes ROOT/data/DATA_MANIFEST.json.
# verified_path is what every loader uses: it checks an output's SHA-256 against the manifest once
# per process and file version, and purge permanently deletes a dataset that is no longer wanted.
# data_hash_for_direction combines the FLORES, NTREX and one hub-target pair manifest into the data hash
# that per-direction training checkpoints record (data_hash_for_condition covers all of a hub's targets).

from __future__ import annotations

import datetime as _dt
import functools
import os
import shutil
import stat
import sys
from pathlib import Path

from ..core.config import CONDITIONS, TARGETS, resolve_langs
from ..core.io import locked, make_readonly, read_json, sha256_file, sha256_json, write_json_atomic
from ..core.results import git_commit, now_iso

SCHEMA = "byteresearch/data-manifest/v1"
SUPERSEDED = "_superseded"


class ImmutableDataError(RuntimeError):
    # Raised when a stage would overwrite a dataset whose manifest is complete without --rebuild-data.
    pass


def read_manifest(directory):
    # Input: a dataset directory. Output: its parsed manifest.json dict, or None when missing/unreadable.
    data = read_json(Path(directory) / "manifest.json")
    return data if isinstance(data, dict) else None


def manifest_complete(directory) -> bool:
    # Input: a dataset directory. Output: whether its manifest exists and is marked complete.
    data = read_manifest(directory)
    return bool(data and data.get("complete"))


def rebuild_requested(ctx) -> bool:
    # Input: Ctx. Output: whether --rebuild-data was passed on the command line.
    return bool(ctx.opt("rebuild_data", False))


def supersede(ctx, directory) -> Path:
    # Inputs: Ctx and a dataset directory under ROOT/data. Moves the directory to
    # ROOT/data/_superseded/<UTC timestamp>/<relative path>. Output: the new location.
    directory = Path(directory)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rel = directory.resolve().relative_to(ctx.paths.data().resolve())
    dest = ctx.paths.data(SUPERSEDED, stamp, *rel.parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(dest))
    return dest


def begin_build(ctx, directory) -> Path:
    # Inputs: Ctx and the dataset directory about to be built. Refuses (ImmutableDataError) when a
    # complete manifest exists and --rebuild-data is absent; supersedes it when present; clears an
    # incomplete leftover. Output: the empty, existing directory.
    directory = Path(directory)
    if manifest_complete(directory) and not rebuild_requested(ctx):
        raise ImmutableDataError(f"{directory} has a complete manifest; pass --rebuild-data to supersede it")
    if manifest_complete(directory):
        print(f"[data] superseded {directory} -> {supersede(ctx, directory)}", flush=True)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_record(path) -> dict:
    # Input: an output file path. Output: {"sha256", "bytes"} describing it.
    path = Path(path)
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def finish_build(ctx, directory, *, sources, config, outputs, stats=None, extra=None, readonly=True) -> dict:
    # Inputs: Ctx, the dataset directory, source records (URLs, revisions, headers, hashes), the config
    # dict, output file names relative to the directory, optional stats and extra top-level fields.
    # Hashes the outputs, optionally marks them read-only, writes the complete manifest and refreshes
    # DATA_MANIFEST.json. Output: the manifest dict.
    directory = Path(directory)
    records = {name: output_record(directory / name) for name in outputs}
    manifest = {
        "schema": SCHEMA,
        "dataset": directory.resolve().relative_to(ctx.paths.data().resolve()).as_posix(),
        "complete": True,
        "created": now_iso(),
        "code_commit": git_commit(),
        "sources": list(sources),
        "config": config,
        "outputs": records,
        "stats": stats or {},
    }
    manifest.update(extra or {})
    for name in outputs if readonly else ():
        make_readonly(directory / name)
    write_json_atomic(directory / "manifest.json", manifest)
    update_data_manifest(ctx)
    return manifest


@functools.lru_cache(maxsize=256)
def _checked(path: str, size: int, mtime_ns: int, expected: str) -> bool:
    # Inputs: a file path, its size and mtime (cache key) and the expected SHA-256. Output: whether the
    # file hashes to the expected value (memoised per process and file version).
    return sha256_file(path) == expected


def verified_path(directory, name) -> Path:
    # Inputs: a dataset directory and an output name from its manifest. Output: the output path after
    # checking the manifest is complete and the file's SHA-256 matches; raises otherwise.
    directory = Path(directory)
    manifest = read_manifest(directory)
    if not (manifest and manifest.get("complete")):
        raise FileNotFoundError(f"{directory} has no complete manifest; run the matching data stage first")
    record = manifest["outputs"].get(name)
    path = directory / name
    if record is None or not path.exists():
        raise FileNotFoundError(f"{path} is not a recorded output of {directory}")
    st = path.stat()
    if not _checked(str(path), st.st_size, st.st_mtime_ns, record["sha256"]):
        raise ValueError(f"SHA-256 mismatch for {path}; the cache was modified")
    return path


def _force_remove(func, path, _exc) -> None:
    # Inputs: the os function that failed inside shutil.rmtree, its path and the error. Makes the path and
    # its parent writable (outputs are read-only) and retries the call.
    for p in (path, os.path.dirname(path)):
        os.chmod(p, os.stat(p).st_mode | stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    func(path)


def purge(ctx, directory) -> int:
    # Inputs: Ctx and a dataset directory under ROOT/data. Permanently deletes it, read-only outputs
    # included, and refreshes DATA_MANIFEST.json. Output: the bytes freed (0 when it did not exist).
    directory = Path(directory)
    if not directory.exists():
        return 0
    directory.resolve().relative_to(ctx.paths.data().resolve())
    size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    hook = {"onexc": _force_remove} if sys.version_info >= (3, 12) else {"onerror": _force_remove}
    shutil.rmtree(directory, **hook)
    update_data_manifest(ctx)
    return size


def source_outputs(directory) -> dict:
    # Input: a dataset directory. Output: {dataset name: {output: sha256}} of the input datasets its manifest
    # records in its sources (empty when it has no manifest).
    m = read_manifest(directory) or {}
    return {s["dataset"]: s.get("outputs") for s in m.get("sources", []) if isinstance(s, dict) and "dataset" in s}


def inputs_changed(directory, inputs: dict) -> bool:
    # Inputs: a dataset directory and {dataset name: input directory}. Output: whether any recorded input's
    # output hashes differ from that input's current manifest (so the dataset was built from older inputs).
    recorded = source_outputs(directory)
    current = {name: {k: v["sha256"] for k, v in (read_manifest(d) or {}).get("outputs", {}).items()}
               for name, d in inputs.items()}
    return any(recorded.get(name) != outs for name, outs in current.items())


def complete_manifests(ctx) -> dict:
    # Input: Ctx. Output: {dataset relative path: manifest} for every complete manifest under ROOT/data,
    # excluding superseded copies.
    base = ctx.paths.data()
    found = {}
    for path in sorted(base.rglob("manifest.json")) if base.exists() else ():
        rel = path.parent.relative_to(base).as_posix()
        data = read_json(path)
        if not rel.startswith(SUPERSEDED) and isinstance(data, dict) and data.get("complete"):
            found[rel] = data
    return found


def update_data_manifest(ctx) -> dict:
    # Input: Ctx. Rewrites ROOT/data/DATA_MANIFEST.json (under a lock) with each complete dataset's
    # output hashes and config hash; it holds no timestamps, so its hash depends only on content.
    # Output: the written dict.
    path = ctx.paths.data_manifest()
    with locked(path):
        datasets = {rel: {"outputs": {k: v["sha256"] for k, v in m["outputs"].items()},
                          "config_sha256": sha256_json(m.get("config", {}))}
                    for rel, m in complete_manifests(ctx).items()}
        data = {"schema": SCHEMA + "/combined", "datasets": datasets}
        write_json_atomic(path, data)
    return data


def _outputs_of(directory) -> dict:
    # Input: a dataset directory. Output: {output name: sha256}; raises when the manifest is incomplete.
    manifest = read_manifest(directory)
    if not (manifest and manifest.get("complete")):
        raise FileNotFoundError(f"{directory} has no complete manifest")
    return {k: v["sha256"] for k, v in manifest["outputs"].items()}


def _pairs_hash(ctx, hub: str, targets) -> str:
    # Inputs: Ctx, a hub code and target codes. Output: the SHA-256 over the output hashes of FLORES, NTREX and
    # the hub's pair manifest for each target. Wikipedia is deliberately excluded.
    blob = {
        "flores": _outputs_of(ctx.paths.flores_dir()),
        "ntrex": _outputs_of(ctx.paths.ntrex_dir()),
        "nllb": {f"{hub}__{x}": _outputs_of(ctx.paths.nllb_dir(hub, x)) for x in targets},
    }
    return sha256_json(blob)


def data_hash_for_condition(ctx, condition: str) -> str:
    # Inputs: Ctx and a condition key. Output: the data hash over FLORES, NTREX and the condition hub's pair
    # manifests for every target (the smoke targets under --smoke).
    return _pairs_hash(ctx, CONDITIONS[condition].hub, resolve_langs("all", TARGETS, ctx.smoke))


def data_hash_for_direction(ctx, condition: str, target: str) -> str:
    # Inputs: Ctx, a condition key and one of its targets. Output: the data hash a per-direction training
    # checkpoint records: FLORES, NTREX and that single hub-target pair manifest.
    if target not in CONDITIONS[condition].targets:
        raise ValueError(f"{target} is not a target of {condition}")
    return _pairs_hash(ctx, CONDITIONS[condition].hub, [target])
