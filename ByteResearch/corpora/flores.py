# FLORES-200 evaluation data for the study and control languages and the candidates, plus the fixed
# dev subsets. build_flores downloads mteb/flores dev.parquet and devtest.parquet at the pinned revision into
# ROOT/data/flores/raw, extracts those language columns, checks the 997/1012 row counts and
# saves one row-aligned JSON per split keyed by our short language codes.
# load_flores verifies the JSON against the manifest before returning it.
# dev_subset_ids gives the seeded FLORES-dev rows used for model selection (the same rows for every
# target and condition) and dev_halves gives the two seeded halves used by steering.
# Both are saved under ROOT/data/splits the first time and re-checked against the saved copy later.

from __future__ import annotations

import functools

import numpy as np

from ..core.config import ALL_LANGS, DATA_LANGS, PINS, DataConfig, language
from ..core.io import read_json, write_json_atomic
from .http import hf_fetch
from .manifest import begin_build, finish_build, verified_path


def build_flores(ctx) -> dict:
    # Input: Ctx. Downloads the pinned parquet files, extracts and checks the study and candidate language
    # columns and writes ROOT/data/flores/{dev,devtest}.json with its manifest. Output: the manifest.
    import pyarrow.parquet as pq
    pin = PINS["flores"]
    out = begin_build(ctx, ctx.paths.flores_dir())
    fetched = hf_fetch(pin["repo"], out / "raw", files=list(pin["files"].values()), repo_type="dataset",
                       revision=pin["revision"])
    columns = {code: language(code).flores for code in DATA_LANGS}
    rows = {}
    for split, name in pin["files"].items():
        table = pq.read_table(out / "raw" / name, columns=list(columns.values()))
        if table.num_rows != pin["rows"][split]:
            raise ValueError(f"FLORES {split}: {table.num_rows} rows, expected {pin['rows'][split]}")
        write_json_atomic(out / f"{split}.json", {code: table.column(col).to_pylist() for code, col in columns.items()})
        rows[split] = table.num_rows
    outputs = [f"raw/{n}" for n in pin["files"].values()] + [f"{s}.json" for s in pin["files"]]
    return finish_build(ctx, out, sources=[fetched], config={"columns": columns, "pin": pin},
                        outputs=outputs, stats={"rows": rows})


@functools.lru_cache(maxsize=8)
def _read_split(path: str, mtime_ns: int) -> dict:
    # Inputs: a verified split JSON path and its mtime (cache key). Output: its {code: [sentences]}
    # dict, cached per process and file version.
    return read_json(path)


def load_flores(ctx, split: str, langs=None) -> dict:
    # Inputs: Ctx, split ("dev" or "devtest") and optional language codes (default all study and control languages).
    # Output: {code: [sentences]} with rows aligned across languages, after a SHA check.
    path = verified_path(ctx.paths.flores_dir(), f"{split}.json")
    data = _read_split(str(path), path.stat().st_mtime_ns)
    return {code: list(data[code]) for code in (langs or ALL_LANGS)}


def _saved_ids(ctx, name: str, compute):
    # Inputs: Ctx, a split-file name under ROOT/data/splits and a zero-argument function computing the
    # ids. Output: the ids; written on first use and checked against the saved copy afterwards.
    path = ctx.paths.data("splits", name)
    ids = compute()
    saved = read_json(path)
    if saved is None:
        write_json_atomic(path, {"ids": ids})
        return ids
    if saved.get("ids") != ids:
        raise ValueError(f"{path} disagrees with the seeded computation; the seed or code changed")
    return ids


def _permutation(seed: int, stream: int, n: int) -> list:
    # Inputs: base seed, a stream id separating independent uses and the population size. Output: a
    # seeded permutation of range(n) as Python ints.
    return [int(i) for i in np.random.default_rng([seed, stream]).permutation(n)]


def dev_subset_ids(ctx, n_per_target: int = 63) -> list:
    # Inputs: Ctx and the subset size (pass ctx.config(TrainConfig).dev_per_target for the smoke size).
    # Output: sorted FLORES-dev row ids shared by every target and condition; smaller sizes are
    # prefixes of the same seeded permutation.
    n = int(n_per_target)
    seed = ctx.config(DataConfig).seed
    perm = lambda: sorted(_permutation(seed, 1, PINS["flores"]["rows"]["dev"])[:n])
    return _saved_ids(ctx, f"flores_dev_subset_n{n}_seed{seed}.json", perm)


def dev_halves(ctx) -> tuple:
    # Input: Ctx. Output: (ids_A, ids_B), a seeded split of all FLORES-dev rows into two sorted halves
    # (A gets the extra row when the count is odd), used to fit and select steering directions.
    seed = ctx.config(DataConfig).seed
    n = PINS["flores"]["rows"]["dev"]
    perm = _permutation(seed, 2, n)
    half = (n + 1) // 2
    ids = _saved_ids(ctx, f"flores_dev_halves_seed{seed}.json", lambda: [sorted(perm[:half]), sorted(perm[half:])])
    return ids[0], ids[1]
