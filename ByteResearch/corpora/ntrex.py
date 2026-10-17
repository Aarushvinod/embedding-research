# NTREX-128 evaluation data for the eleven study languages (eight targets and three hubs) and the candidates.
# build_ntrex downloads each language's file from raw.githubusercontent.com at the pinned commit.
# English uses newstest2019-src.eng.txt and the others newstest2019-ref.{code}.txt (fil for Tagalog, swa
# for Swahili, arb for Arabic, rus for Russian, and so on).
# Every file must have exactly 1997 lines.
# The raw files are kept under ROOT/data/ntrex/raw and a row-aligned ntrex.json keyed by our short codes is
# written next to them; load_ntrex verifies that JSON against the manifest before returning it.

from __future__ import annotations

import functools
from pathlib import PurePosixPath

from ..core.config import NTREX_LANGS, PINS, STUDY11, ntrex_file
from ..core.io import read_json, write_bytes_atomic, write_json_atomic
from .http import fetch_bytes
from .manifest import begin_build, finish_build, verified_path

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def ntrex_url(code: str) -> str:
    # Input: a study or candidate language code. Output: the pinned raw GitHub URL of its NTREX file.
    pin = PINS["ntrex"]
    return RAW_URL.format(repo=pin["repo"], commit=pin["commit"], path=ntrex_file(code))


def split_lines(data: bytes) -> list:
    # Input: raw file bytes. Output: the lines decoded as UTF-8 (BOM removed, "\r" stripped), without the
    # empty string a trailing newline would add.
    text = data.decode("utf-8-sig")
    lines = [line.rstrip("\r") for line in text.split("\n")]
    return lines[:-1] if lines and lines[-1] == "" else lines


def build_ntrex(ctx) -> dict:
    # Input: Ctx. Downloads, checks and saves the study and candidate NTREX files and ntrex.json with a
    # manifest. Output: the manifest.
    pin = PINS["ntrex"]
    out = begin_build(ctx, ctx.paths.ntrex_dir())
    sources, texts, outputs = [], {}, []
    for code in NTREX_LANGS:
        data, info = fetch_bytes(ntrex_url(code))
        lines = split_lines(data)
        if len(lines) != pin["rows"]:
            raise ValueError(f"NTREX {code}: {len(lines)} lines, expected {pin['rows']}")
        name = f"raw/{PurePosixPath(ntrex_file(code)).name}"
        write_bytes_atomic(out / name, data)
        sources.append({**info, "lang": code, "commit": pin["commit"], "path": ntrex_file(code)})
        texts[code] = lines
        outputs.append(name)
    write_json_atomic(out / "ntrex.json", texts)
    return finish_build(ctx, out, sources=sources, config={"pin": pin, "langs": list(NTREX_LANGS)},
                        outputs=outputs + ["ntrex.json"], stats={"rows": pin["rows"]})


@functools.lru_cache(maxsize=4)
def _read(path: str, mtime_ns: int) -> dict:
    # Inputs: the verified ntrex.json path and its mtime (cache key). Output: {code: [lines]} (cached).
    return read_json(path)


def load_ntrex(ctx, langs=None) -> dict:
    # Inputs: Ctx and optional language codes (default all eleven). Output: {code: [sentences]} with rows
    # aligned across languages, after a SHA check.
    path = verified_path(ctx.paths.ntrex_dir(), "ntrex.json")
    data = _read(str(path), path.stat().st_mtime_ns)
    return {code: list(data[code]) for code in (langs or STUDY11)}
