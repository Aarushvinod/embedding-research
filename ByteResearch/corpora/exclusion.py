# The evaluation exclusion set used by every training and Wikipedia filter.
# It holds the normalized form (core.text.normalize_for_match) of every FLORES dev and devtest
# sentence and every NTREX sentence in the study and candidate languages, so candidate pairs are filtered
# exactly as they would be as targets.
# build_exclusion writes the sorted set to ROOT/data/exclusion/normalized.jsonl.gz with a manifest,
# so the set and its SHA-256 are fixed once and shared by all pair files and Wikipedia samples.
# exclusion_set loads it (after a SHA check) and returns the set together with that SHA.
# is_excluded tests one raw text against the set.

from __future__ import annotations

import functools

from ..core.config import DATA_LANGS, NTREX_LANGS
from ..core.io import read_jsonl_gz, write_jsonl_gz_atomic
from ..core.text import normalize_for_match
from .flores import load_flores
from .manifest import begin_build, finish_build, read_manifest, verified_path
from .ntrex import load_ntrex

FILE = "normalized.jsonl.gz"


def exclusion_dir(ctx):
    # Input: Ctx. Output: ROOT/data/exclusion.
    return ctx.paths.data("exclusion")


def collect_texts(ctx) -> list:
    # Input: Ctx. Output: every FLORES dev/devtest sentence (study languages, controls and candidates) and
    # NTREX sentence (study languages and candidates), unnormalized.
    texts = []
    for split in ("dev", "devtest"):
        for rows in load_flores(ctx, split, DATA_LANGS).values():
            texts.extend(rows)
    for rows in load_ntrex(ctx, NTREX_LANGS).values():
        texts.extend(rows)
    return texts


def build_exclusion(ctx) -> dict:
    # Input: Ctx. Normalizes all evaluation sentences, writes the sorted unique set and its manifest
    # (recording the FLORES and NTREX output hashes it was built from). Output: the manifest.
    out = begin_build(ctx, exclusion_dir(ctx))
    texts = collect_texts(ctx)
    keys = sorted({k for k in map(normalize_for_match, texts) if k})
    write_jsonl_gz_atomic(out / FILE, keys)
    sources = [{"dataset": name, "outputs": {k: v["sha256"] for k, v in read_manifest(d)["outputs"].items()}}
               for name, d in (("flores", ctx.paths.flores_dir()), ("ntrex", ctx.paths.ntrex_dir()))]
    return finish_build(ctx, out, sources=sources, config={"normalize": "NFKC+casefold+whitespace"},
                        outputs=[FILE], stats={"sentences": len(texts), "unique_normalized": len(keys)})


@functools.lru_cache(maxsize=4)
def _read(path: str, sha: str) -> frozenset:
    # Inputs: the verified set path and its SHA-256 (cache key). Output: the frozenset of keys (cached).
    return frozenset(read_jsonl_gz(path))


def exclusion_set(ctx) -> tuple:
    # Input: Ctx. Output: (frozenset of normalized evaluation sentences, SHA-256 of the stored set).
    path = verified_path(exclusion_dir(ctx), FILE)
    sha = read_manifest(exclusion_dir(ctx))["outputs"][FILE]["sha256"]
    return _read(str(path), sha), sha


def is_excluded(text: str, excl) -> bool:
    # Inputs: a raw text and the exclusion set. Output: whether its normalized form is in the set.
    return normalize_for_match(text) in excl
