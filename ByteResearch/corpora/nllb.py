# NLLB-mined training pairs: 24 files (three hubs times eight targets) streamed from AllenAI and CCMatrix.
# parse_line reads one TSV line in the file's own layout (AllenAI: text, text, laser score, ...; CCMatrix:
# score, text, text) and orients it to (hub_text, x_text, score) using core.config.NLLB_SOURCES.
# build_pairs streams the score-sorted file with GzLineStream and applies the ordered filter chain of
# corpora.filters with GlotLID and LaBSE, stopping at the cap (100k, 2k under --smoke, or --pairs).
# It writes ROOT/data/nllb/{hub}__{X}/pairs.jsonl.gz read-only, with a manifest holding the stream provenance,
# filter config (LaBSE dtype and LID rule with both sides' GlotLID labels, so both enter the data hashes),
# counters and shortfall.
# load_pairs verifies and reads a pair file, and the filter-table helpers print and save the per-pair
# survival table and the per-target summary to ROOT/data/nllb/FILTER_TABLE.json.

from __future__ import annotations

import dataclasses
import functools
import math
import time

from ..core.config import HUBS, LANGUAGES, NLLB_SOURCES, TARGETS, DataConfig
from ..core.io import read_json, read_jsonl_gz, sha256_json, write_json_atomic, write_jsonl_gz_atomic
from .assets import labse_cosines, labse_dtype, load_labse
from .exclusion import exclusion_set
from .filters import STEPS, FilterSpec, run_filters, survival
from .http import GzLineStream
from .lid import lid_record, load_aliases, load_glotlid, predict
from .manifest import begin_build, finish_build, manifest_complete, read_manifest, verified_path

PAIRS_FILE = "pairs.jsonl.gz"
TABLE_FILE = "FILTER_TABLE.json"
BLOCK = 5000


def parse_line(line: str, src) -> tuple | None:
    # Inputs: one TSV line and its NllbSource. Output: (hub_text, x_text, score) with texts stripped, or
    # None when the line is malformed (wrong column count, non-numeric or non-finite score, an empty
    # side or undecodable bytes).
    cols = line.rstrip("\r\n").split("\t")
    if len(cols) != src.n_cols:
        return None
    score_col, a, b = (2, 0, 1) if src.host == "allenai" else (0, 1, 2)
    try:
        score = float(cols[score_col])
    except ValueError:
        return None
    first, second = cols[a].strip(), cols[b].strip()
    hub, x = (first, second) if src.hub_first else (second, first)
    if not (math.isfinite(score) and hub and x) or "�" in hub or "�" in x:
        return None
    return hub, x, score


def pairs_cap(ctx) -> int:
    # Input: Ctx. Output: the number of pairs to keep (--pairs, else DataConfig.nllb_cap).
    return int(ctx.opt("pairs") or ctx.config(DataConfig).nllb_cap)


def filter_spec(ctx, hub: str, x: str) -> FilterSpec:
    # Inputs: Ctx, hub and target codes. Output: the FilterSpec with DataConfig thresholds, both scripts,
    # both sides' accepted GlotLID labels (corpora.lid.load_aliases) and the shared exclusion set.
    cfg, aliases = ctx.config(DataConfig), load_aliases(ctx)
    return FilterSpec(hub=hub, x=x, hub_script=LANGUAGES[hub].script, x_script=LANGUAGES[x].script,
                      min_chars=cfg.min_chars, max_chars=cfg.max_chars, lid_min_prob=cfg.glotlid_min_prob,
                      lid_top_k=cfg.glotlid_top_k, hub_lid=tuple(sorted(aliases[hub])), x_lid=tuple(sorted(aliases[x])),
                      labse_min=cfg.labse_min_cos, exclusion=exclusion_set(ctx)[0])


def model_functions(ctx) -> tuple:
    # Input: Ctx. Output: (lid_fn, labse_fn) wrapping the cached GlotLID (top glotlid_top_k labels per
    # text) and LaBSE models.
    lid_model, labse = load_glotlid(ctx), load_labse(ctx)
    cfg = ctx.config(DataConfig)
    batch, k = cfg.labse_batch, cfg.glotlid_top_k
    return (lambda texts: predict(lid_model, texts, k=k),
            lambda left, right: labse_cosines(labse, left, right, batch))


def asset_revisions(ctx) -> dict:
    # Input: Ctx. Output: {asset: resolved repository revision} for GlotLID and LaBSE.
    return {name: read_manifest(ctx.paths.models_dir() / name)["sources"][0]["revision"]
            for name in ("glotlid", "labse")}


def _progress(label: str, stream, t0: float):
    # Inputs: a pair label, the stream and the start time. Output: a progress(state) callback printing
    # lines read, pairs kept, compressed megabytes and throughput after each block.
    def report(state):
        # Input: the FilterState. Prints one progress line.
        secs = max(time.time() - t0, 1e-9)
        print(f"[data.nllb] {label}: read {state.lines:,} kept {len(state.rows):,} "
              f"({stream.consumed / 2**20:.1f} MB, {state.lines / secs:,.0f} lines/s)", flush=True)
    return report


def build_pairs(ctx, hub: str, x: str) -> dict:
    # Inputs: Ctx, hub and target codes. Streams, filters and saves the pair file with its manifest
    # (a shortfall is recorded, not fatal). Output: the manifest.
    src = NLLB_SOURCES[(hub, x)]
    out = begin_build(ctx, ctx.paths.nllb_dir(hub, x))
    spec, cap = filter_spec(ctx, hub, x), pairs_cap(ctx)
    lid_fn, labse_fn = model_functions(ctx)
    stream, t0 = GzLineStream(src.url), time.time()
    entries = ((n, parse_line(line, src)) for n, line in enumerate(stream, 1))
    try:
        result = run_filters(entries, spec, lid_fn, labse_fn, cap, block=min(BLOCK, cap),
                             progress=_progress(f"{hub}__{x}", stream, t0))
    finally:
        entries.close()
        stream.close()
    write_jsonl_gz_atomic(out / PAIRS_FILE, result["rows"])
    kept = len(result["rows"])
    config = {"source": dataclasses.asdict(src), "cap": cap, "block": min(BLOCK, cap),
              "filters": {k: v for k, v in dataclasses.asdict(spec).items() if k != "exclusion"},
              "exclusion_sha256": exclusion_set(ctx)[1], "assets": asset_revisions(ctx), "device": ctx.device,
              "labse_dtype": labse_dtype(ctx), "lid": lid_record(ctx, load_aliases(ctx), (hub, x))}
    stats = {"counts": result["counts"], "survival": survival(result["counts"], result["lines_read"]),
             "lines_read": result["lines_read"], "order_violations": result["order_violations"],
             "kept": kept, "shortfall": max(cap - kept, 0), "exhausted": result["exhausted"],
             "blocks": result["blocks"], "seconds": round(time.time() - t0, 1)}
    source = {**stream.stats(), "host": src.host, "n_cols": src.n_cols, "hub_first": src.hub_first}
    return finish_build(ctx, out, sources=[source], config=config, outputs=[PAIRS_FILE], stats=stats)


@functools.lru_cache(maxsize=4)
def _read_pairs(path: str, mtime_ns: int) -> tuple:
    # Inputs: a verified pair-file path and its mtime (cache key). Output: the rows as a tuple (cached).
    return tuple(read_jsonl_gz(path))


def load_pairs(ctx, hub: str, x: str) -> list:
    # Inputs: Ctx, hub and target codes. Output: the list of pair rows
    # {i, line_no, score, hub, x, labse, lid_hub, lid_x, lid_hub_p, lid_x_p}, after a SHA check.
    path = verified_path(ctx.paths.nllb_dir(hub, x), PAIRS_FILE)
    return list(_read_pairs(str(path), path.stat().st_mtime_ns))


def built_pairs(ctx) -> list:
    # Input: Ctx. Output: [(hub, x)] of every pair with a complete manifest, in table order.
    return [(h, x) for h in HUBS for x in TARGETS if manifest_complete(ctx.paths.nllb_dir(h, x))]


def table_fingerprint(ctx) -> str:
    # Input: Ctx. Output: SHA-256 over the output hashes of every built pair (changes when any pair does).
    return sha256_json({f"{h}__{x}": read_manifest(ctx.paths.nllb_dir(h, x))["outputs"] for h, x in built_pairs(ctx)})


def filter_table(ctx) -> dict:
    # Input: Ctx. Output: {"pairs": {hub__x: survival counts, host, cap, order violations, shortfall,
    # exhausted}, "per_language": {x: {hub: kept}}, "fingerprint"} over the built pairs.
    pairs, per_lang = {}, {}
    for h, x in built_pairs(ctx):
        m = read_manifest(ctx.paths.nllb_dir(h, x))
        s = m["stats"]
        pairs[f"{h}__{x}"] = {"host": m["config"]["source"]["host"], "cap": m["config"]["cap"], **s["survival"],
                              "order_violations": s["order_violations"], "shortfall": s["shortfall"],
                              "exhausted": s["exhausted"], "mb_read": round(m["sources"][0]["bytes_consumed"] / 2**20, 1),
                              "seconds": s["seconds"]}
        per_lang.setdefault(x, {})[h] = s["kept"]
    return {"steps": ["read", *STEPS, "kept"], "pairs": pairs, "per_language": per_lang,
            "fingerprint": table_fingerprint(ctx)}


def format_filter_table(table: dict) -> str:
    # Input: a filter_table dict. Output: the printable survival table (remaining after each filter) and
    # the per-target summary of pairs kept from each hub.
    cols = ["read", *STEPS, "kept"]
    head = f"{'pair':8} {'host':8} " + " ".join(f"{c:>9}" for c in cols) + f" {'short':>7} {'ord.viol':>8} {'MB':>7}"
    lines = ["NLLB filter table (remaining after each filter, in order)", head, "-" * len(head)]
    for name, r in table["pairs"].items():
        lines.append(f"{name:8} {r['host']:8} " + " ".join(f"{r[c]:>9,}" for c in cols)
                     + f" {r['shortfall']:>7,} {r['order_violations']:>8,} {r['mb_read']:>7}")
    lines += ["", "Per-target pairs kept by hub", f"{'target':8} " + " ".join(f"{h:>9}" for h in HUBS)]
    for x, by_hub in table["per_language"].items():
        lines.append(f"{x:8} " + " ".join(f"{by_hub.get(h, '-'):>9}" for h in HUBS))
    return "\n".join(lines)


def write_filter_table(ctx) -> dict:
    # Input: Ctx. Builds the table, prints it and saves ROOT/data/nllb/FILTER_TABLE.json. Output: the table.
    table = filter_table(ctx)
    print(format_filter_table(table), flush=True)
    write_json_atomic(ctx.paths.data("nllb", TABLE_FILE), table)
    return table


def filter_table_done(ctx) -> bool:
    # Input: Ctx. Output: whether FILTER_TABLE.json exists and matches the currently built pairs.
    saved = read_json(ctx.paths.data("nllb", TABLE_FILE))
    return bool(saved) and saved.get("fingerprint") == table_fingerprint(ctx)
