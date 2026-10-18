# Filter thresholds calibrated on FLORES dev + devtest instead of fixed by hand.
# build_thresholds scores every FLORES sentence with the measures the filters use: the probability of the
# language's GlotLID label (per language), and the LaBSE cosine and the character-length ratio x/hub of each
# aligned sentence pair (per hub -> target direction, candidates included); the script ratio is scored too but
# only recorded for reference, since the filters use the fixed DataConfig.min_script.
# Each threshold is a percentile of those human-translation scores (DataConfig.calib_percentile, 1st by default,
# with no floor), and the length ratio keeps the band between the two DataConfig.ratio_percentiles.
# Per-sentence scores, thresholds and distribution summaries are saved under ROOT/data/thresholds and printed;
# lid_passes applies a language's LID rule and purge_thresholds drops the languages data.purge discards.

from __future__ import annotations

import numpy as np

from ..core.config import CANDIDATES, DATA_LANGS, HUBS, LID_LANGS, TARGETS, DataConfig, language
from ..core.io import read_json, write_json_atomic
from ..core.text import script_ratio
from .assets import asset_revisions, labse_cosines, labse_dtype, load_labse
from .flores import load_flores
from .lid import LID_RULE, label_prob, lid_label, lid_labels, lid_ok, load_glotlid, predict
from .manifest import begin_build, finish_build, manifest_complete, purge, read_manifest, verified_path

TABLE_FILE = "thresholds.json"
SCORES_FILE = "scores.json"
SPLITS = ("dev", "devtest")
MEASURES = ("lid", "labse", "length_ratio", "script_reference")


def thresholds_dir(ctx):
    # Input: Ctx. Output: ROOT/data/thresholds.
    return ctx.paths.data("thresholds")


def directions() -> list:
    # Output: every calibrated (hub, x) direction: each hub to each target and each candidate.
    return [(h, x) for h in HUBS for x in TARGETS + CANDIDATES]


def direction_key(hub: str, x: str) -> str:
    # Inputs: hub and target codes. Output: the direction key "hub__x".
    return f"{hub}__{x}"


def summarize(values) -> dict:
    # Input: a list of scores. Output: {n, min, p1, p5, p10, median, mean, max} of those scores.
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [1, 5, 10, 50])
    return {"n": int(v.size), "min": float(v.min()), "p1": float(q[0]), "p5": float(q[1]), "p10": float(q[2]),
            "median": float(q[3]), "mean": float(v.mean()), "max": float(v.max())}


def calibrate(scores: dict, percentile: float, ratio_percentiles: tuple) -> dict:
    # Inputs: {"lid": {lang: [..]}, "script_reference": {lang: [..]}, "labse": {dir: [..]}, "length_ratio":
    # {dir: [..]}}, the low percentile and the (low, high) length-ratio percentiles. Output: the threshold table
    # with the percentile settings, one minimum per language (lid, script_reference) and direction (labse), one
    # [low, high] band per direction (length_ratio), and per-measure distribution summaries.
    pct = lambda v, p: float(np.percentile(np.asarray(v, dtype=np.float64), p))
    lo, hi = ratio_percentiles
    table = {"percentile": percentile, "ratio_percentiles": [lo, hi], "splits": list(SPLITS)}
    for measure in ("lid", "script_reference", "labse"):
        table[measure] = {k: pct(v, percentile) for k, v in scores[measure].items()}
    table["length_ratio"] = {k: [pct(v, lo), pct(v, hi)] for k, v in scores["length_ratio"].items()}
    table["summary"] = {m: {k: summarize(v) for k, v in scores[m].items()} for m in scores}
    return table


def flores_scores(ctx) -> dict:
    # Input: Ctx. Output: the per-sentence FLORES dev + devtest scores {"lid", "script_reference", "labse",
    # "length_ratio"} for every LID language (lid, script_reference) and calibrated direction (labse, length ratio).
    cfg = ctx.config(DataConfig)
    parts = [load_flores(ctx, split, DATA_LANGS) for split in SPLITS]
    dev = {lang: [t for part in parts for t in part[lang]] for lang in DATA_LANGS}
    model, labels, labse = load_glotlid(ctx), lid_labels(ctx), load_labse(ctx)
    lid = {lang: [label_prob(p, labels[lang]) for p in predict(model, dev[lang], k=cfg.glotlid_top_k)]
           for lang in LID_LANGS}
    script = {lang: [script_ratio(t, language(lang).script) for t in dev[lang]] for lang in LID_LANGS}
    labse_cos = {direction_key(h, x): [float(c) for c in labse_cosines(labse, dev[h], dev[x], cfg.labse_batch)]
                 for h, x in directions()}
    ratio = {direction_key(h, x): [len(b) / max(len(a), 1) for a, b in zip(dev[h], dev[x])] for h, x in directions()}
    return {"lid": lid, "script_reference": script, "labse": labse_cos, "length_ratio": ratio}


def write_table(ctx, scores: dict, table: dict, sources: list, purged: list) -> dict:
    # Inputs: Ctx, the per-sentence scores, the threshold table, the manifest sources and the purged codes.
    # Writes both files with a manifest. Output: the manifest.
    out = begin_build(ctx, thresholds_dir(ctx))
    write_json_atomic(out / SCORES_FILE, scores)
    write_json_atomic(out / TABLE_FILE, table)
    cfg = ctx.config(DataConfig)
    config = {"percentile": cfg.calib_percentile, "ratio_percentiles": list(cfg.ratio_percentiles),
              "splits": list(SPLITS), "script_used": cfg.min_script, "lid_rule": LID_RULE,
              "lid_top_k": cfg.glotlid_top_k, "labse_dtype": labse_dtype(ctx),
              "assets": asset_revisions(ctx), "purged": sorted(purged)}
    return finish_build(ctx, out, sources=sources, config=config, outputs=[SCORES_FILE, TABLE_FILE],
                        stats={"languages": len(table["lid"]), "directions": len(table["labse"])})


def build_thresholds(ctx) -> dict:
    # Input: Ctx. Scores FLORES dev + devtest, calibrates every threshold, prints them with their statistics and
    # saves the table. Output: the manifest.
    cfg = ctx.config(DataConfig)
    scores = flores_scores(ctx)
    table = {**calibrate(scores, cfg.calib_percentile, tuple(cfg.ratio_percentiles)), "script_used": cfg.min_script}
    print(format_thresholds(table), flush=True)
    flores = read_manifest(ctx.paths.flores_dir())
    sources = [{"dataset": "flores", "splits": list(SPLITS),
                "outputs": {k: v["sha256"] for k, v in flores["outputs"].items()}}]
    return write_table(ctx, scores, table, sources, [])


def load_thresholds(ctx, required: bool = True) -> dict:
    # Inputs: Ctx and whether a missing table is an error. Output: the verified threshold table, or {} when it
    # is missing and not required.
    if not required and not manifest_complete(thresholds_dir(ctx)):
        return {}
    return read_json(verified_path(thresholds_dir(ctx), TABLE_FILE))


def lookup(table: dict, measure: str, key: str):
    # Inputs: a threshold table, a measure and a language or direction key. Output: the threshold; raises
    # KeyError naming the fix when the key was never calibrated (or was purged).
    if key not in table.get(measure, {}):
        raise KeyError(f"no calibrated {measure} threshold for {key}; rebuild data.thresholds with --force --rebuild-data")
    return table[measure][key]


def lid_passes(ctx, texts, lang: str) -> list:
    # Inputs: Ctx, raw texts and a language code. Output: per text, whether it passes the NLLB LID filter (the
    # language's label is the top label with at least the language's calibrated probability).
    threshold, label = lookup(load_thresholds(ctx), "lid", lang), lid_label(lang)
    return [lid_ok(p, label, threshold) for p in predict(load_glotlid(ctx), list(texts), k=1)]


def involves(key: str, codes: set) -> bool:
    # Inputs: a language or "hub__x" direction key and a set of codes. Output: whether the key mentions any code.
    return bool(set(key.split("__")) & codes)


def purge_thresholds(ctx, codes) -> bool:
    # Inputs: Ctx and language codes. Rewrites the threshold table and scores without every language and
    # direction that involves those codes (kept pairs record their own thresholds, so they are unaffected).
    # Output: whether anything was removed.
    codes, directory = set(codes), thresholds_dir(ctx)
    if not manifest_complete(directory):
        return False
    table, scores = load_thresholds(ctx), read_json(verified_path(directory, SCORES_FILE))
    if not any(involves(k, codes) for m in ("lid", "labse") for k in table[m]):
        return False
    keep = lambda d: {k: v for k, v in d.items() if not involves(k, codes)}
    table = {**table, **{m: keep(table[m]) for m in MEASURES},
             "summary": {m: keep(v) for m, v in table["summary"].items()}}
    scores = {m: keep(v) for m, v in scores.items()}
    manifest = read_manifest(directory)
    purge(ctx, directory)
    write_table(ctx, scores, table, manifest["sources"], [*manifest["config"].get("purged", []), *codes])
    return True


def stats_line(name: str, threshold: str, summary: dict) -> str:
    # Inputs: a row name, the formatted threshold and the distribution summary. Output: one printable row with
    # the threshold and n, min, p1, p5, p10, median, mean and max.
    cols = " ".join(f"{summary[k]:>8.3f}" for k in ("min", "p1", "p5", "p10", "median", "mean", "max"))
    return f"{name:10} {threshold:>15} {summary['n']:>6} {cols}"


def format_thresholds(table: dict) -> str:
    # Input: a threshold table. Output: the printable thresholds with their FLORES distribution statistics: LID
    # and the script reference per language, LaBSE and the length-ratio band per direction.
    p, (lo, hi), s = table["percentile"], table["ratio_percentiles"], table["summary"]
    head = f"{'':10} {'threshold':>15} {'n':>6} " + " ".join(f"{k:>8}" for k in ("min", "p1", "p5", "p10", "median",
                                                                               "mean", "max"))
    lines = [f"Calibrated thresholds (FLORES {'+'.join(table['splits'])}, percentile {p}; length ratio p{lo}-p{hi})"]
    lines += ["", "GlotLID: own label on top with probability >= threshold", head]
    lines += [stats_line(k, f"{v:.4f}", s["lid"][k]) for k, v in table["lid"].items()]
    lines += ["", f"Script ratio: filters use the fixed {table.get('script_used', 0.8)}; FLORES value for reference", head]
    lines += [stats_line(k, f"{v:.4f}", s["script_reference"][k]) for k, v in table["script_reference"].items()]
    lines += ["", "LaBSE cosine per direction", head]
    lines += [stats_line(k, f"{v:.4f}", s["labse"][k]) for k, v in table["labse"].items()]
    lines += ["", "Length ratio x/hub per direction (band)", head]
    lines += [stats_line(k, f"{v[0]:.3f}-{v[1]:.3f}", s["length_ratio"][k]) for k, v in table["length_ratio"].items()]
    return "\n".join(lines)
