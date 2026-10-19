# Registered data stages: `python -m ByteResearch.run data[.<part>] [--smoke] [--pairs N] [--rebuild-data]`.
# data.flores, data.ntrex and data.exclusion build the evaluation sets and the exclusion set; data.lid_model,
# data.labse_model and data.sacrebleu cache GlotLID, LaBSE and the flores200 SPM model, and data.thresholds
# calibrates the LID, LaBSE and length-ratio thresholds (and the script reference) on FLORES.
# data.nllb runs over condition x target and data.wiki over the study languages; data.table prints and saves the
# per-pair filter table, and data.candidates, data.candidate_wiki and data.candidate_table do the same for the
# candidate targets and candidate hubs (never trained on, but a promoted candidate finds its data ready).
# data.purge --purge CODES permanently deletes every pair, Wikipedia sample and threshold of unchosen candidates;
# it refuses study and control languages and does nothing without --purge.
# A stage is done when its manifest is complete, and a complete manifest is only replaced with --force
# --rebuild-data (the old directory moves to ROOT/data/_superseded/), except that the shared caches (FLORES,
# NTREX, exclusion set, thresholds) rebuild themselves when the language set grows or their inputs change.

from __future__ import annotations

import functools

from ..core.config import (ALL_LANGS, CANDIDATE_HUBS, CANDIDATE_SOURCES, CONDITIONS, DATA_LANGS, NTREX_LANGS,
                           NLLB_SOURCES, WIKI_CANDIDATES, TrainConfig, resolve_conditions)
from ..core.registry import Unit, composite, opt, register_stage
from .assets import asset_dir, build_labse, build_sacrebleu
from .exclusion import build_exclusion, exclusion_dir
from .flores import build_flores, dev_halves, dev_subset_ids
from .lid import build_glotlid
from .manifest import inputs_changed, manifest_complete, purge, read_manifest, rebuild_requested, supersede
from .nllb import build_pairs, filter_table_done, pairs_cap, write_filter_table
from .ntrex import build_ntrex
from .thresholds import build_thresholds, purge_thresholds, thresholds_dir
from .wiki import build_wiki

DATA_OPTS = (
    opt("--rebuild-data", action="store_true",
        help="with --force: supersede complete data caches (moved to ROOT/data/_superseded/<ts>/)"),
    opt("--pairs", type=int, default=None, help="pairs kept per NLLB file (default 100k, 2k under --smoke)"),
    opt("--labse-dtype", choices=("fp32", "fp16"), default=None,
        help="LaBSE inference dtype for the NLLB filter (default fp32; fp16 is about 3x faster)"),
)
WIKI_OPTS = DATA_OPTS + (opt("--lang", default="all", help="study languages for data.wiki (all | codes)"),)
PURGE_OPTS = (opt("--purge", default=None, help="data.purge: candidate codes whose data to delete, comma list"),)


def data_stage(name: str, directory_fn, stale_fn=None, **kwargs):
    # Inputs: stage name, a function (ctx, unit) -> dataset directory, an optional stale_fn(ctx, unit) -> whether
    # a complete dataset no longer matches the configured languages, inputs or cap, and register_stage keywords.
    # Output: a decorator registering fn as a stage that is done when the directory's manifest is complete and
    # not stale; a stale dataset is superseded and rebuilt automatically, and a forced run otherwise leaves a
    # complete dataset untouched (printing a note) unless --rebuild-data is also given.
    stale = stale_fn or (lambda ctx, unit: False)

    def deco(fn):
        # Input: the stage function. Output: its registered, guarded wrapper.
        @functools.wraps(fn)
        def guarded(ctx, unit):
            # Inputs: Ctx and Unit. Runs fn unless the dataset is complete, current and --rebuild-data is absent.
            directory = directory_fn(ctx, unit)
            if manifest_complete(directory) and stale(ctx, unit) and not rebuild_requested(ctx):
                print(f"[{name}] languages, inputs or cap changed; superseded {directory} -> {supersede(ctx, directory)}",
                      flush=True)
            if manifest_complete(directory) and not rebuild_requested(ctx):
                print(f"[{name}] kept immutable {directory}; add --rebuild-data to supersede it", flush=True)
                return None
            return fn(ctx, unit)

        def done(ctx, unit):
            # Inputs: Ctx and Unit. Output: whether the dataset is complete and not stale.
            return manifest_complete(directory_fn(ctx, unit)) and not stale(ctx, unit)
        return register_stage(name, is_done=done, **kwargs)(guarded)
    return deco


def missing_langs(directory, key: str, langs) -> bool:
    # Inputs: a dataset directory, the manifest config key listing its languages and the configured languages.
    # Output: whether the complete dataset lacks any configured language (False when it is not built yet).
    m = read_manifest(directory)
    return bool(m and m.get("complete")) and not set(langs) <= set(m.get("config", {}).get(key, []))


def flores_stale(ctx, unit=None) -> bool:
    # Inputs: Ctx and the (unused) Unit. Output: whether the built FLORES lacks a configured data language.
    return missing_langs(ctx.paths.flores_dir(), "columns", DATA_LANGS)


def ntrex_stale(ctx, unit=None) -> bool:
    # Inputs: Ctx and the (unused) Unit. Output: whether the built NTREX lacks a configured NTREX language.
    return missing_langs(ctx.paths.ntrex_dir(), "langs", NTREX_LANGS)


def exclusion_stale(ctx, unit=None) -> bool:
    # Inputs: Ctx and the (unused) Unit. Output: whether the exclusion set was built from other FLORES/NTREX
    # outputs than the current ones.
    return manifest_complete(exclusion_dir(ctx)) and inputs_changed(
        exclusion_dir(ctx), {"flores": ctx.paths.flores_dir(), "ntrex": ctx.paths.ntrex_dir()})


def thresholds_stale(ctx, unit=None) -> bool:
    # Inputs: Ctx and the (unused) Unit. Output: whether the thresholds were calibrated on other FLORES outputs
    # than the current ones.
    return manifest_complete(thresholds_dir(ctx)) and inputs_changed(thresholds_dir(ctx), {"flores": ctx.paths.flores_dir()})


@data_stage("data.flores", lambda c, u: c.paths.flores_dir(), flores_stale, grid=(), order=10, options=DATA_OPTS)
def flores_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Builds FLORES dev/devtest and saves the dev subsets (200 rows and the
    # configured size) and the dev halves.
    build_flores(ctx)
    dev_subset_ids(ctx)
    dev_subset_ids(ctx, ctx.config(TrainConfig).dev_per_target)
    dev_halves(ctx)


@data_stage("data.ntrex", lambda c, u: c.paths.ntrex_dir(), ntrex_stale, grid=(), order=11, options=DATA_OPTS)
def ntrex_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Builds the NTREX files.
    build_ntrex(ctx)


@data_stage("data.exclusion", lambda c, u: exclusion_dir(c), exclusion_stale, grid=(),
            requires=("data.flores", "data.ntrex"), order=12, options=DATA_OPTS)
def exclusion_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Builds the normalized FLORES/NTREX exclusion set.
    build_exclusion(ctx)


@data_stage("data.lid_model", lambda c, u: asset_dir(c, "glotlid"), grid=(), order=13, options=DATA_OPTS)
def lid_model_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Downloads GlotLID.
    build_glotlid(ctx)


@data_stage("data.labse_model", lambda c, u: asset_dir(c, "labse"), grid=(), order=14, options=DATA_OPTS)
def labse_model_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Downloads LaBSE.
    build_labse(ctx)


@data_stage("data.sacrebleu", lambda c, u: c.paths.sacrebleu_dir(), grid=(), order=15, options=DATA_OPTS)
def sacrebleu_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Caches the sacreBLEU flores200 SPM model.
    build_sacrebleu(ctx)


@data_stage("data.thresholds", lambda c, u: thresholds_dir(c), thresholds_stale, grid=(), order=15, options=DATA_OPTS,
            requires=("data.flores", "data.lid_model", "data.labse_model"))
def thresholds_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Calibrates and saves the filter thresholds on FLORES dev.
    build_thresholds(ctx)


def cap_stale(hub_fn):
    # Input: a function unit -> (hub, x). Output: stale_fn(ctx, unit) that is true when the unit's complete pair
    # file was built with another cap than the current one (--pairs or DataConfig.nllb_cap).
    def stale(ctx, unit):
        # Inputs: Ctx and a pair Unit. Output: whether its complete pair file has a different cap.
        m = read_manifest(ctx.paths.nllb_dir(*hub_fn(unit)))
        return bool(m and m.get("complete")) and m.get("config", {}).get("cap") != pairs_cap(ctx)
    return stale


@data_stage("data.nllb", lambda c, u: c.paths.nllb_dir(CONDITIONS[u.condition].hub, u.target),
            cap_stale(lambda u: (CONDITIONS[u.condition].hub, u.target)),
            grid=("condition", "target"), order=20, options=DATA_OPTS,
            requires=("data.flores", "data.ntrex", "data.exclusion", "data.lid_model", "data.labse_model", "data.thresholds"))
def nllb_stage(ctx, unit):
    # Inputs: Ctx and a (condition, target) Unit. Streams, filters and saves that hub-target pair file.
    build_pairs(ctx, CONDITIONS[unit.condition].hub, unit.target)


def candidate_units(sel: dict, smoke: bool = False) -> list:
    # Inputs: selections and smoke flag. Output: (hub, lang) Units for every candidate pair whose hub is the hub
    # of a selected condition or a candidate hub: each hub to each candidate target, and each candidate hub to
    # every target and candidate it has NLLB pairs with.
    hubs = {CONDITIONS[c].hub for c in resolve_conditions(sel.get("condition"), smoke)} | set(CANDIDATE_HUBS)
    return [Unit(hub=h, lang=x) for h, x in CANDIDATE_SOURCES if h in hubs]


def candidate_wiki_units(sel: dict, smoke: bool = False) -> list:
    # Inputs: selections and smoke flag (unused). Output: one (lang) Unit per candidate target and candidate hub.
    return [Unit(lang=x) for x in WIKI_CANDIDATES]


@data_stage("data.candidates", lambda c, u: c.paths.nllb_dir(u.hub, u.lang), cap_stale(lambda u: (u.hub, u.lang)),
            grid=candidate_units, order=22, options=DATA_OPTS,
            requires=("data.flores", "data.ntrex", "data.exclusion", "data.lid_model", "data.labse_model", "data.thresholds"))
def candidates_stage(ctx, unit):
    # Inputs: Ctx and a (hub, lang) Unit. Streams, filters and saves that candidate pair file.
    build_pairs(ctx, unit.hub, unit.lang)


@data_stage("data.wiki", lambda c, u: c.paths.wiki_dir(u.lang), grid=("lang",), order=21, options=WIKI_OPTS,
            requires=("data.exclusion", "data.lid_model", "data.thresholds"))
def wiki_stage(ctx, unit):
    # Inputs: Ctx and a (lang) Unit. Samples that language's Wikipedia sentences.
    build_wiki(ctx, unit.lang)


@register_stage("data.table", grid=(), is_done=lambda ctx, unit: filter_table_done(ctx), requires=("data.nllb",),
                order=30, options=DATA_OPTS)
def table_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Prints and saves the NLLB filter table over all built pairs.
    write_filter_table(ctx)


@data_stage("data.candidate_wiki", lambda c, u: c.paths.wiki_dir(u.lang), grid=candidate_wiki_units, order=23,
            options=DATA_OPTS, requires=("data.exclusion", "data.lid_model", "data.thresholds"))
def candidate_wiki_stage(ctx, unit):
    # Inputs: Ctx and a (lang) Unit. Samples that candidate's (or candidate hub's) Wikipedia sentences exactly as
    # for a study language.
    build_wiki(ctx, unit.lang)


@register_stage("data.candidate_table", grid=(), is_done=lambda ctx, unit: filter_table_done(ctx, candidates=True),
                requires=("data.candidates",), order=31, options=DATA_OPTS)
def candidate_table_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Prints and saves the candidate yield table over all built candidate pairs.
    write_filter_table(ctx, candidates=True)


def purge_codes(ctx) -> list:
    # Input: Ctx. Output: the language codes given with --purge; raises ValueError for a study or control
    # language, whose data training, evaluation or interp still need.
    codes = [c.strip() for c in str(ctx.opt("purge") or "").split(",") if c.strip()]
    protected = [c for c in codes if c in ALL_LANGS]
    if protected:
        raise ValueError(f"refusing to purge study/control languages {protected}")
    return codes


def purge_done(ctx, unit) -> bool:
    # Inputs: Ctx and the empty Unit. Output: True unless --purge names languages, so `all` never purges.
    return not purge_codes(ctx)


@register_stage("data.purge", grid=(), is_done=purge_done, order=40, options=PURGE_OPTS)
def purge_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Permanently deletes every pair cache involving each --purge language (as
    # target or as candidate hub) and its Wikipedia sample, whether or not it is still listed as a candidate,
    # prints the space freed and removes its languages and directions from the calibrated thresholds.
    codes = purge_codes(ctx)
    pairs = list(NLLB_SOURCES) + list(CANDIDATE_SOURCES)
    for code in codes:
        dirs = [ctx.paths.nllb_dir(h, x) for h, x in pairs if code in (h, x)] + [ctx.paths.wiki_dir(code)]
        freed = sum(purge(ctx, d) for d in dirs)
        print(f"[data.purge] {code}: deleted {freed / 2**20:.1f} MB", flush=True)
    if codes and purge_thresholds(ctx, codes):
        print(f"[data.purge] removed {', '.join(codes)} from the calibrated thresholds", flush=True)


composite("data", ("data.flores", "data.ntrex", "data.exclusion", "data.lid_model", "data.labse_model",
                   "data.sacrebleu", "data.thresholds", "data.nllb", "data.wiki", "data.table", "data.candidates",
                   "data.candidate_wiki", "data.candidate_table"))
