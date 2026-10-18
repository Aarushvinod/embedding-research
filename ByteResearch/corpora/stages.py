# Registered data stages: `python -m ByteResearch.run data[.<part>] [--smoke] [--pairs N] [--rebuild-data]`.
# data.flores, data.ntrex and data.exclusion build the evaluation sets and the exclusion set;
# data.lid_model, data.labse_model and data.sacrebleu cache GlotLID, LaBSE and the flores200 SPM model, and
# data.thresholds
# calibrates every LID, script, LaBSE and length-ratio threshold on FLORES dev.
# data.nllb runs over condition x target (the condition's hub to each target) and data.wiki over the
# study languages; data.table prints and saves the per-pair filter table whenever the pairs change.
# data.candidates builds every hub's pairs for the candidate targets (selected through the conditions)
# exactly as for targets, data.candidate_wiki samples their Wikipedia sentences and data.candidate_table
# prints and saves their table; none of it is trained on, but a promoted candidate finds its data ready.
# data.purge --purge CODES permanently deletes the pairs, Wikipedia samples and calibrated thresholds of
# candidates that were not chosen; it refuses study and control languages and does nothing without --purge.
# Every stage is done when its manifest is complete, and a complete manifest is only replaced with
# --force --rebuild-data (the old directory moves to ROOT/data/_superseded/); --force alone keeps it.
# The composite `data` runs all of them in dependency order.

from __future__ import annotations

import functools

from ..core.config import ALL14, CANDIDATE_HUBS, CANDIDATES, CONDITIONS, TrainConfig, resolve_conditions
from ..core.registry import Unit, composite, opt, register_stage
from .assets import asset_dir, build_labse, build_sacrebleu
from .exclusion import build_exclusion, exclusion_dir
from .flores import build_flores, dev_halves, dev_subset_ids
from .lid import build_glotlid
from .manifest import manifest_complete, purge, rebuild_requested
from .nllb import build_pairs, filter_table_done, write_filter_table
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


def _complete(directory_fn):
    # Input: a function (ctx, unit) -> dataset directory. Output: an is_done predicate testing that the
    # directory's manifest is complete.
    def done(ctx, unit):
        # Inputs: Ctx and Unit. Output: whether the unit's dataset manifest is complete.
        return manifest_complete(directory_fn(ctx, unit))
    return done


def data_stage(name: str, directory_fn, **kwargs):
    # Inputs: stage name, a function (ctx, unit) -> dataset directory and register_stage keywords.
    # Output: a decorator registering fn as a stage that is done when the directory's manifest is
    # complete; a forced run leaves a complete dataset untouched (printing a note) unless
    # --rebuild-data is also given, so --force alone never fails on, or rebuilds, an immutable cache.
    def deco(fn):
        # Input: the stage function. Output: its registered, guarded wrapper.
        @functools.wraps(fn)
        def guarded(ctx, unit):
            # Inputs: Ctx and Unit. Runs fn unless the dataset is complete and --rebuild-data is absent.
            directory = directory_fn(ctx, unit)
            if manifest_complete(directory) and not rebuild_requested(ctx):
                print(f"[{name}] kept immutable {directory}; add --rebuild-data to supersede it", flush=True)
                return None
            return fn(ctx, unit)
        return register_stage(name, is_done=_complete(directory_fn), **kwargs)(guarded)
    return deco


@data_stage("data.flores", lambda c, u: c.paths.flores_dir(), grid=(), order=10, options=DATA_OPTS)
def flores_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Builds FLORES dev/devtest and saves the dev subsets (63 rows and the
    # configured size) and the dev halves.
    build_flores(ctx)
    dev_subset_ids(ctx)
    dev_subset_ids(ctx, ctx.config(TrainConfig).dev_per_target)
    dev_halves(ctx)


@data_stage("data.ntrex", lambda c, u: c.paths.ntrex_dir(), grid=(), order=11, options=DATA_OPTS)
def ntrex_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Builds the NTREX files.
    build_ntrex(ctx)


@data_stage("data.exclusion", lambda c, u: exclusion_dir(c), grid=(), requires=("data.flores", "data.ntrex"),
            order=12, options=DATA_OPTS)
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


@data_stage("data.thresholds", lambda c, u: thresholds_dir(c), grid=(), order=15, options=DATA_OPTS,
            requires=("data.flores", "data.lid_model", "data.labse_model"))
def thresholds_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Calibrates and saves the filter thresholds on FLORES dev.
    build_thresholds(ctx)


@data_stage("data.nllb", lambda c, u: c.paths.nllb_dir(CONDITIONS[u.condition].hub, u.target),
            grid=("condition", "target"), order=20, options=DATA_OPTS,
            requires=("data.flores", "data.ntrex", "data.exclusion", "data.lid_model", "data.labse_model", "data.thresholds"))
def nllb_stage(ctx, unit):
    # Inputs: Ctx and a (condition, target) Unit. Streams, filters and saves that hub-target pair file.
    build_pairs(ctx, CONDITIONS[unit.condition].hub, unit.target)


def candidate_units(sel: dict, smoke: bool = False) -> list:
    # Inputs: selections and smoke flag. Output: (condition, lang) Units for every selected condition whose
    # hub is a candidate hub, times every candidate target.
    conds = [c for c in resolve_conditions(sel.get("condition"), smoke) if CONDITIONS[c].hub in CANDIDATE_HUBS]
    return [Unit(condition=c, lang=x) for c in conds for x in CANDIDATES]


def candidate_wiki_units(sel: dict, smoke: bool = False) -> list:
    # Inputs: selections and smoke flag (unused). Output: one (lang) Unit per candidate target.
    return [Unit(lang=x) for x in CANDIDATES]


@data_stage("data.candidates", lambda c, u: c.paths.nllb_dir(CONDITIONS[u.condition].hub, u.lang),
            grid=candidate_units, order=22, options=DATA_OPTS,
            requires=("data.flores", "data.ntrex", "data.exclusion", "data.lid_model", "data.labse_model", "data.thresholds"))
def candidates_stage(ctx, unit):
    # Inputs: Ctx and a (condition, lang) Unit. Streams, filters and saves that hub-candidate pair file.
    build_pairs(ctx, CONDITIONS[unit.condition].hub, unit.lang)


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
    # Inputs: Ctx and a (lang) Unit. Samples that candidate's Wikipedia sentences exactly as for a study language.
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
    protected = [c for c in codes if c in ALL14]
    if protected:
        raise ValueError(f"refusing to purge study/control languages {protected}")
    return codes


def purge_done(ctx, unit) -> bool:
    # Inputs: Ctx and the empty Unit. Output: True unless --purge names languages, so `all` never purges.
    return not purge_codes(ctx)


@register_stage("data.purge", grid=(), is_done=purge_done, order=40, options=PURGE_OPTS)
def purge_stage(ctx, unit):
    # Inputs: Ctx and the empty Unit. Permanently deletes every hub's pair cache and the Wikipedia sample of
    # each --purge language (whether or not it is still listed as a candidate), prints the space freed and
    # removes their languages and directions from the calibrated thresholds.
    codes = purge_codes(ctx)
    for code in codes:
        dirs = [ctx.paths.nllb_dir(h, code) for h in CANDIDATE_HUBS] + [ctx.paths.wiki_dir(code)]
        freed = sum(purge(ctx, d) for d in dirs)
        print(f"[data.purge] {code}: deleted {freed / 2**20:.1f} MB", flush=True)
    if codes and purge_thresholds(ctx, codes):
        print(f"[data.purge] removed {', '.join(codes)} from the calibrated thresholds", flush=True)


composite("data", ("data.flores", "data.ntrex", "data.exclusion", "data.lid_model", "data.labse_model",
                   "data.sacrebleu", "data.thresholds", "data.nllb", "data.wiki", "data.table", "data.candidates",
                   "data.candidate_wiki", "data.candidate_table"))
