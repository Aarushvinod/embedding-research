# Registers the translation-evaluation stages: translate, score, compare and the eval composite.
# translate runs over model x condition x target x test set, one unit per trained direction, and writes
# that direction's hypotheses; score adds one metric at a time (chrF++, BLEU, spBLEU, COMET, MetricX) to
# the same result file. compare runs over size x condition x test set and writes the ByT5 versus mT5 paired
# bootstrap per direction and pooled over the hub's targets.
# Test sets are FLORES devtest, NTREX and FLORES dev; "--testset main" selects the first two.
# Every stage skips finished units, several score units share neural-metric worker calls, and `eval`
# runs translate, score and compare in order.

from __future__ import annotations

import dataclasses

from ..core.config import ARCHS, model_for
from ..core.registry import Unit, axis_values, composite, opt, product_units, register_stage
from ..metrics.api import canonical, is_scored, score_result_files
from ..metrics.batching import METRIC_OPTIONS
from .compare import compare_done, compare_size_condition, translate_file
from .generate import translate_unit, translation_current
from .testsets import resolve_eval_testsets


def _per_testset(axes, sel: dict, smoke: bool) -> list:
    # Inputs: grid axes, CLI selections and smoke flag. Output: the units of those axes for every
    # selected evaluation test set.
    return [u for t in resolve_eval_testsets(sel.get("testset")) for u in product_units(axes, sel, smoke, testset=t)]


def translate_grid(sel: dict, smoke: bool) -> list:
    # Inputs: CLI selections and smoke flag. Output: model x condition x target x test-set units.
    return _per_testset(("model", "condition", "target"), sel, smoke)


def score_grid(sel: dict, smoke: bool) -> list:
    # Inputs: CLI selections and smoke flag. Output: model x condition x target x metric x test-set units,
    # with metric names in their canonical METRICS form.
    units = _per_testset(("model", "condition", "target", "metric"), sel, smoke)
    return [dataclasses.replace(u, metric=canonical(u.metric)) for u in units]


def compare_grid(sel: dict, smoke: bool) -> list:
    # Inputs: CLI selections and smoke flag. Output: size x condition x test-set units.
    return _per_testset(("size", "condition"), sel, smoke)


def selected_metrics(ctx) -> list:
    # Input: Ctx. Output: the canonical metric names selected by --metric.
    return [canonical(m) for m in axis_values("metric", ctx.sel, ctx.smoke)]


def _translate_done(ctx, unit) -> bool:
    # Inputs: Ctx and unit. Output: whether the unit's translate result is complete and matches the
    # current data, decoding and checkpoint (a stale file is rerun, which refuses it without --force).
    return translation_current(ctx, unit)


def _score_done(ctx, unit) -> bool:
    # Inputs: Ctx and unit. Output: whether the unit's metric is stored for the current hypotheses.
    data = translate_file(ctx, unit.model, unit.condition, unit.target, unit.testset).load()
    return bool(data and data["status"].get("complete")) and is_scored(data, "per_item.rows", unit.metric)


def _compare_done(ctx, unit) -> bool:
    # Inputs: Ctx and unit. Output: whether the comparison matches the current config and scores.
    return compare_done(ctx, unit.size, unit.condition, unit.testset, selected_metrics(ctx))


@register_stage("translate", grid=translate_grid, is_done=_translate_done, requires=("train",), order=600, options=(
        opt("--eval-rows", type=int, default=None, help="evaluate only the first N sentence ids (64 in smoke)"),
        opt("--gen-batch", type=int, default=None, help="maximum generation batch rows")))
def translate(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition, target, testset) unit. Generates and stores the direction's
    # hypotheses. No output.
    translate_unit(ctx, unit)


def _complete_translation(ctx, unit):
    # Inputs: Ctx and a score unit. Output: its complete translate ResultFile; raises otherwise.
    rf = translate_file(ctx, unit.model, unit.condition, unit.target, unit.testset)
    if not rf.is_complete():
        raise RuntimeError(f"{rf.path} is missing or incomplete; run the translate stage first")
    return rf


def score_units(ctx, units) -> None:
    # Inputs: Ctx and score units. Adds each unit's metric to its translate result; units sharing a
    # metric are scored together, so COMET or MetricX loads once per batch of files. No output.
    for metric in dict.fromkeys(u.metric for u in units):
        files = [_complete_translation(ctx, u) for u in units if u.metric == metric]
        score_result_files(ctx, files, [metric], force=ctx.force)


@register_stage("score", grid=score_grid, is_done=_score_done, requires=("translate", "data.sacrebleu"), order=610,
                batch=score_units, options=(
        opt("--metricx-batch-size", type=int, default=8, help="MetricX prediction batch size"),
        opt("--comet-batch-size", type=int, default=64, help="COMET prediction batch size"),
        opt("--metricx-model", default=None, help="MetricX checkpoint (default: the pinned hybrid-xl-v2p6)"),
        *METRIC_OPTIONS))
def score(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition, target, testset, metric) unit. Adds the metric's per-sentence
    # scores, statistics and aggregates to the translate result. No output.
    score_units(ctx, [unit])


def both_archs(unit) -> list:
    # Input: a (size, condition, testset) unit. Output: units pinning the ByT5 and mT5 models of that
    # size (with the condition and test set, every selected target), so both are scored whatever --model
    # selected.
    return [Unit(model=model_for(a, unit.size), condition=unit.condition, testset=unit.testset) for a in ARCHS]


@register_stage("compare", grid=compare_grid, is_done=_compare_done, requires=(("score", both_archs),), order=620)
def compare(ctx, unit) -> None:
    # Inputs: Ctx and a (size, condition, testset) unit. Writes the ByT5 vs mT5 comparison per direction
    # and pooled over the condition's targets. No output.
    compare_size_condition(ctx, unit.size, unit.condition, unit.testset, selected_metrics(ctx))


composite("eval", ("translate", "score", "compare"))
