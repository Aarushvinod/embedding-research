# Hypothesis generation for the translate stage.
# Each unit (model, condition, test set) translates from the condition's hub into every target with
# the model's best checkpoint, through interp.scoring.generate, the single generation path of the
# codebase (beam 4, length penalty 1.0, 512-character caps, sentinels suppressed, prefix sliced off
# by position).
# Results go to ROOT/results/translate/{run_id}/{testset}[__pilot].json as columnar per_item rows
# {id, target, sid, src, ref, hyp}, with a "generate" status section and summary aggregate.
# The file is rewritten after every target, so an interrupted run resumes at the next target, and
# a config fingerprint (data, decoding, checkpoint meta) marks stale files as not done and refuses them.
# The model is freed (and the CUDA cache emptied) before the unit returns.

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.config import CONDITIONS, TARGETS, DecodingConfig, InterpConfig, config_hash, resolve_langs
from ..core.registry import axis_values
from ..core.results import run_result
from ..interp.scoring import Batching, generate
from ..modeling.loading import checkpoint_info as describe_checkpoint
from ..modeling.loading import loaded
from .testsets import rows_fingerprint, testset_rows

EXP = "translate"
ROW_FIELDS = ("id", "target", "sid", "src", "ref", "hyp")


def file_targets(ctx, condition: str) -> list:
    # Inputs: Ctx and condition key. Output: the targets a complete translate file must hold (all
    # of the condition's targets, restricted to the smoke targets in smoke mode).
    allowed = resolve_langs("all", TARGETS, ctx.smoke)
    return [t for t in CONDITIONS[condition].targets if t in allowed]


def run_targets(ctx, condition: str) -> list:
    # Inputs: Ctx and condition key. Output: the targets to generate in this invocation (the file
    # targets narrowed by --target).
    chosen = axis_values("target", ctx.sel, ctx.smoke)
    return [t for t in file_targets(ctx, condition) if t in chosen]


def checkpoint_info(ctx, model: str, condition: str) -> dict:
    # Inputs: Ctx, model key and condition. Output: {"path", "meta_sha256", "step"} of the best
    # checkpoint; raises FileNotFoundError when its meta file is missing.
    info = describe_checkpoint(ctx, model, condition)
    if info["meta_sha256"] is None:
        raise FileNotFoundError(f"no trained checkpoint meta under {info['path']}")
    return {k: info[k] for k in ("path", "meta_sha256", "step")}


def translate_config(ctx, unit, rows) -> dict:
    # Inputs: Ctx, the unit and its testset rows. Output: the config dict whose hash fingerprints
    # the file: model, condition, hub, test set, targets, data hash, decoding and checkpoint.
    return {"model": unit.model, "condition": unit.condition, "hub": CONDITIONS[unit.condition].hub,
            "testset": unit.testset, "targets": sorted({r["target"] for r in rows}, key=TARGETS.index),
            "n_rows": len(rows), "rows_sha256": rows_fingerprint(rows),
            "decoding": dataclasses.asdict(ctx.config(DecodingConfig)),
            "checkpoint": checkpoint_info(ctx, unit.model, unit.condition)}


def stored_hyps(rf, fp: str, force: bool) -> dict:
    # Inputs: the ResultFile, the config fingerprint and the force flag. Output: {row id: hyp} from
    # an existing file with the same fingerprint ({} when absent or forced); a different fingerprint
    # raises RuntimeError unless forced.
    data = rf.load()
    if data is None or force:
        return {}
    if data.get("status", {}).get("fingerprint") != fp:
        raise RuntimeError(f"{rf.path} was produced under a different config; rerun with --force")
    rows = data.get("per_item", {}).get("rows", {})
    return dict(zip(rows.get("id", []), rows.get("hyp", [])))


def targets_done(rows, hyps: dict) -> list:
    # Inputs: testset rows and {row id: hyp}. Output: the targets whose every row has a hypothesis.
    targets = list(dict.fromkeys(r["target"] for r in rows))
    return [t for t in targets if all(r["id"] in hyps for r in rows if r["target"] == t)]


def generation_summary(rows) -> dict:
    # Input: finished rows with hyps. Output: {target: {n, empty_hyps, mean_hyp_chars,
    # mean_ref_chars}} as the "generate" aggregate.
    out = {}
    for t in dict.fromkeys(r["target"] for r in rows):
        sub = [r for r in rows if r["target"] == t]
        out[t] = {"n": len(sub), "empty_hyps": sum(not r["hyp"].strip() for r in sub),
                  "mean_hyp_chars": float(np.mean([len(r["hyp"]) for r in sub])),
                  "mean_ref_chars": float(np.mean([len(r["ref"]) for r in sub]))}
    return out


def write_translation(rf, cfg: dict, fp: str, rows, hyps: dict) -> dict:
    # Inputs: the ResultFile, config, fingerprint, testset rows and {row id: hyp}. Writes the rows of
    # finished targets as columnar per_item rows with the generate status and summary. Output: the
    # envelope.
    done = targets_done(rows, hyps)
    kept = [{**r, "hyp": hyps[r["id"]]} for r in rows if r["target"] in done]
    complete = len(done) == len(cfg["targets"])
    per_item = {"rows": {f: [r[f] for r in kept] for f in ROW_FIELDS}}
    status = {"fingerprint": fp, "units_done": done,
              "generate": {"complete": complete, "targets_done": done, "targets": cfg["targets"]}}
    return rf.write(per_item, {"generate": generation_summary(kept)}, config=cfg, complete=complete,
                    extra={"status": status, "provenance": {"checkpoint": cfg["checkpoint"]}})


def generate_targets(ctx, unit, rows, hyps: dict, todo, on_target) -> None:
    # Inputs: Ctx, unit, testset rows, {row id: hyp} (updated in place), targets to generate and a
    # callback run after each target. Loads the best checkpoint, generates every target in its
    # trained direction and frees the model at the end. No output.
    decoding, icfg = ctx.config(DecodingConfig), ctx.config(InterpConfig)
    batching = Batching(icfg.budget_tokens, int(ctx.opt("gen_batch", icfg.gen_max_batch)))
    with loaded(ctx, unit.model, unit.condition) as trained:
        for t in todo:
            sub = [r for r in rows if r["target"] == t]
            hyps.update(zip([r["id"] for r in sub], generate(trained, [r["src"] for r in sub], t, decoding, batching)))
            on_target(t)


def translation_current(ctx, unit) -> bool:
    # Inputs: Ctx and a (model, condition, testset) unit. Output: whether its translate result is
    # complete and was produced under the current config (rows, decoding and best-checkpoint meta);
    # False when the file, the data or the checkpoint is missing, so a retrained model is never
    # silently paired with old hypotheses.
    data = run_result(ctx, EXP, unit.run_id, unit.testset).load()
    if not (data and data.get("status", {}).get("complete")):
        return False
    try:
        rows = testset_rows(ctx, unit.condition, unit.testset, file_targets(ctx, unit.condition))
        fp = config_hash(translate_config(ctx, unit, rows))
    except FileNotFoundError:
        return False
    return data["status"].get("fingerprint") == fp


def translate_unit(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition, testset) unit. Generates hypotheses for every target not
    # yet in the result file (narrowed by --target) and writes the file after each target. No output.
    rf = run_result(ctx, EXP, unit.run_id, unit.testset, unit=unit)
    rows = testset_rows(ctx, unit.condition, unit.testset, file_targets(ctx, unit.condition))
    cfg = translate_config(ctx, unit, rows)
    fp = config_hash(cfg)
    hyps = stored_hyps(rf, fp, ctx.force)
    done = targets_done(rows, hyps)
    todo = [t for t in run_targets(ctx, unit.condition) if t not in done]
    if todo:
        generate_targets(ctx, unit, rows, hyps, todo, lambda t: write_translation(rf, cfg, fp, rows, hyps))
    write_translation(rf, cfg, fp, rows, hyps)
