# Hypothesis generation for the translate stage.
# Each unit (model, condition, target, test set) is one trained direction: it loads that direction's
# best checkpoint (ROOT/checkpoints/{model}__{condition}__{target}/best) and translates the hub sources of
# the test set into the target through interp.scoring.generate, the single generation path of the codebase
# (beam 4, length penalty 1.0, 512-character caps, tagless decoder prefix sliced off by position).
# Only the top beam hypothesis is kept. Results go to ROOT/results/translate/{model}__{condition}__{target}/
# {testset}[__pilot].json as columnar per_item rows {id, target, sid, src, ref, hyp}, with a "generate"
# status section and summary aggregate, written once the direction's sentences are translated.
# A config fingerprint (data, decoding, that direction's checkpoint meta) marks stale files as not done
# and refuses them, so a retrained direction is never paired with old hypotheses.
# The model is freed (and the CUDA cache emptied) before the unit returns.

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.config import CONDITIONS, DecodingConfig, InterpConfig, config_hash
from ..core.results import run_result
from ..interp.scoring import Batching, generate
from ..modeling.loading import checkpoint_info as describe_checkpoint
from ..modeling.loading import loaded
from .testsets import rows_fingerprint, testset_rows

EXP = "translate"
ROW_FIELDS = ("id", "target", "sid", "src", "ref", "hyp")


def checkpoint_info(ctx, model: str, condition: str, target: str) -> dict:
    # Inputs: Ctx, model key, condition and target. Output: {"path", "meta_sha256", "step"} of that
    # direction's best checkpoint; raises FileNotFoundError when its meta file is missing.
    info = describe_checkpoint(ctx, model, condition, target=target)
    if info["meta_sha256"] is None:
        raise FileNotFoundError(f"no trained checkpoint meta under {info['path']}")
    return {k: info[k] for k in ("path", "meta_sha256", "step")}


def translate_config(ctx, unit, rows) -> dict:
    # Inputs: Ctx, the unit and its testset rows. Output: the config dict whose hash fingerprints
    # the file: model, condition, hub, target, test set, data hash, decoding and checkpoint.
    return {"model": unit.model, "condition": unit.condition, "hub": CONDITIONS[unit.condition].hub,
            "target": unit.target, "testset": unit.testset, "n_rows": len(rows), "rows_sha256": rows_fingerprint(rows),
            "decoding": dataclasses.asdict(ctx.config(DecodingConfig)),
            "checkpoint": checkpoint_info(ctx, unit.model, unit.condition, unit.target)}


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


def generation_summary(rows) -> dict:
    # Input: finished rows with hyps (one target). Output: {target: {n, empty_hyps, mean_hyp_chars,
    # mean_ref_chars}} as the "generate" aggregate.
    return {rows[0]["target"]: {"n": len(rows), "empty_hyps": sum(not r["hyp"].strip() for r in rows),
                                "mean_hyp_chars": float(np.mean([len(r["hyp"]) for r in rows])),
                                "mean_ref_chars": float(np.mean([len(r["ref"]) for r in rows]))}}


def write_translation(rf, cfg: dict, fp: str, rows, hyps: dict) -> dict:
    # Inputs: the ResultFile, config, fingerprint, testset rows and {row id: hyp} for every row. Writes
    # the rows with their hypotheses as columnar per_item rows, the generate status and summary, marked
    # complete. Output: the envelope.
    kept = [{**r, "hyp": hyps[r["id"]]} for r in rows]
    per_item = {"rows": {f: [r[f] for r in kept] for f in ROW_FIELDS}}
    status = {"fingerprint": fp, "generate": {"complete": True, "target": cfg["target"]}}
    return rf.write(per_item, {"generate": generation_summary(kept)}, config=cfg, complete=True,
                    extra={"status": status, "provenance": {"checkpoint": cfg["checkpoint"]}})


def generate_direction(ctx, unit, rows) -> dict:
    # Inputs: Ctx, a (model, condition, target, testset) unit and its testset rows. Loads that direction's
    # best checkpoint, translates every row into the target and frees the model. Output: {row id: hyp}.
    decoding, icfg = ctx.config(DecodingConfig), ctx.config(InterpConfig)
    batching = Batching(icfg.budget_tokens, int(ctx.opt("gen_batch", icfg.gen_max_batch)))
    with loaded(ctx, unit.model, unit.condition, target=unit.target) as trained:
        hyps = generate(trained, [r["src"] for r in rows], unit.target, decoding, batching)
    return dict(zip([r["id"] for r in rows], hyps))


def direction_rows(ctx, unit) -> list:
    # Inputs: Ctx and a (model, condition, target, testset) unit. Output: the test-set rows of that
    # direction (hub sources, target references).
    return testset_rows(ctx, unit.condition, unit.testset, [unit.target])


def translation_current(ctx, unit) -> bool:
    # Inputs: Ctx and a (model, condition, target, testset) unit. Output: whether its translate result is
    # complete and was produced under the current config (rows, decoding and best-checkpoint meta);
    # False when the file, the data or the checkpoint is missing, so a retrained model is never
    # silently paired with old hypotheses.
    data = run_result(ctx, EXP, unit.run_id, unit.testset).load()
    if not (data and data.get("status", {}).get("complete")):
        return False
    try:
        fp = config_hash(translate_config(ctx, unit, direction_rows(ctx, unit)))
    except FileNotFoundError:
        return False
    return data["status"].get("fingerprint") == fp


def translate_unit(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition, target, testset) unit. Generates the direction's hypotheses
    # (reusing a complete file with the same fingerprint) and writes the result file. No output.
    rf = run_result(ctx, EXP, unit.run_id, unit.testset, unit=unit)
    rows = direction_rows(ctx, unit)
    cfg = translate_config(ctx, unit, rows)
    fp = config_hash(cfg)
    hyps = stored_hyps(rf, fp, ctx.force)
    if not all(r["id"] in hyps for r in rows):
        hyps = generate_direction(ctx, unit, rows)
    write_translation(rf, cfg, fp, rows, hyps)
