# Measurement stages of the cross-model stitching experiment (interp_stitching, fine-tuned only).
# Each unit is one recipient (model, condition C2 or C3): its English donor (C1) and its control
# donor (the other non-English hub) hand their encoder states after l blocks, for every l in 0..L,
# to the recipient's upper encoder, final norm and decoder on the same FLORES devtest rows.
# interp.stitching.logprob scores per-char log-probs of the X references for self, every donor
# stitch and each donor's own normal forward (context), after an identity check that aborts unless
# self-stitching at every l reproduces the normal forward bitwise.
# interp.stitching.generate decodes the same units exactly like translation evaluation, and
# interp.stitching.metrics scores those outputs with chrF++, spBLEU, COMET and MetricX.
# Every measurement resumes per (unit, target) or per metric, --pilot uses the core.results.locate
# pilot paths, and several metrics units share each COMET/MetricX worker call.

from __future__ import annotations

import contextlib
import dataclasses
import shutil
from dataclasses import dataclass
from typing import Callable

import numpy as np
from transformers import AutoConfig

from ...core.config import CONDITIONS, PINS, SMOKE_TARGETS, TARGETS, DecodingConfig, InterpConfig, resolve_conditions, \
    resolve_langs, resolve_metrics
from ...core.paths import run_id
from ...core.registry import Unit, opt, product_units, register_stage
from ...core.results import ResultFile, locate, run_result, save_run_array
from ...interp.readouts import generate_unit, logprob_unit, plain_provider
from ...interp.scoring import Batching, batching_from
from ...interp.units import UnitFile, fingerprint, measurement_done
from ...metrics.batching import METRIC_OPTIONS, score_keyed
from ...modeling.loading import free_memory, load_trained
from .mechanics import assert_compatible, cached_provider, identity_check, precompute_states, stitch_provider

EXP = "interp_stitching"
SELF = "self"
ROLES = ("en", "ctrl")
PILOT_ROWS = 64
IDENTITY_ROWS = 16
OPTIONS = (
    opt("--stitch-layers", default=None, help="comma list of stitch layers l (default: all 0..L; pilot 0,L/2,L)"),
    opt("--stitch-rows", type=int, default=None, help="seeded subset of FLORES devtest rows (pilot default 64)"),
    opt("--stitch-precompute", action="store_true",
        help="stage donor states in per-layer memmaps so donor and recipient are never resident together"),
    *METRIC_OPTIONS,
)


def load_flores(ctx, split, langs):
    # Inputs: Ctx, FLORES split and language codes. Output: {code: [sentence]} from the corpora agent's
    # loader (imported lazily so this module imports without it).
    from ...corpora.flores import load_flores as load
    return load(ctx, split, langs)


def score_groups(ctx, groups, metric):
    # Inputs: Ctx, {key: rows of {id, src, hyp, ref, target}} and a metric name. Output: {key:
    # metrics.api result}, one worker call for the neural metrics (imported lazily).
    from ...metrics.api import score_row_groups
    return score_row_groups(ctx, groups, metric)


def decoding(ctx) -> DecodingConfig:
    # Input: Ctx. Output: the DecodingConfig used by translation evaluation.
    return ctx.config(DecodingConfig)


def recipients() -> tuple:
    # Output: the recipient conditions of InterpConfig.stitch_donors (C2, C3).
    return tuple(r for r, _ in InterpConfig().stitch_donors)


def stitch_grid(sel: dict, smoke: bool) -> list:
    # Inputs: CLI selections and smoke flag. Output: model x condition Units restricted to recipients.
    return [u for u in product_units(("model", "condition"), sel, smoke) if u.condition in recipients()]


def stitch_models(unit) -> list:
    # Input: a recipient (model, condition) Unit. Output: the trained models it needs: the recipient
    # and both of its donors (C2 needs C1, C2, C3; C3 needs C1, C2, C3), whatever --condition selected.
    donors = dict(InterpConfig().stitch_donors)[unit.condition]
    return [Unit(model=unit.model, condition=c) for c in dict.fromkeys((unit.condition, *donors))]


def donor_roles(recipient: str, smoke: bool = False) -> tuple:
    # Inputs: recipient condition and smoke flag. Output: (("en", english donor), ("ctrl", control
    # donor)); under smoke a donor condition that is never trained is replaced by the recipient.
    donors = dict(InterpConfig().stitch_donors)[recipient]
    present = resolve_conditions("all", smoke)
    return tuple((role, d if d in present else recipient) for role, d in zip(ROLES, donors))


def stitch_name(role: str, layer: int) -> str:
    # Inputs: donor role and stitch layer. Output: the unit name, like "en:l3".
    return f"{role}:l{int(layer)}"


def context_name(role: str) -> str:
    # Input: donor role. Output: the name of that donor's own normal-forward unit, like "context:en".
    return f"context:{role}"


def hub(condition: str) -> str:
    # Input: a condition key. Output: its source (hub) language code.
    return CONDITIONS[condition].hub


def n_layers(ctx, model: str, condition: str) -> int:
    # Inputs: Ctx, model key and condition. Output: the encoder depth L read from the best
    # checkpoint's config (raises OSError when the checkpoint is missing).
    return int(AutoConfig.from_pretrained(str(ctx.paths.checkpoint_for(run_id(model, condition), "best"))).num_layers)


def _row_count(ctx) -> int | None:
    # Input: Ctx. Output: the requested row-subset size (pilot default 64, smoke pilot uses the smoke
    # pilot size) or None for all rows.
    k = ctx.opt("stitch_rows")
    if ctx.pilot and not k:
        k = ctx.config(InterpConfig).pilot_rows if ctx.smoke else PILOT_ROWS
    return int(k) if k else None


def variant_of(ctx) -> str | None:
    # Input: Ctx. Output: the measurement variant ("rows<k>" and/or "l<layers>" for explicit row and
    # layer subsets, joined by "_") or None; core.results.locate adds the pilot suffix.
    k = ctx.opt("stitch_rows")
    parts = [f"rows{int(k)}"] if k else []
    if ctx.opt("stitch_layers"):
        parts.append("l" + "-".join(str(l) for l in _parse_layers(ctx.opt("stitch_layers"))))
    return "_".join(parts) or None


def _parse_layers(spec) -> list:
    # Input: a comma list of layers. Output: the sorted distinct ints.
    return sorted({int(x) for x in str(spec).split(",") if x.strip()})


def _layers(ctx, L: int) -> list:
    # Inputs: Ctx and the encoder depth L. Output: the stitch layers (explicit, pilot {0, L/2, L},
    # or all of 0..L); out-of-range layers raise ValueError.
    spec = ctx.opt("stitch_layers")
    layers = _parse_layers(spec) if spec else (sorted({0, L // 2, L}) if ctx.pilot else list(range(L + 1)))
    if any(l < 0 or l > L for l in layers):
        raise ValueError(f"stitch layers {layers} outside 0..{L}")
    return layers


@dataclass(frozen=True)
class Plan:
    # One recipient's measurement plan: model key, recipient condition, (role, donor condition)
    # pairs, FLORES rows, target codes, stitch layers, encoder depth L and result variant.
    model: str
    recipient: str
    donors: tuple
    rows: tuple
    targets: tuple
    layers: tuple
    n_layers: int
    variant: str | None


def make_plan(ctx, unit) -> Plan:
    # Inputs: Ctx and a (model, condition) Unit. Output: the Plan (pilot: 64 rows, the selected targets
    # among core.config.SMOKE_TARGETS or else the first two, layers {0, L/2, L}); raises OSError when
    # the recipient checkpoint is missing.
    L = n_layers(ctx, unit.model, unit.condition)
    n, k = PINS["flores"]["rows"]["devtest"], _row_count(ctx)
    rows = dataclasses.replace(ctx, pilot=True).rows(n, k) if k else list(range(n))
    targets = resolve_langs(ctx.sel.get("target"), TARGETS, ctx.smoke)
    targets = ([t for t in targets if t in SMOKE_TARGETS] or targets[:2]) if ctx.pilot else targets
    if not targets:
        raise ValueError("no stitching targets selected")
    return Plan(unit.model, unit.condition, donor_roles(unit.condition, ctx.smoke), tuple(rows), tuple(targets),
                tuple(_layers(ctx, L)), L, variant_of(ctx))


def plan_config(ctx, plan: Plan, kind: str) -> dict:
    # Inputs: Ctx, Plan and kind ("logprob" or "generate"). Output: the fingerprinted measurement
    # config (batch sizes and the precompute switch are left out because they do not change results).
    dec = decoding(ctx)
    base = {"model": plan.model, "recipient": plan.recipient, "donors": dict(plan.donors), "rows": list(plan.rows),
            "targets": list(plan.targets), "layers": list(plan.layers), "n_layers": plan.n_layers,
            "testset": "flores_devtest", "units": "self, context:<role>, <role>:l<layer>"}
    if kind == "logprob":
        return {**base, "max_chars": ctx.config(InterpConfig).max_chars, "factor": dec.token_factor}
    return {**base, "decoding": dataclasses.asdict(dec)}


def _rf(ctx, unit, measurement: str, variant):
    # Inputs: Ctx, Unit, measurement name and variant. Output: the bound, pilot-aware ResultFile.
    return run_result(ctx, EXP, unit.run_id, measurement, variant, unit)


def stored_result(ctx, rid: str, measurement: str, variant):
    # Inputs: Ctx, run id, measurement and variant. Output: the stored (pilot-aware) result or None.
    return ResultFile(locate(ctx, EXP, rid, measurement, variant)["result"]).load()


class ModelPool:
    # Keeps the fine-tuned models of one model key resident by condition, loading and freeing them
    # so only the conditions the current job needs stay in memory.

    def __init__(self, ctx, plan: Plan):
        # Inputs: Ctx and the Plan whose model key and depth every loaded model must match.
        self.ctx, self.plan, self.loaded = ctx, plan, {}

    def _load(self, condition: str):
        # Input: a condition. Output: its Trained bundle, checked for depth and compatibility.
        t = load_trained(self.ctx, self.plan.model, condition)
        if t.n_enc != self.plan.n_layers:
            raise AssertionError(f"{condition}: encoder depth {t.n_enc} != planned {self.plan.n_layers}")
        for other in self.loaded.values():
            assert_compatible(t, other)
        return t

    def keep(self, needs) -> dict:
        # Input: the conditions to keep resident. Frees every other model and loads missing ones.
        # Output: {condition: Trained}.
        for c in [c for c in self.loaded if c not in needs]:
            del self.loaded[c]
            free_memory()
        for c in dict.fromkeys(needs):
            self.loaded[c] = self.loaded.get(c) or self._load(c)
        return self.loaded


@dataclass(frozen=True)
class Reader:
    # A readout: read(dec Trained, provider, source texts, targets) -> {target: stored fields}, plus
    # the source character cap, token factor and Batching it encodes with.
    read: Callable
    max_chars: int
    factor: int
    batching: Batching


def _logprob_reader(ctx, plan: Plan, texts: dict) -> Reader:
    # Inputs: Ctx, Plan and FLORES texts. Output: the Reader scoring per-char log-probs of the X
    # references (the provider runs once per batch and is reused across targets).
    cfg, dec = ctx.config(InterpConfig), decoding(ctx)
    batch = batching_from(cfg)

    def read(trained, provider, src, targets):
        # Inputs: decoder model, provider, source texts and targets. Output: {t: {"per_char": [...]}}.
        res = logprob_unit(trained, provider, None, src, {t: texts[t] for t in targets}, plan.rows, batch,
                           cfg.max_chars, dec.token_factor, cfg.logprob_chunk_rows)
        return {t: {"per_char": res[t].per_char.tolist()} for t in targets}
    return Reader(read, cfg.max_chars, dec.token_factor, batch)


def _generate_reader(ctx, plan: Plan, texts: dict) -> Reader:
    # Inputs: Ctx, Plan and FLORES texts (unused). Output: the Reader decoding hypotheses exactly like
    # translation evaluation.
    dec, batch = decoding(ctx), batching_from(ctx.config(InterpConfig), generation=True)

    def read(trained, provider, src, targets):
        # Inputs: decoder model, provider, source texts and targets. Output: {t: {"hyps": [...]}}.
        res = generate_unit(trained, provider, None, src, list(targets), plan.rows, dec, batch)
        return {t: {"hyps": res[t]} for t in targets}
    return Reader(read, dec.max_input_chars, dec.token_factor, batch)


READERS = {"logprob": _logprob_reader, "generate": _generate_reader}


@dataclass(frozen=True)
class Job:
    # One step of a measurement: a unit job (name, resident conditions, decoder condition, source
    # condition and provider factory fn(models)) or an action job (action fn(models) run when
    # when() is true).
    name: str
    needs: tuple
    dec: str = ""
    src: str = ""
    provider: Callable | None = None
    action: Callable | None = None
    when: Callable | None = None


def _pending(uf, plan: Plan, name: str) -> list:
    # Inputs: UnitFile, Plan and a unit name. Output: the targets still missing for that unit.
    return [t for t in plan.targets if not uf.has((name, t))]


def _identity_job(ctx, plan: Plan, texts: dict, reader: Reader, state: dict) -> Job:
    # Inputs: Ctx, Plan, texts, Reader and a state dict. Output: the job that self-stitches the
    # recipient at every l on 16 rows, stores the result in state and aborts on any mismatch.
    R = plan.recipient

    def run(models):
        # Input: resident models. Runs the identity check; raises RuntimeError when it fails.
        res = identity_check(models[R], texts[hub(R)], plan.rows[:IDENTITY_ROWS], reader.max_chars, reader.factor)
        state["identity_check"] = res
        if not res["ok"]:
            raise RuntimeError(f"stitching identity check failed: {res['max_abs_diff']}")
    return Job("", (R,), action=run, when=lambda: True)


def _direct_jobs(ctx, plan: Plan, uf, texts: dict, reader: Reader, identity: list) -> list:
    # Inputs: Ctx, Plan, UnitFile, texts, Reader and the identity-check jobs (possibly none). Output:
    # jobs with donor and recipient resident together: the identity check, self, then per donor its
    # context unit and its stitch at every layer.
    R = plan.recipient
    jobs = identity + [Job(SELF, (R,), R, R, lambda m: plain_provider(m[R]))]
    for role, d in plan.donors:
        jobs.append(Job(context_name(role), (R, d), d, d, lambda m, d=d: plain_provider(m[d])))
        jobs += [Job(stitch_name(role, l), (R, d), R, d, lambda m, d=d, l=l: stitch_provider(m[d], m[R], l))
                 for l in plan.layers]
    return jobs


def _cache_dir(ctx, plan: Plan, role: str):
    # Inputs: Ctx, Plan and donor role. Output: the scratch directory of that donor's memmaps.
    rid = run_id(plan.model, plan.recipient)
    return ctx.paths.tmp(EXP, rid, locate(ctx, EXP, rid, "", plan.variant)["variant"] or "full", role)


def _cached_jobs(ctx, plan: Plan, uf, texts: dict, reader: Reader, identity: list) -> list:
    # Inputs: Ctx, Plan, UnitFile, texts, Reader and the identity-check jobs. Output: jobs that never
    # keep donor and recipient together: per donor its context unit and a memmap precompute (donor
    # only), then the identity check and self (first donor only) and the cached stitches (recipient
    # only), then the cache is deleted.
    R, jobs = plan.recipient, []
    for i, (role, d) in enumerate(plan.donors):
        layers = lambda role=role: [l for l in plan.layers if _pending(uf, plan, stitch_name(role, l))]
        out = _cache_dir(ctx, plan, role)
        cache = lambda m, d=d, out=out, layers=layers: precompute_states(
            m[d], texts[hub(d)], plan.rows, layers(), out, reader.max_chars, reader.factor, reader.batching)
        jobs += [Job(context_name(role), (d,), d, d, lambda m, d=d: plain_provider(m[d])),
                 Job("", (d,), action=cache, when=lambda layers=layers: bool(layers()))]
        jobs += identity + [Job(SELF, (R,), R, R, lambda m: plain_provider(m[R]))] if i == 0 else []
        jobs += [Job(stitch_name(role, l), (R,), R, d, lambda m, out=out, l=l: cached_provider(m[R], out, l))
                 for l in plan.layers]
        jobs.append(Job("", (), action=lambda m, out=out: shutil.rmtree(out, ignore_errors=True),
                        when=lambda out=out: out.exists()))
    return jobs


def run_jobs(jobs, pool: ModelPool, uf, plan: Plan, texts: dict, reader: Reader) -> None:
    # Inputs: jobs, ModelPool, UnitFile, Plan, texts and Reader. Runs each action job when needed and
    # each unit job for its pending targets, storing every (unit, target) as soon as it is read.
    for job in jobs:
        if job.action is not None:
            if job.when():
                job.action(pool.keep(job.needs))
            continue
        pending = _pending(uf, plan, job.name)
        if not pending:
            continue
        models = pool.keep(job.needs)
        res = reader.read(models[job.dec], job.provider(models), texts[hub(job.src)], pending)
        for t in pending:
            uf.put((job.name, t), res[t])


def _aggregate(kind: str, per_item: dict, state: dict) -> dict:
    # Inputs: kind, stored per_item and job state. Output: the aggregate (logprob: identity check
    # and mean per-char log-prob per unit; generate: mean hypothesis length per unit).
    if kind == "logprob":
        means = {k: float(np.nanmean(v["per_char"])) for k, v in per_item.items()}
        return {"identity_check": state.get("identity_check"), "mean_per_char": means}
    return {"mean_hyp_chars": {k: float(np.mean([len(h) for h in v["hyps"]])) for k, v in per_item.items()}}


def measure(ctx, unit, kind: str) -> dict:
    # Inputs: Ctx, a recipient Unit and kind ("logprob" or "generate"). Loads FLORES, builds the
    # plan and runs every pending (unit, target) with resumable storage, then finalizes the
    # aggregate. Output: the written envelope.
    plan = make_plan(ctx, unit)
    texts = load_flores(ctx, "devtest", sorted({hub(plan.recipient), *(hub(d) for _, d in plan.donors), *plan.targets}))
    reader = READERS[kind](ctx, plan, texts)
    cfg = plan_config(ctx, plan, kind)
    envelope = {"config": {**cfg, "batching": dataclasses.asdict(reader.batching),
                           "precompute": bool(ctx.opt("stitch_precompute", False))}}
    state = {}
    with UnitFile.open(_rf(ctx, unit, kind, plan.variant), envelope, fingerprint(cfg), force=ctx.force,
                       flush_seconds=ctx.config(InterpConfig).flush_seconds) as uf:
        identity = [_identity_job(ctx, plan, texts, reader, state)] if kind == "logprob" else []
        build = _cached_jobs if ctx.opt("stitch_precompute") else _direct_jobs
        run_jobs(build(ctx, plan, uf, texts, reader, identity), ModelPool(ctx, plan), uf, plan, texts, reader)
        return uf.finalize(_aggregate(kind, uf.per_item, state))


def _measured(kind: str):
    # Input: kind. Output: is_done(ctx, unit), true when the result is complete under the current
    # plan's fingerprint (false when the checkpoint is missing).
    def done(ctx, unit):
        # Inputs: Ctx and Unit. Output: whether this measurement is finished.
        try:
            plan = make_plan(ctx, unit)
        except OSError:
            return False
        return measurement_done(_rf(ctx, unit, kind, plan.variant), fingerprint(plan_config(ctx, plan, kind)))
    return done


@register_stage("interp.stitching.logprob", grid=stitch_grid, is_done=_measured("logprob"),
                requires=(("train", stitch_models),), order=900, options=OPTIONS)
def logprob(ctx, unit):
    # Inputs: Ctx and a recipient Unit. Writes logprob[__variant].json: per (unit, target) per-char
    # log-probs aligned to the plan rows, with the identity check in the aggregate.
    measure(ctx, unit, "logprob")


@register_stage("interp.stitching.generate", grid=stitch_grid, is_done=_measured("generate"),
                requires=(("train", stitch_models),), order=901, options=OPTIONS)
def generate(ctx, unit):
    # Inputs: Ctx and a recipient Unit. Writes generate[__variant].json: per (unit, target)
    # hypotheses aligned to the plan rows.
    measure(ctx, unit, "generate")


def _metric_fp(gen: dict) -> str:
    # Input: the generate result. Output: the metrics fingerprint (tied to the generation fingerprint).
    return fingerprint({"generate": gen["status"]["fingerprint"]})


def _metrics_wanted(ctx) -> list:
    # Input: Ctx. Output: the selected metric names (chrf_pp, spbleu, comet, metricx).
    return resolve_metrics(ctx.sel.get("metric"))


def _metrics_done(ctx, unit) -> bool:
    # Inputs: Ctx and Unit. Output: whether every selected metric is stored under the fingerprint of
    # the current generation result.
    variant = variant_of(ctx)
    gen = stored_result(ctx, unit.run_id, "generate", variant)
    data = stored_result(ctx, unit.run_id, "metrics", variant)
    if not (gen and data and gen["status"].get("complete")):
        return False
    status = data["status"]
    return status.get("fingerprint") == _metric_fp(gen) and set(_metrics_wanted(ctx)) <= set(status["units_done"])


def metric_rows(gen: dict, texts: dict) -> tuple:
    # Inputs: the generate result and FLORES texts. Output: (unit keys, rows for metrics.api); the
    # source is the recipient's own sentence for self and stitched units (identical across the
    # compared units) and the donor's own sentence for context units.
    cfg, donors = gen["config"], gen["config"]["donors"]
    keys = list(gen["per_item"])
    rows = []
    for key in keys:
        name, t = key.split("|")
        src = texts[hub(donors[name.split(":")[1]] if name.startswith("context:") else cfg["recipient"])]
        rows += [{"id": f"{key}|{r}", "src": src[r], "hyp": h, "ref": texts[t][r], "target": t}
                 for r, h in zip(cfg["rows"], gen["per_item"][key]["hyps"])]
    return keys, rows


def store_metric(ctx, job: dict, metric: str, out: dict) -> None:
    # Inputs: Ctx, one recipient's metrics job, the metric and its metrics.api result. Stores values
    # [U, n] (plus statistics [U, n, k] for corpus metrics) in an .npz and records the metric unit.
    keys, variant = job["keys"], job["variant"]
    arrays = {"values": np.asarray(out["values"], dtype=np.float64).reshape(len(keys), -1)}
    if out.get("stats") is not None:
        stats = np.asarray(out["stats"], dtype=np.int64)
        arrays["stats"] = stats.reshape(len(keys), -1, stats.shape[-1])
    name = f"metrics{'__' + variant if variant else ''}__{metric}.npz"
    job["uf"].put(metric, {"units": keys, "array": save_run_array(ctx, EXP, job["rid"], name, **arrays),
                           "signature": out.get("signature")})


def metrics_job(ctx, unit, stack) -> dict:
    # Inputs: Ctx, a recipient Unit and an ExitStack. Output: the recipient's metrics job {rid, variant,
    # keys, rows, uf} with its resumable result opened on the stack; raises before generate is done.
    variant = variant_of(ctx)
    gen = stored_result(ctx, unit.run_id, "generate", variant)
    if not (gen and gen["status"].get("complete")):
        raise FileNotFoundError(f"generate result for {unit.run_id} ({variant}) is missing or incomplete")
    langs = sorted({hub(gen["config"]["recipient"]), *map(hub, gen["config"]["donors"].values()),
                    *gen["config"]["targets"]})
    keys, rows = metric_rows(gen, load_flores(ctx, "devtest", langs))
    envelope = {"config": {"generate_fingerprint": gen["status"]["fingerprint"],
                           "rows": gen["config"]["rows"], "units": keys}}
    uf = stack.enter_context(UnitFile.open(_rf(ctx, unit, "metrics", variant), envelope, _metric_fp(gen),
                                           force=ctx.force, flush_seconds=0))
    return {"rid": unit.run_id, "variant": variant, "keys": keys, "rows": rows, "uf": uf}


def metrics_units(ctx, units) -> None:
    # Inputs: Ctx and recipient Units. Frees GPU memory, then scores every recipient's stored
    # generations with each selected metric; the recipients still missing a metric share its worker
    # calls, and each recipient's metric is stored as soon as its chunk is scored.
    free_memory()
    with contextlib.ExitStack() as stack:
        jobs = [metrics_job(ctx, u, stack) for u in units]
        for m in _metrics_wanted(ctx):
            groups = {i: job["rows"] for i, job in enumerate(jobs) if not job["uf"].has(m)}
            score_keyed(ctx, groups, m, lambda i, res, m=m: store_metric(ctx, jobs[i], m, res), score_groups)
        for job in jobs:
            job["uf"].finalize({"metrics": list(job["uf"].done)})


@register_stage("interp.stitching.metrics", grid=stitch_grid, is_done=_metrics_done,
                requires=("interp.stitching.generate",), order=902, options=OPTIONS, batch=metrics_units)
def metrics(ctx, unit):
    # Inputs: Ctx and a recipient Unit. Scores its stored generations with each selected metric
    # (resumable per metric) into metrics[__variant].json plus .npz arrays (metrics_units for one).
    metrics_units(ctx, [unit])
