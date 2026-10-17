# Steering measurements on the fine-tuned models, each a separately registered resumable stage.
# directions caches per-language sentence-mean statistics on FLORES-dev half A for the fine-tuned and
# the pretrained checkpoint (the latter only for the drift-from-pretraining cosines), select sweeps
# every single layer on dev half B and applies the pre-registered layer rule, and logprob scores the
# X references on devtest for the full grid after a bitwise identity check of the c = 0 hooks.
# generate decodes the full grid with the translation-evaluation settings, metrics scores the
# hypotheses with chrF++, spBLEU, COMET and MetricX, and check measures whether the removed
# component is rebuilt by the blocks downstream of the edit.
# C2 and C3 edit the encoder residual stream, and C1 edits the decoder block outputs from the tag
# position on, with directions fitted at the same site; --pilot uses 64 rows, the config's smoke
# targets, a reduced grid and the pilot paths of core.results.locate, and estimate times one pass.
# Data, checkpoints and metrics are reached through small accessors so tests can substitute fakes, and
# several metrics units share each COMET/MetricX worker call.

from __future__ import annotations

import contextlib
import functools
import time
from dataclasses import asdict, dataclass

import numpy as np

from ...core.config import CONDITIONS, MODELS, PINS, SMOKE_TARGETS, TARGETS, DecodingConfig, InterpConfig, \
    resolve_langs, resolve_metrics
from ...core.io import read_json
from ...core.paths import META_FILE, run_id
from ...core.registry import opt, register_stage
from ...core.results import load_array, locate, run_result, save_run_array
from ...interp.readouts import decoder_edits, generate_unit, logprob_unit
from ...interp.scoring import batching_from, encode_pass, run_batched
from ...interp.units import UnitFile, fingerprint, unit_key
from ...metrics.batching import METRIC_OPTIONS, score_keyed
from ...modeling import hooks
from ...modeling.loading import free_memory, release
from .directions import cosines, decoder_means, decoder_sums, diagnostics, direction_set, encoder_means, \
    encoder_sums, random_units, stat_key
from .settings import BASE_ID, PILOT_GRID, Setting, check_settings, edits_for, eval_settings, full_grid, \
    parse_setting, select_layer, select_settings, setting_id, site_edits, with_scales

EXP = "interp_steering"
GRID = ("model", "condition")
PILOT_ROWS = 64
FT, PRE = "ft", "pre"
CHECK_ROWS = 128
IDENTITY_ROWS = 16
ESTIMATE_ROWS = 32
OPTIONS = (opt("--steer-scales", default=None, help="steering: single-layer scales, comma list"),
           opt("--steer-all-scales", default=None, help="steering: all-layers scales, comma list"),
           opt("--steer-select-scales", default=None, help="steering: select-sweep scales, comma list"),
           *METRIC_OPTIONS)


class IdentityCheckError(RuntimeError):
    # Raised when c = 0 steering hooks do not reproduce the unhooked pass bitwise.
    pass


def load_flores(ctx, split: str, langs=None) -> dict:
    # Inputs: Ctx, FLORES split and language codes. Output: {code: [sentences]} (accessor for tests).
    from ...corpora.flores import load_flores as load
    return load(ctx, split, langs)


def dev_halves(ctx) -> tuple:
    # Input: Ctx. Output: (ids_A, ids_B) FLORES-dev row indices (accessor for tests).
    from ...corpora.flores import dev_halves as halves
    return halves(ctx)


def load_trained(ctx, model: str, condition: str):
    # Inputs: Ctx, model key and condition. Output: the fine-tuned Trained bundle (accessor for tests).
    from ...modeling.loading import load_trained as load
    return load(ctx, model, condition)


def load_pretrained(ctx, model: str, tags: dict):
    # Inputs: Ctx, model key and the fine-tuned checkpoint's {lang: tag id}. Output: the pretrained
    # Hugging Face checkpoint of that key as a float32 interp Trained bundle carrying the same tag
    # ids, so decoder prefixes are identical (accessor for tests).
    from ...modeling.loading import load_pretrained as load
    from ...modeling.loading import make_trained, prepare_for_interp
    model_obj, tok = load(model, device=ctx.device)
    meta = {"source": "pretrained", "hf_id": MODELS[model].hf_id}
    return make_trained(prepare_for_interp(model_obj), tok, tags, meta, MODELS[model].arch, model, None)


def checkpoint_tags(ctx, plan) -> dict:
    # Inputs: Ctx and Plan. Output: the {lang: tag id} stored in the fine-tuned best checkpoint's meta.
    meta = read_json(ctx.paths.checkpoint_for(run_id(plan.model, plan.condition), "best") / META_FILE)
    if not meta or not meta.get("tags"):
        raise FileNotFoundError(f"{plan.run_id}: best checkpoint meta with tags is missing")
    return {k: int(v) for k, v in meta["tags"].items()}


def score_groups(ctx, groups: dict, metric: str) -> dict:
    # Inputs: Ctx, {key: metric rows} and a metric name. Output: {key: metrics.api result}, scored in
    # one worker call for the neural metrics (accessor for tests).
    from ...metrics.api import score_row_groups
    return score_row_groups(ctx, groups, metric)


@dataclass(frozen=True)
class Plan:
    # Everything a steering stage needs to know about one (model, condition) unit: run id, edit site,
    # hub language, targets, result variant ("pilot" under --pilot, else None) and the scale grid.
    model: str
    condition: str
    run_id: str
    site: str
    hub: str
    targets: tuple
    variant: str | None
    grid: object

    @property
    def controls(self) -> tuple:
        # Output: the language-control direction codes of the grid.
        return tuple(self.grid.directions[1:-1])


def pilot_targets(ctx, targets) -> list:
    # Inputs: Ctx and the selected targets. Output: the targets, or under --pilot those among
    # core.config.SMOKE_TARGETS (the first two selected when none is).
    if not ctx.pilot:
        return list(targets)
    return [t for t in targets if t in SMOKE_TARGETS] or list(targets)[:2]


def make_plan(ctx, unit) -> Plan:
    # Inputs: Ctx and a (model, condition) Unit. Output: the Plan; --pilot restricts targets to the
    # config's smoke targets (or the first two selected) and switches to the pilot grid and paths.
    targets = pilot_targets(ctx, resolve_langs(ctx.sel.get("target"), TARGETS, ctx.smoke))
    grid = PILOT_GRID if ctx.pilot else full_grid(ctx.config(InterpConfig))
    grid = with_scales(grid, ctx.opt("steer_scales"), ctx.opt("steer_all_scales"), ctx.opt("steer_select_scales"))
    hub = CONDITIONS[unit.condition].hub
    return Plan(unit.model, unit.condition, unit.run_id, "dec" if hub == "en" else "enc", hub, tuple(targets),
                locate(ctx, EXP, unit.run_id)["variant"], grid)


def split_rows(ctx, split: str) -> list:
    # Inputs: Ctx and split ("A"/"B" dev halves or "devtest"). Output: the row indices used, a fixed
    # seeded subset of PILOT_ROWS rows under --pilot.
    ids = list(range(PINS["flores"]["rows"]["devtest"])) if split == "devtest" else list(dev_halves(ctx)[split == "B"])
    return [ids[p] for p in ctx.rows(len(ids), k=PILOT_ROWS)]


def caps(ctx) -> dict:
    # Input: Ctx. Output: the character cap, token factor, chunk size and scoring batching.
    icfg, dec = ctx.config(InterpConfig), ctx.config(DecodingConfig)
    return {"max_chars": icfg.max_chars, "factor": dec.token_factor, "chunk_rows": icfg.logprob_chunk_rows,
            "batching": batching_from(icfg)}


def result(ctx, plan: Plan, measurement: str):
    # Inputs: Ctx, Plan and measurement name. Output: the (pilot-aware) ResultFile of that measurement.
    return run_result(ctx, EXP, plan.run_id, measurement)


def complete(ctx, plan: Plan, measurement: str):
    # Inputs: Ctx, Plan and measurement. Output: the stored result dict when complete, else None.
    data = result(ctx, plan, measurement).load()
    return data if data and data.get("status", {}).get("complete") else None


def units_done(ctx, plan: Plan, measurement: str, cfg: dict, keys) -> bool:
    # Inputs: Ctx, Plan, measurement, its config and expected unit keys. Output: whether the result is
    # complete under the same fingerprint and holds every expected unit.
    status = (result(ctx, plan, measurement).load() or {}).get("status", {})
    return bool(status.get("complete")) and status.get("fingerprint") == fingerprint(cfg) and \
        {unit_key(k) for k in keys} <= set(status.get("units_done", []))


def run_units(ctx, plan: Plan, measurement: str, cfg: dict, keys, compute, aggregate) -> dict:
    # Inputs: Ctx, Plan, measurement, config, unit keys, compute(key) -> fields and aggregate(uf) ->
    # dict or (dict, final per_item). Computes every missing unit (resumable, flushed on errors) and
    # finalizes. Output: envelope.
    flush = ctx.config(InterpConfig).flush_seconds
    with UnitFile.open(result(ctx, plan, measurement), {"config": cfg}, fingerprint(cfg), force=ctx.force,
                       flush_seconds=flush) as uf:
        for key in keys:
            if not uf.has(key):
                uf.put(key, compute(key))
        out = aggregate(uf)
        return uf.finalize(*(out if isinstance(out, tuple) else (out,)))


class Work:
    # Lazily loaded resources of one stage run: the model (fine-tuned, or pretrained for the drift
    # statistics; never both resident), the FLORES texts of a split, the cached direction statistics
    # and per-target DirectionSets.

    def __init__(self, ctx, plan: Plan, split: str, rows, langs=None):
        # Inputs: Ctx, Plan, FLORES split ("dev" or "devtest"), the row indices and the languages to
        # load (default: the hub and the targets).
        self.ctx, self.plan, self.split, self.rows = ctx, plan, split, list(rows)
        self.langs = list(dict.fromkeys(langs or (plan.hub,) + plan.targets))
        self.caps = caps(ctx)
        self._dsets, self._models = {}, {}

    def model(self, source: str = FT):
        # Input: FT (fine-tuned checkpoint) or PRE (pretrained checkpoint). Output: that Trained
        # bundle, loaded on first use after releasing the other one.
        if source not in self._models:
            for other in list(self._models):
                release(self._models.pop(other))
            self._models[source] = load_trained(self.ctx, self.plan.model, self.plan.condition) if source == FT \
                else load_pretrained(self.ctx, self.plan.model, checkpoint_tags(self.ctx, self.plan))
        return self._models[source]

    @property
    def trained(self):
        # Output: the fine-tuned Trained bundle (loaded on first use).
        return self.model(FT)

    @functools.cached_property
    def texts(self) -> dict:
        # Output: {code: sentences} of the split for the Work's languages.
        return load_flores(self.ctx, self.split, self.langs)

    @functools.cached_property
    def arrays(self) -> dict:
        # Output: the directions.npz arrays of this run and variant.
        return load_directions(self.ctx, self.plan)

    def dset(self, target: str):
        # Input: target code. Output: its DirectionSet (cached).
        if target not in self._dsets:
            self._dsets[target] = target_directions(self.arrays, self.plan, target)
        return self._dsets[target]


def load_directions(ctx, plan: Plan) -> dict:
    # Inputs: Ctx and Plan. Output: {name: array} from the completed directions measurement.
    data = complete(ctx, plan, "directions")
    if data is None:
        raise RuntimeError(f"{plan.run_id}: run interp.steering.directions first")
    with load_array(ctx, data["aggregate"]["npz"]) as z:
        return {k: z[k] for k in z.files}


def stat_group(plan: Plan, source: str, unit: str) -> str:
    # Inputs: Plan, FT or PRE and a statistics unit (language or target). Output: the statistic group
    # of directions.npz ("enc" or the target at the decoder site, prefixed "pre_" for pretrained).
    group = "enc" if plan.site == "enc" else unit
    return group if source == FT else f"{PRE}_{group}"


def target_directions(arrays: dict, plan: Plan, target: str, source: str = FT):
    # Inputs: directions arrays, Plan, target and FT or PRE. Output: the target's DirectionSet; the
    # projection centre is the hub-source mean (encoder site) or the X-reference decoder mean.
    mu_lang = plan.hub if plan.site == "enc" else target
    return direction_set(arrays, stat_group(plan, source, target), target, mu_lang, plan.controls)


def stat_units(plan: Plan) -> list:
    # Input: Plan. Output: the directions units: languages (encoder site) or targets (decoder site)
    # of the fine-tuned model, then the same units "pre:<unit>" of the pretrained model.
    units = list(dict.fromkeys(("en",) + plan.controls + (plan.hub,) + plan.targets)) if plan.site == "enc" \
        else list(plan.targets)
    return units + [f"{PRE}:{u}" for u in units]


def split_unit(key: str) -> tuple:
    # Input: a directions unit key. Output: (FT or PRE, the language or target it covers).
    return tuple(key.split(":", 1)) if key.startswith(f"{PRE}:") else (FT, key)


def directions_spec(ctx, plan: Plan) -> tuple:
    # Inputs: Ctx and Plan. Output: (config, unit keys) of the directions measurement.
    c = caps(ctx)
    cfg = {"site": plan.site, "rows": split_rows(ctx, "A"), "max_chars": c["max_chars"], "factor": c["factor"],
           "controls": list(plan.controls), "seed": ctx.config(InterpConfig).seed}
    return cfg, stat_units(plan)


def compute_stats(ctx, plan: Plan, work: Work, key: str) -> dict:
    # Inputs: Ctx, Plan, Work over dev half A and a directions unit. Computes the unit's per-language
    # sums with the fine-tuned or pretrained model (saved as .npy parts). Output: {"n": rows, "parts":
    # {lang: array ref}}.
    source, unit = split_unit(key)
    tr, c, rows = work.model(source), work.caps, work.rows
    if plan.site == "enc":
        sums = {unit: encoder_sums(tr, work.texts[unit], rows, c["max_chars"], c["factor"], c["batching"])}
    else:
        refs = {lang: work.texts[lang] for lang in dict.fromkeys(("en", unit) + plan.controls)}
        sums = decoder_sums(tr, work.texts["en"], refs, unit, rows, c["max_chars"], c["factor"], c["batching"])
    group = stat_group(plan, source, unit)
    parts = {lang: save_run_array(ctx, EXP, plan.run_id, f"dirpart__{group}__{lang}.npy", s)
             for lang, s in sums.items()}
    return {"n": len(rows), "parts": parts}


def ready_targets(plan: Plan, done: set, source: str) -> list:
    # Inputs: Plan, the finished unit keys and FT or PRE. Output: the targets whose statistics from
    # that model are all finished.
    pre = "" if source == FT else f"{PRE}:"
    need = lambda x: {x} if plan.site == "dec" else {x, "en", plan.hub, *plan.controls}
    return [x for x in plan.targets if {pre + u for u in need(x)} <= done]


def drift(arrays: dict, plan: Plan, targets) -> tuple:
    # Inputs: directions arrays, Plan and the targets with both models' statistics. Output: (per_item
    # {"drift|X": {dir: [L] cos(fine-tuned, pretrained)}}, aggregate with the means over layers per
    # target and over layers and targets per direction) for English and the language controls.
    dirs = ("en",) + plan.controls
    pairs = {x: (target_directions(arrays, plan, x).raw, target_directions(arrays, plan, x, PRE).raw) for x in targets}
    cos = {x: {d: cosines(ft[d], pre[d]) for d in dirs} for x, (ft, pre) in pairs.items()}
    note = "pretrained decoder states use the fine-tuned tag ids, which are untrained sentinels in the pretrained " \
           "model" if plan.site == "dec" else "pretrained encoder states on the same rows and inputs"
    agg = {"per_target": {x: {d: float(np.mean(v)) for d, v in c.items()} for x, c in cos.items()},
           "mean": {d: float(np.mean([cos[x][d] for x in targets])) for d in dirs} if targets else {},
           "pretrained": MODELS[plan.model].hf_id, "note": note}
    return {f"drift|{x}": {d: v.tolist() for d, v in c.items()} for x, c in cos.items()}, agg


def assemble_directions(ctx, plan: Plan, uf) -> tuple:
    # Inputs: Ctx, Plan and the directions UnitFile. Writes directions.npz from every finished unit
    # plus the seeded random directions. Output: (aggregate with the npz ref, sizes, diagnostics and
    # drift summary, the final per_item: the units plus the per-layer drift cosines).
    arrays = {}
    for key in uf.done:
        group = stat_group(plan, *split_unit(key))
        for lang, ref in uf.get(key)["parts"].items():
            arrays[stat_key("sum", group, lang)] = load_array(ctx, ref)
            arrays[stat_key("n", group, lang)] = np.asarray(uf.get(key)["n"])
    n_layers, d = next(v.shape for k, v in arrays.items() if k.startswith("sum__"))
    arrays["rand"] = random_units(n_layers, d, ctx.config(InterpConfig).seed)
    ref = save_run_array(ctx, EXP, plan.run_id, "directions.npz", **arrays)
    done = set(uf.done)
    ready = ready_targets(plan, done, FT)
    diag = {x: diagnostics(target_directions(arrays, plan, x), len(uf.config["rows"])) for x in ready}
    items, drift_agg = drift(arrays, plan, [x for x in ready if x in ready_targets(plan, done, PRE)])
    agg = {"npz": ref, "site": plan.site, "n_layers": n_layers, "d_model": d, "diagnostics": diag, "drift": drift_agg}
    return agg, {**{k: v for k, v in uf.per_item.items() if not k.startswith("drift|")}, **items}


def directions(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Caches the dev-half-A statistics of every needed
    # language for the fine-tuned and then the pretrained model (resumable per unit), and writes
    # directions.npz plus diagnostics and the drift-from-pretraining cosines.
    plan = make_plan(ctx, unit)
    cfg, keys = directions_spec(ctx, plan)
    work = Work(ctx, plan, "dev", cfg["rows"], ("en",) + plan.controls + (plan.hub,) + plan.targets)
    run_units(ctx, plan, "directions", cfg, keys, lambda key: compute_stats(ctx, plan, work, key),
              lambda uf: assemble_directions(ctx, plan, uf))


def select_spec(ctx, plan: Plan):
    # Inputs: Ctx and Plan. Output: (config, unit keys) of the select sweep, or None before directions.
    data = complete(ctx, plan, "directions")
    if data is None:
        return None
    c, g = caps(ctx), plan.grid
    cfg = {"rows": split_rows(ctx, "B"), "directions_fp": fingerprint(directions_spec(ctx, plan)[0]),
           "scales": list(g.select_scales), "select_c": g.select_c, "proj": g.proj, "dirs": list(g.directions),
           "max_chars": c["max_chars"], "factor": c["factor"]}
    sids = [setting_id(s) for s in select_settings(data["aggregate"]["n_layers"], g)]
    return cfg, [(sid, x) for sid in sids for x in plan.targets]


def score_setting(work: Work, sid: str, target: str, rows=None):
    # Inputs: Work, a setting id, the target and optional rows (default work.rows). Output: the
    # CharLogprob of the X references under that setting, teacher-forced from the hub sources.
    tr, c, s = work.trained, work.caps, parse_setting(sid)
    provider, dec = site_edits(tr, work.plan.site, s, work.dset(target) if s.scope != BASE_ID else None)
    res = logprob_unit(tr, provider, dec, work.texts[work.plan.hub], {target: work.texts[target]},
                       work.rows if rows is None else rows, c["batching"], c["max_chars"], c["factor"], c["chunk_rows"])
    return res[target]


def unit_means(per_item: dict, keys) -> dict:
    # Inputs: per_item of a sweep and its (sid, target) keys. Output: {sid: {target: mean per-char
    # log-prob}} from stored "mean" or "per_char" fields.
    out = {}
    for sid, x in keys:
        item = per_item[unit_key((sid, x))]
        out.setdefault(sid, {})[x] = float(item["mean"]) if "mean" in item else float(np.nanmean(item["per_char"]))
    return out


def drop_table(means: dict, targets) -> dict:
    # Inputs: {sid: {target: mean}} and targets. Output: {sid: {target: base - setting, "pooled": mean
    # over targets}} for every non-base setting (positive = damage).
    table = {}
    for sid, row in means.items():
        drops = {x: means[BASE_ID][x] - row[x] for x in targets}
        table[sid] = {**drops, "pooled": float(np.mean(list(drops.values())))}
    return {sid: v for sid, v in table.items() if sid != BASE_ID}


def select_aggregate(per_item: dict, keys, plan: Plan, n_layers: int) -> dict:
    # Inputs: select per_item, its keys, Plan and the layer count. Output: the drop table, the
    # per-layer English-specific damage at select_c and the chosen layer (ties -> shallower).
    g = plan.grid
    drops = drop_table(unit_means(per_item, keys), plan.targets)
    at = lambda l, d: drops[setting_id(Setting(f"L{l}", d, "c", float(g.select_c)))]["pooled"]
    layers = {l: {d: at(l, d) for d in g.directions} for l in range(n_layers)}
    excess = {l: v["en"] - float(np.mean([v[c] for c in plan.controls])) for l, v in layers.items()}
    return {"chosen_layer": select_layer(excess), "rule": "argmax_l pooled drop(en) - mean drop(controls) at select_c; ties shallower",
            "select_c": g.select_c, "per_layer": {str(l): {"drop": layers[l], "excess": excess[l]} for l in layers},
            "drops": drops, "targets": list(plan.targets)}


def select(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Runs the single-layer sweep on dev half B (log-prob
    # readout only, resumable per (setting, target)) and records the chosen layer; a select_c outside
    # the select scales is refused before any unit runs.
    plan = make_plan(ctx, unit)
    g = plan.grid
    if float(g.select_c) not in [float(c) for c in g.select_scales]:
        raise ValueError(f"select_c {g.select_c} is not in the select scales {g.select_scales}")
    spec = select_spec(ctx, plan)
    if spec is None:
        raise RuntimeError(f"{plan.run_id}: run interp.steering.directions first")
    cfg, keys = spec
    work = Work(ctx, plan, "dev", cfg["rows"])
    n_layers = complete(ctx, plan, "directions")["aggregate"]["n_layers"]
    compute = lambda key: {"mean": float(np.nanmean(score_setting(work, *key).per_char))}
    run_units(ctx, plan, "select", cfg, keys, compute, lambda uf: select_aggregate(uf.per_item, keys, plan, n_layers))


def chosen_layer(ctx, plan: Plan):
    # Inputs: Ctx and Plan. Output: the chosen layer from the completed select sweep, or None.
    data = complete(ctx, plan, "select")
    return None if data is None else int(data["aggregate"]["chosen_layer"])


def eval_spec(ctx, plan: Plan, measurement: str):
    # Inputs: Ctx, Plan and "logprob" or "generate". Output: (config, unit keys) over the devtest grid,
    # or None before select.
    chosen = chosen_layer(ctx, plan)
    if chosen is None:
        return None
    c, g = caps(ctx), plan.grid
    cfg = {"rows": split_rows(ctx, "devtest"), "chosen_layer": chosen, "grid": asdict(g),
           "directions_fp": fingerprint(directions_spec(ctx, plan)[0]), "max_chars": c["max_chars"],
           "factor": c["factor"]}
    if measurement == "generate":
        cfg["decoding"] = asdict(ctx.config(DecodingConfig))
    sids = [setting_id(s) for s in eval_settings(chosen, g)]
    return cfg, [(sid, x) for sid in sids for x in plan.targets]


def identity_check(work: Work, chosen: int) -> dict:
    # Inputs: Work over devtest and the chosen layer. Output: whether c = 0 hooks (all layers and the
    # chosen layer, English direction) reproduce the unhooked per-char log-probs bitwise on 16 rows.
    x, rows = work.plan.targets[0], work.rows[:IDENTITY_ROWS]
    base = score_setting(work, BASE_ID, x, rows)
    sids = [setting_id(Setting(scope, "en", "c", 0.0)) for scope in ("all", f"L{chosen}")]
    got = {sid: score_setting(work, sid, x, rows) for sid in sids}
    checks = {sid: bool(np.array_equal(base.total, r.total) and np.array_equal(base.per_char, r.per_char, equal_nan=True))
              for sid, r in got.items()}
    return {"target": x, "rows": rows, "checks": checks, "passed": all(checks.values())}


def logprob(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Runs the identity check (aborting on mismatch), then
    # scores the X references on devtest for every setting and target (resumable).
    plan = make_plan(ctx, unit)
    spec = eval_spec(ctx, plan, "logprob")
    if spec is None:
        raise RuntimeError(f"{plan.run_id}: run interp.steering.select first")
    cfg, keys = spec
    work = Work(ctx, plan, "devtest", cfg["rows"])
    ident = identity_check(work, cfg["chosen_layer"])
    if not ident["passed"]:
        raise IdentityCheckError(f"{plan.run_id}: c = 0 hooks changed the scores: {ident['checks']}")
    compute = lambda key: {"per_char": score_setting(work, *key).per_char.tolist()}
    aggregate = lambda uf: {"identity_check": ident, "chosen_layer": cfg["chosen_layer"], "targets": list(plan.targets),
                            "drops": drop_table(unit_means(uf.per_item, keys), plan.targets)}
    run_units(ctx, plan, "logprob", cfg, keys, compute, aggregate)


def generate_setting(work: Work, sid: str, target: str, rows=None) -> list:
    # Inputs: Work, setting id, target and optional rows. Output: hypotheses decoded with the
    # translation-evaluation settings while the setting's edits are active.
    tr, s = work.trained, parse_setting(sid)
    provider, dec = site_edits(tr, work.plan.site, s, work.dset(target) if s.scope != BASE_ID else None)
    gen = batching_from(work.ctx.config(InterpConfig), generation=True)
    return generate_unit(tr, provider, dec, work.texts[work.plan.hub], target, work.rows if rows is None else rows,
                         work.ctx.config(DecodingConfig), gen)[target]


def generate(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Decodes devtest for every setting and target and
    # stores the hypotheses (resumable per (setting, target)).
    plan = make_plan(ctx, unit)
    spec = eval_spec(ctx, plan, "generate")
    if spec is None:
        raise RuntimeError(f"{plan.run_id}: run interp.steering.select first")
    cfg, keys = spec
    work = Work(ctx, plan, "devtest", cfg["rows"])
    compute = lambda key: {"hyps": generate_setting(work, *key)}
    aggregate = lambda uf: {"targets": list(plan.targets), "units": len(keys),
                            "mean_chars": {unit_key(k): float(np.mean([len(h) for h in uf.get(k)["hyps"]])) for k in keys}}
    run_units(ctx, plan, "generate", cfg, keys, compute, aggregate)


def metrics_spec(ctx, plan: Plan):
    # Inputs: Ctx and Plan. Output: (config, (metric, target) keys) for the selected metrics, or None
    # before generate is complete.
    spec = eval_spec(ctx, plan, "generate")
    if spec is None or not units_done(ctx, plan, "generate", *spec):
        return None
    metrics = resolve_metrics(ctx.sel.get("metric") or "all")
    cfg = {"generate_fp": fingerprint(spec[0]), "rows": spec[0]["rows"]}
    return cfg, [(m, x) for m in metrics for x in plan.targets]


def metric_rows(plan: Plan, gen: dict, texts: dict, rows: list, targets) -> list:
    # Inputs: Plan, the generate result, FLORES devtest texts, rows and targets. Output: metrics.api
    # rows {id, src, hyp, ref, target} for every setting, target and row, in sid-major order per target.
    sids = metric_sids(plan, gen)
    return [{"id": f"{sid}|{x}|{r}", "src": texts[plan.hub][r], "hyp": h, "ref": texts[x][r], "target": x}
            for x in targets for sid in sids for r, h in zip(rows, gen["per_item"][unit_key((sid, x))]["hyps"])]


def metric_sids(plan: Plan, gen: dict) -> list:
    # Inputs: Plan and the generate result. Output: the evaluation setting ids in grid order.
    return [setting_id(s) for s in eval_settings(gen["config"]["chosen_layer"], plan.grid)]


def aligned(values, rows: list):
    # Inputs: per-item values (list aligned to rows, dict keyed by row id, or None) and the rows.
    # Output: a numpy array aligned to rows (None stays None).
    if values is None:
        return None
    return np.asarray([values[r["id"]] for r in rows] if isinstance(values, dict) else values, dtype=np.float64)


def store_metric(ctx, job: dict, metric: str, target: str, res: dict) -> None:
    # Inputs: Ctx, one unit's metrics job, the metric, the target and its metrics.api result. Saves the
    # values (and statistics) [settings, rows] as .npz and records the (metric, target) unit.
    rows, sids = job["gen"]["config"]["rows"], metric_sids(job["plan"], job["gen"])
    items = metric_rows(job["plan"], job["gen"], job["texts"], rows, [target])
    vals, stats = aligned(res.get("values"), items), aligned(res.get("stats"), items)
    arrs = {"values": vals.reshape(len(sids), len(rows))}
    if stats is not None:
        arrs["stats"] = stats.reshape(len(sids), len(rows), -1)
    ref = save_run_array(ctx, EXP, job["plan"].run_id, f"metric__{metric}__{target}.npz", **arrs)
    job["uf"].put((metric, target), {"settings": sids, "array": ref, "mean": arrs["values"].mean(axis=1).tolist()})


def metrics_job(ctx, unit, stack) -> dict:
    # Inputs: Ctx, a (model, condition) Unit and an ExitStack. Output: the unit's metrics job {plan,
    # keys, gen, texts, uf} with its resumable result opened on the stack; raises before generate.
    plan = make_plan(ctx, unit)
    spec = metrics_spec(ctx, plan)
    if spec is None:
        raise RuntimeError(f"{plan.run_id}: run interp.steering.generate first")
    cfg, keys = spec
    uf = stack.enter_context(UnitFile.open(result(ctx, plan, "metrics"), {"config": cfg}, fingerprint(cfg),
                                           force=ctx.force, flush_seconds=0))
    texts = load_flores(ctx, "devtest", list(dict.fromkeys((plan.hub,) + plan.targets)))
    return {"plan": plan, "keys": keys, "gen": result(ctx, plan, "generate").load(), "texts": texts, "uf": uf}


def metric_groups(jobs: list, metric: str) -> dict:
    # Inputs: the metrics jobs and a metric. Output: {(job index, target): metric rows} for every
    # (metric, target) unit still missing.
    out = {}
    for i, job in enumerate(jobs):
        rows = job["gen"]["config"]["rows"]
        out.update({(i, x): metric_rows(job["plan"], job["gen"], job["texts"], rows, [x])
                    for m, x in job["keys"] if m == metric and not job["uf"].has((m, x))})
    return out


def metrics_units(ctx, units) -> None:
    # Inputs: Ctx and (model, condition) Units. Frees GPU memory, then scores the generated hypotheses
    # of every unit with each selected metric; all units' missing targets share the worker calls of
    # that metric, and every (metric, target) is stored as soon as its chunk is scored.
    free_memory()
    with contextlib.ExitStack() as stack:
        jobs = [metrics_job(ctx, u, stack) for u in units]
        for m in dict.fromkeys(k[0] for job in jobs for k in job["keys"]):
            store = lambda key, res, m=m: store_metric(ctx, jobs[key[0]], m, key[1], res)
            score_keyed(ctx, metric_groups(jobs, m), m, store, score_groups)
        for job in jobs:
            job["uf"].finalize({"metrics": list(dict.fromkeys(k[0] for k in job["keys"])),
                                "targets": list(job["plan"].targets)})


def metrics(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Scores that unit's hypotheses (metrics_units for one).
    metrics_units(ctx, [unit])


def check_spec(ctx, plan: Plan):
    # Inputs: Ctx and Plan. Output: (config, (setting, target) keys) of the rebuild check, or None
    # before select.
    chosen = chosen_layer(ctx, plan)
    if chosen is None:
        return None
    devtest = split_rows(ctx, "devtest")
    rows = sorted(int(devtest[i]) for i in np.random.default_rng([ctx.config(InterpConfig).seed, 128]).permutation(
        len(devtest))[:CHECK_ROWS])
    cfg = {"rows": rows, "chosen_layer": chosen, "directions_fp": fingerprint(directions_spec(ctx, plan)[0])}
    sids = [BASE_ID] + [setting_id(s) for s in check_settings(chosen)]
    return cfg, [(sid, x) for sid in sids for x in plan.targets]


def site_means(work: Work, sid: str, target: str) -> np.ndarray:
    # Inputs: Work, setting id and target. Output: per-row masked means of every block output at the
    # edit site with the setting active, float32 [n, L, d] (decoder: reference positions only).
    tr, c, plan, s = work.trained, work.caps, work.plan, parse_setting(sid)
    enc, dec = edits_for(tr, plan.site, s, work.dset(target))

    def step(batch, brows):
        # Inputs: encoder batch and rows. Output: per-row [L, d] block means under the edits.
        with hooks.edit(hooks.resolve_edits(tr.model, "enc", enc)), decoder_edits(tr, dec):
            if plan.site == "enc":
                return list(encoder_means(tr, batch))
            out, mask, _ = encode_pass(tr, batch)
            return list(decoder_means(tr, out, mask, target, [work.texts[target][r] for r in brows],
                                      c["max_chars"], c["factor"]))

    return np.asarray(run_batched(tr, work.texts[plan.hub], work.rows, c["max_chars"], c["factor"], c["batching"], step))


def shift_summary(delta: np.ndarray, first: int, expected) -> dict:
    # Inputs: per-layer mean projection change [L], the first edited layer and the expected shift
    # there (None for the projection). Output: the change, the fraction of the first-layer change
    # retained at every layer (None if that change is zero) and the expected shift.
    retained = (delta / delta[first]).tolist() if delta[first] != 0 else None
    return {"delta": delta.tolist(), "retained": retained, "expected_shift_first": expected}


def check_aggregate(per_item: dict, keys, plan: Plan, work: Work) -> dict:
    # Inputs: check per_item, keys, Plan and Work. Output: per probed setting, target and pooled the
    # per-layer change in the mean projection onto the English unit direction caused by the edit,
    # summarised by shift_summary, plus the unedited per-layer mean projections.
    proj = lambda sid, x: np.asarray(per_item[unit_key((sid, x))]["proj"], dtype=np.float64)
    out = {"base": {x: proj(BASE_ID, x).mean(axis=0).tolist() for x in plan.targets}}
    for sid in dict.fromkeys(k[0] for k in keys if k[0] != BASE_ID):
        s = parse_setting(sid)
        first = s.layer or 0
        deltas = {x: (proj(sid, x) - proj(BASE_ID, x)).mean(axis=0) for x in plan.targets}
        expect = {x: -float(s.c) * float(work.dset(x).norm[first]) for x in plan.targets} if s.op == "c" else {}
        deltas["pooled"] = np.mean([deltas[x] for x in plan.targets], axis=0)
        expect["pooled"] = float(np.mean(list(expect.values()))) if expect else None
        out[sid] = {"first_layer": first, **{x: shift_summary(d, first, expect.get(x)) for x, d in deltas.items()}}
    return out


def check(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. On 128 devtest rows, records the per-row projection
    # onto the English unit direction at every layer with and without each probed edit.
    plan = make_plan(ctx, unit)
    spec = check_spec(ctx, plan)
    if spec is None:
        raise RuntimeError(f"{plan.run_id}: run interp.steering.select first")
    cfg, keys = spec
    work = Work(ctx, plan, "devtest", cfg["rows"])
    compute = lambda key: {"proj": np.einsum("nld,ld->nl", site_means(work, *key), work.dset(key[1]).unit["en"]).tolist()}
    run_units(ctx, plan, "check", cfg, keys, compute, lambda uf: check_aggregate(uf.per_item, keys, plan, work))


def projected(ctx, plan: Plan, work: Work, t_score: float, t_gen: float) -> dict:
    # Inputs: Ctx, Plan, Work, and seconds per row for one scoring and one generation pass. Output:
    # projected seconds per stage for the full grid of this plan.
    tr = work.trained
    n_layers = tr.n_enc if plan.site == "enc" else tr.n_dec
    n_a, n_b, n_dt, n_t = len(split_rows(ctx, "A")), len(split_rows(ctx, "B")), len(split_rows(ctx, "devtest")), len(plan.targets)
    stat_passes = len(stat_units(plan)) if plan.site == "enc" else 2 * n_t * (2 + len(plan.controls))
    n_eval = len(eval_settings(0, plan.grid))
    return {"directions": stat_passes * n_a * t_score,
            "select": len(select_settings(n_layers, plan.grid)) * n_t * n_b * t_score,
            "logprob": n_eval * n_t * n_dt * t_score,
            "generate": n_eval * n_t * n_dt * t_gen,
            "check": 4 * n_t * min(CHECK_ROWS, n_dt) * t_score}


def estimate(ctx, unit) -> None:
    # Inputs: Ctx and a (model, condition) Unit. Times one scoring and one generation pass of the base
    # model on 32 devtest rows and prints projected stage times; writes no results.
    plan = make_plan(ctx, unit)
    work = Work(ctx, plan, "devtest", split_rows(ctx, "devtest")[:ESTIMATE_ROWS])
    x, n = plan.targets[0], len(work.rows)
    work.texts, work.trained
    t0 = time.perf_counter()
    score_setting(work, BASE_ID, x)
    t1 = time.perf_counter()
    generate_setting(work, BASE_ID, x)
    t2 = time.perf_counter()
    per = {"score": (t1 - t0) / n, "generate": (t2 - t1) / n}
    times = projected(ctx, plan, work, per["score"], per["generate"])
    print(f"[estimate] {plan.run_id} ({plan.variant or 'full'}): {per['score']:.4f} s/row scoring, "
          f"{per['generate']:.4f} s/row generation")
    for name, sec in times.items():
        print(f"[estimate]   {name:<10} {sec / 3600:8.2f} h")
    print(f"[estimate]   {'total':<10} {sum(times.values()) / 3600:8.2f} h")


def spec_done(spec_fn, measurement: str):
    # Inputs: a spec function (ctx, plan) -> (cfg, keys) | None and its measurement. Output: an
    # is_done(ctx, unit) predicate requiring a complete, same-fingerprint result with every unit.
    def is_done(ctx, unit):
        # Inputs: Ctx and Unit. Output: whether the measurement is done for this unit and selection.
        plan = make_plan(ctx, unit)
        spec = spec_fn(ctx, plan)
        return spec is not None and units_done(ctx, plan, measurement, *spec)
    return is_done


register_stage("interp.steering.directions", grid=GRID, is_done=spec_done(directions_spec, "directions"),
               requires=("train",), options=OPTIONS, order=600)(directions)
register_stage("interp.steering.select", grid=GRID, is_done=spec_done(select_spec, "select"),
               requires=("interp.steering.directions",), options=OPTIONS, order=601)(select)
register_stage("interp.steering.logprob", grid=GRID, is_done=spec_done(lambda c, p: eval_spec(c, p, "logprob"), "logprob"),
               requires=("interp.steering.select",), options=OPTIONS, order=602)(logprob)
register_stage("interp.steering.generate", grid=GRID,
               is_done=spec_done(lambda c, p: eval_spec(c, p, "generate"), "generate"),
               requires=("interp.steering.select",), options=OPTIONS, order=603)(generate)
register_stage("interp.steering.metrics", grid=GRID, is_done=spec_done(metrics_spec, "metrics"),
               requires=("interp.steering.generate", "data.sacrebleu"), options=OPTIONS, order=604,
               batch=metrics_units)(metrics)
register_stage("interp.steering.check", grid=GRID, is_done=spec_done(check_spec, "check"),
               requires=("interp.steering.select",), options=OPTIONS, order=605)(check)
register_stage("interp.steering.estimate", grid=GRID, is_done=lambda ctx, unit: False,
               options=OPTIONS, order=599)(estimate)
