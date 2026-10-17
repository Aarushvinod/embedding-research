# Stage registry and runner behind `python -m ByteResearch.run <stage>`.
# A stage is a function fn(ctx, unit) registered with a grid (the axes it is run over, or a
# callable producing units), an is_done(ctx, unit) predicate, the stages it requires, an optional
# batch function that runs several pending units in one call, and any extra CLI options.
# A requirement is a stage or composite name, whose units follow each dependent unit's own axes, or
# a (name, fn) pair whose fn(unit) lists units pinning the axes of the units that stage must run.
# Composites name groups of stages (`all`, declared composites and every dotted prefix), and
# resolve_requires orders everything topologically, breaking ties by each stage's order and name.
# --force reruns only the explicitly requested stages, each unit's console output is also appended to
# ROOT/logs/{stage}/{run id}.log, and discover() imports the stage-bearing packages and restores
# declarations that a test-isolated registry dropped.

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import itertools
import pkgutil
import sys
import time
import traceback
from dataclasses import dataclass, field, fields
from typing import Callable

from . import config as C
from .dist import is_main, relaunch_with_torchrun, under_torchrun
from .io import tee_output
from .paths import compare_id, run_id

AXES = ("model", "condition", "target", "testset", "metric", "lang", "lr", "size", "variant")
DISCOVER_PACKAGES = ("corpora", "training", "evaluation", "metrics", "experiments")
BUILTIN = ("list", "status")


@dataclass(frozen=True)
class Unit:
    # One cell of a stage grid. Every axis is optional; stages read only the axes they declare.
    # run_id is model__condition[__lr] when a model is set, size__condition for comparisons.
    model: str | None = None
    condition: str | None = None
    target: str | None = None
    testset: str | None = None
    metric: str | None = None
    lang: str | None = None
    lr: float | None = None
    size: str | None = None
    variant: str | None = None

    @property
    def run_id(self):
        # Output: the run id for this unit, or None when neither model nor size is set.
        if self.model and self.condition:
            return run_id(self.model, self.condition, self.lr)
        if self.size and self.condition:
            return compare_id(self.size, self.condition)
        return None

    def as_dict(self) -> dict:
        # Output: {axis: value} for the axes that are set.
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    @property
    def label(self) -> str:
        # Output: a compact human-readable "axis=value ..." description ("-" for an empty unit).
        return " ".join(f"{k}={v}" for k, v in self.as_dict().items()) or "-"


@dataclass(frozen=True)
class Option:
    # An extra argparse option declared by a stage: positional flag strings and argparse kwargs.
    flags: tuple
    kwargs: dict = field(default_factory=dict, hash=False, compare=False)


def opt(*flags, **kwargs) -> Option:
    # Inputs: argparse flag strings and keyword arguments. Output: an Option for register_stage.
    return Option(tuple(flags), dict(kwargs))


@dataclass
class Stage:
    # A registered stage: name, fn(ctx, unit), grid (axis tuple or callable(sel, smoke) -> units),
    # is_done(ctx, unit), requirements (names or (name, fn(unit) -> units) pairs), whether it runs
    # under DDP, extra options, sort order, the experiment/measurement names used for result paths
    # and an optional batch(ctx, units) used when several units are pending at once.
    name: str
    fn: Callable
    grid: object
    is_done: Callable
    requires: tuple = ()
    ddp: bool = False
    options: tuple = ()
    order: int = 100
    experiment: str = ""
    measurement: str = ""
    batch: Callable | None = None


_STAGES: dict = {}
_COMPOSITES: dict = {}
_DECLARED: dict = {}


def default_experiment(name: str) -> tuple:
    # Input: a dotted stage name. Output: (experiment, measurement); "interp.x.m" -> ("interp_x", "m"),
    # "a.b" -> ("a", "b"), "a" -> ("a", "a").
    parts = name.split(".")
    if parts[0] == "interp" and len(parts) >= 3:
        return f"interp_{parts[1]}", ".".join(parts[2:])
    return parts[0], (".".join(parts[1:]) or parts[0])


def _declare(module: str, kind: str, name: str, value) -> None:
    # Inputs: the declaring module, "stages" or "composites", the name and the Stage or members.
    # Records the declaration so discover() can restore it into a registry that lost it.
    _DECLARED.setdefault(module, {"stages": {}, "composites": {}})[kind][name] = value


def register_stage(name, *, grid, is_done, requires=(), ddp=False, options=(), order=100, experiment=None,
                   batch=None):
    # Inputs: stage name, grid, is_done predicate, requirements, DDP flag, Option tuple, sort order,
    # an optional experiment override and an optional batch(ctx, units) function. Output: a decorator
    # that registers fn(ctx, unit) and returns it unchanged. Re-importing the same function replaces
    # silently; a different function under an existing name raises ValueError.
    def deco(fn):
        # Input: the stage function. Output: the same function, after registering it.
        old = _STAGES.get(name)
        if old and (old.fn.__module__, old.fn.__qualname__) != (fn.__module__, fn.__qualname__):
            raise ValueError(f"stage {name!r} already registered by {old.fn.__module__}")
        exp, meas = default_experiment(name)
        stage = Stage(name, fn, grid, is_done, tuple(requires), ddp, tuple(options), order, experiment or exp, meas,
                      batch)
        _STAGES[name] = stage
        _declare(fn.__module__, "stages", name, stage)
        return fn
    return deco


def composite(name: str, members) -> None:
    # Inputs: composite name and member stage/composite names. Registers the composite (an explicit
    # declaration overrides the implicit prefix composite of the same name). No output.
    if name in _STAGES or name in BUILTIN:
        raise ValueError(f"{name!r} is already a stage or built-in")
    _COMPOSITES[name] = tuple(members)
    _declare(sys._getframe(1).f_globals.get("__name__", ""), "composites", name, tuple(members))


def get_stage(name: str) -> Stage:
    # Input: stage name. Output: the Stage; KeyError if unknown.
    if name not in _STAGES:
        raise KeyError(f"unknown stage {name!r}")
    return _STAGES[name]


def list_stages() -> list:
    # Output: every registered Stage sorted by (order, name).
    return sorted(_STAGES.values(), key=lambda s: (s.order, s.name))


def _prefix_members(name: str) -> list:
    # Input: a dotted prefix. Output: names of registered stages strictly below it.
    return [s.name for s in list_stages() if s.name.startswith(name + ".")]


def list_composites() -> dict:
    # Output: {composite name: member names} for "all", explicit composites and implicit prefixes.
    out = {"all": tuple(s.name for s in list_stages())}
    out.update(_COMPOSITES)
    for s in list_stages():
        parts = s.name.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix not in _STAGES:
                out.setdefault(prefix, tuple(_prefix_members(prefix)))
    return out


def is_known(name: str) -> bool:
    # Input: a name. Output: whether it is a stage, a built-in or a composite.
    return name in _STAGES or name in BUILTIN or name in list_composites()


def expand_names(names, _seen=None) -> list:
    # Input: stage and/or composite names. Output: the stage names they denote, in first-seen
    # order; unknown names raise KeyError and composite cycles raise ValueError.
    seen = set(_seen or ())
    comps = list_composites()
    out = []
    for name in names:
        if name in _STAGES:
            out.append(name)
            continue
        if name not in comps:
            raise KeyError(f"unknown stage or composite {name!r}")
        if name in seen:
            raise ValueError(f"composite cycle through {name!r}")
        out.extend(expand_names(comps[name], seen | {name}))
    return list(dict.fromkeys(out))


def requirement(item) -> tuple:
    # Input: one requires entry (a name, or a (name, fn) pair). Output: (name, fn or None).
    return (item, None) if isinstance(item, str) else (item[0], item[1])


def requirement_names(stage: Stage) -> list:
    # Input: a Stage. Output: the stage names its requirements expand to.
    return expand_names([requirement(r)[0] for r in stage.requires])


def _closure(names: list) -> set:
    # Input: stage names. Output: those names plus all transitive requirements.
    todo, done = list(names), set()
    while todo:
        name = todo.pop()
        if name not in done:
            done.add(name)
            todo.extend(requirement_names(get_stage(name)))
    return done


def resolve_requires(names, no_deps: bool = False) -> list:
    # Inputs: requested stage/composite names and the --no-deps flag. Output: Stage objects in
    # dependency order (requirements first, ties by order then name). With no_deps only the
    # requested stages run, still ordered among themselves. Cycles raise ValueError.
    wanted = expand_names(names)
    chosen = set(wanted) if no_deps else _closure(wanted)
    deps = {n: set(requirement_names(get_stage(n))) & chosen for n in chosen}
    ordered = []
    while deps:
        ready = sorted((n for n, d in deps.items() if not d), key=lambda n: (get_stage(n).order, n))
        if not ready:
            raise ValueError(f"dependency cycle among {sorted(deps)}")
        ordered.append(ready[0])
        deps.pop(ready[0])
        for d in deps.values():
            d.discard(ready[0])
    return [get_stage(n) for n in ordered]


def axis_values(axis: str, sel: dict, smoke: bool = False) -> list:
    # Inputs: an axis name, the CLI selections dict and the smoke flag. Output: the values that
    # axis takes for this invocation (e.g. resolved model keys, target codes, float lrs).
    get = sel.get
    resolvers = {
        "model": lambda: C.resolve_models(get("model"), smoke),
        "condition": lambda: C.resolve_conditions(get("condition"), smoke),
        "target": lambda: C.resolve_langs(get("target"), C.TARGETS, smoke),
        "lang": lambda: C.resolve_langs(get("lang"), C.STUDY11, smoke),
        "testset": lambda: C.resolve_testsets(get("testset")),
        "metric": lambda: C.resolve_metrics(get("metric")),
        "lr": lambda: C.resolve_lrs(get("lr")),
        "size": lambda: C.resolve_sizes(C.resolve_models(get("model"), smoke)),
        "variant": lambda: [v for v in str(get("variant") or "").split(",") if v],
    }
    if axis not in resolvers:
        raise KeyError(f"unknown axis {axis!r}")
    return resolvers[axis]()


def product_units(axes, sel: dict, smoke: bool = False, **fixed) -> list:
    # Inputs: axis names, selections, smoke flag and fixed axis values. Output: the cartesian
    # product of the resolved axis values as Units (fixed values override/extend each unit).
    axes = tuple(axes)
    values = [axis_values(a, sel, smoke) for a in axes]
    return [Unit(**dict(zip(axes, combo)), **fixed) for combo in itertools.product(*values)]


def expand_units(stage: Stage, selections: dict, smoke: bool = False) -> list:
    # Inputs: a Stage, the selections dict and smoke flag. Output: the stage's Units — the product
    # over its axis tuple, or whatever its grid callable(selections, smoke) returns.
    if callable(stage.grid):
        return list(stage.grid(selections, smoke))
    return product_units(stage.grid, selections, smoke)


def grid_selections(ctx) -> dict:
    # Input: Ctx. Output: the selections handed to grids: the CLI selections plus "pilot", so grids
    # can restrict themselves under --pilot.
    return {**ctx.sel, "pilot": bool(ctx.pilot)}


def units_for(ctx, stage: Stage) -> list:
    # Inputs: Ctx and a Stage. Output: the stage's units for the current selections and flags.
    return expand_units(stage, grid_selections(ctx), ctx.smoke)


def unit_flags(unit: Unit) -> dict:
    # Input: a Unit. Output: {selection key: CLI string} pinning its axes (a size-only unit selects
    # that size's models; lrs use the %g form).
    vals = unit.as_dict()
    out = {"model": vals["size"]} if "size" in vals and "model" not in vals else {}
    out.update({k: (f"{v:g}" if k == "lr" else str(v)) for k, v in vals.items() if k != "size"})
    return out


def derived_units(dep: Stage, parents, fn, sel: dict, smoke: bool) -> list:
    # Inputs: the required Stage, the dependent stage's units, the requirement's fn (None for a plain
    # name), the selections and the smoke flag. Output: the units of dep those parents need: dep's
    # grid expanded with the axes of each pin pinned (the pins are fn(parent), or the parent itself
    # for a plain name) and every other axis following the CLI selection.
    pins = [q for p in parents for q in (fn(p) if fn is not None else [p])]
    return [u for q in pins for u in expand_units(dep, {**sel, **unit_flags(q)}, smoke)]


def plan_units(ctx, stages, requested) -> dict:
    # Inputs: Ctx, Stages in dependency order and the explicitly requested stage names. Output:
    # {stage name: units}: requested stages expand the CLI selections, and every requirement adds the
    # units its dependents need (dependents are visited first, so needs propagate transitively).
    sel = grid_selections(ctx)
    chosen = {s.name for s in stages}
    units = {s.name: (units_for(ctx, s) if s.name in requested else []) for s in stages}
    for stage in reversed(stages):
        for name, fn in map(requirement, stage.requires):
            for dep in (d for d in expand_names([name]) if d in chosen):
                units[dep].extend(derived_units(get_stage(dep), units[stage.name], fn, sel, ctx.smoke))
    return {name: list(dict.fromkeys(us)) for name, us in units.items()}


def _say(msg: str) -> None:
    # Input: a message. Prints it (flushed) on the main rank only.
    if is_main():
        print(msg, flush=True)


def with_force(ctx, force: bool):
    # Inputs: Ctx and the force flag of one stage. Output: ctx itself when the flag already matches,
    # else a copy with that flag, so a dependency never sees the --force of the requested stage.
    return ctx if bool(ctx.force) == bool(force) else dataclasses.replace(ctx, force=bool(force))


def _attempt(summary: dict, stage: str, label: str, call, keep_going: bool) -> bool:
    # Inputs: a run summary, the stage name, the unit label, a zero-argument call and the keep-going
    # flag. Runs the call; a failure is printed with its traceback (so it reaches the unit log) and
    # recorded under "failed" with keep_going, else re-raised. Output: whether it worked.
    try:
        call()
    except Exception as exc:
        _say(f"[{stage}] {label} FAILED: {exc!r}\n{traceback.format_exc()}")
        if not keep_going:
            raise
        summary["failed"].append((label, repr(exc)))
        return False
    return True


def log_name(unit: Unit) -> str:
    # Input: a Unit. Output: its log file stem: the run id when set, else its axis values joined by "_",
    # else "all".
    return unit.run_id or "_".join(str(v) for v in unit.as_dict().values()) or "all"


def unit_log(ctx, stage: str, name: str, label: str):
    # Inputs: Ctx, stage name, log file stem and unit label. Output: a context manager that appends the
    # console output to ROOT/logs/{stage}/{name}.log (with a timestamped header) on the main rank, and
    # does nothing on other ranks.
    if not is_main():
        return contextlib.nullcontext()
    return tee_output(ctx.paths.log(stage, name), f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage} {label}")


def _run_batch(ctx, stage: Stage, todo: list, summary: dict, keep_going: bool) -> None:
    # Inputs: Ctx, Stage, its pending units, the summary and keep-going flag. Runs stage.batch once
    # over all pending units, logging to ROOT/logs/{stage}/batch.log, and records them as ran (or the
    # batch as failed).
    t0 = time.time()
    with unit_log(ctx, stage.name, "batch", f"{len(todo)} units"):
        _say(f"[{stage.name}] batch of {len(todo)} units ...")
        if _attempt(summary, stage.name, "batch", lambda: stage.batch(ctx, todo), keep_going):
            summary["ran"].extend(u.label for u in todo)
            _say(f"[{stage.name}] batch of {len(todo)} units done in {time.time() - t0:.1f}s")


def _run_each(ctx, stage: Stage, todo: list, summary: dict, keep_going: bool) -> None:
    # Inputs: Ctx, Stage, its pending units, the summary and keep-going flag. Runs fn(ctx, unit) for
    # every pending unit inside its unit log, timing each and recording ran or failed units.
    for unit in todo:
        t0 = time.time()
        with unit_log(ctx, stage.name, log_name(unit), unit.label):
            _say(f"[{stage.name}] {unit.label} ...")
            if _attempt(summary, stage.name, unit.label, lambda u=unit: stage.fn(ctx, u), keep_going):
                summary["ran"].append(unit.label)
                _say(f"[{stage.name}] {unit.label} done in {time.time() - t0:.1f}s")


def run_stage(ctx, stage: Stage, units, keep_going: bool = False, force=None) -> dict:
    # Inputs: Ctx, Stage, its Units, whether to continue past failures and an optional force
    # override (default ctx.force). Runs every unit that is not done (or all, when forced) with a ctx
    # carrying this stage's force flag, through stage.batch when it has one and several units are
    # pending. Output: {"ran": [...], "skipped": [...], "failed": [(label, error)]}.
    force = bool(ctx.force if force is None else force)
    sctx = with_force(ctx, force)
    todo = [u for u in units if force or not stage.is_done(sctx, u)]
    summary = {"ran": [], "skipped": [u.label for u in units if u not in todo], "failed": []}
    if stage.batch is not None and len(todo) > 1:
        _run_batch(sctx, stage, todo, summary, keep_going)
    else:
        _run_each(sctx, stage, todo, summary, keep_going)
    return summary


def unit_argv(argv, unit: Unit, force: bool) -> list:
    # Inputs: the CLI flags (without the stage), a Unit and the force flag. Output: the flags with
    # the unit's axes pinned (replacing any selection of those axes) and --force only when forced.
    pins = unit_flags(unit)
    drop, out, skip = {f"--{k}" for k in pins} | {"--force"}, [], False
    for arg in argv:
        flag = arg.split("=", 1)[0]
        if not skip and flag in drop:
            skip = "=" not in arg and flag != "--force"
            continue
        if not skip:
            out.append(arg)
        skip = False
    return out + [x for k, v in pins.items() for x in (f"--{k}", v)] + (["--force"] if force else [])


def relaunch_stage(ctx, stage: Stage, units, gpus: int, argv, force: bool) -> dict:
    # Inputs: Ctx, a DDP Stage, its units, the GPU count, the CLI flags and the force flag. Re-executes
    # the stage under torchrun once per pending unit, with that unit's axes pinned and --no-deps.
    # Output: the run summary.
    sctx = with_force(ctx, force)
    summary = {"ran": [], "skipped": [], "failed": []}
    for unit in units:
        if not force and stage.is_done(sctx, unit):
            summary["skipped"].append(unit.label)
            continue
        code = relaunch_with_torchrun([stage.name, "--no-deps", *unit_argv(argv, unit, force)], gpus)
        summary["ran" if code == 0 else "failed"].append(unit.label if code == 0 else (unit.label, f"exit {code}"))
    return summary


def run_plan(ctx, names, keep_going: bool = False, no_deps: bool = False, gpus: int = 1, argv=()) -> dict:
    # Inputs: Ctx, requested names, keep-going and no-deps flags, GPU count and the CLI flag list
    # (without the stage name). Runs every resolved stage in order over the units plan_units assigns
    # it; DDP stages with gpus > 1 are re-executed under torchrun per unit. --force applies only to
    # explicitly requested stages. Output: {stage name: summary}.
    requested = set(expand_names(names))
    stages = resolve_requires(names, no_deps)
    plan = plan_units(ctx, stages, requested)
    results = {}
    for stage in stages:
        force = bool(ctx.force) and stage.name in requested
        if stage.ddp and int(gpus) > 1 and not under_torchrun():
            results[stage.name] = relaunch_stage(ctx, stage, plan[stage.name], gpus, argv, force)
        else:
            results[stage.name] = run_stage(ctx, stage, plan[stage.name], keep_going, force)
        if results[stage.name]["failed"] and not keep_going:
            raise RuntimeError(f"stage {stage.name} failed: {results[stage.name]['failed']}")
    return results


def unit_status(ctx, stage: Stage, unit: Unit) -> str:
    # Inputs: Ctx, Stage and Unit. Output: "done", "missing" or "error: ..." from is_done.
    try:
        return "done" if stage.is_done(ctx, unit) else "missing"
    except Exception as exc:
        return f"error: {exc!r}"


def _restore(module: str) -> None:
    # Input: a module name. Re-adds the stages and composites that module declared but the current
    # registry lacks (e.g. after the module was first imported inside isolated_registry).
    declared = _DECLARED.get(module, {})
    for name, stage in declared.get("stages", {}).items():
        _STAGES.setdefault(name, stage)
    for name, members in declared.get("composites", {}).items():
        _COMPOSITES.setdefault(name, members)


def discover(packages=DISCOVER_PACKAGES, base: str = "ByteResearch") -> list:
    # Inputs: stage-bearing package names and the base package. Imports each package and every
    # module and subpackage below it (skipping names starting with "_" and names listed in the
    # parent's DISCOVER_SKIP), so @register_stage decorators run, and restores any declaration of an
    # already-imported module that the registry lost. Output: the imported module names.
    imported = []
    for pkg_name in packages:
        pkg = importlib.import_module(f"{base}.{pkg_name}")
        for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            parent, leaf = info.name.rsplit(".", 1)
            skip = getattr(sys.modules.get(parent), "DISCOVER_SKIP", ())
            if leaf.startswith("_") or leaf in skip:
                continue
            importlib.import_module(info.name)
            imported.append(info.name)
    for name in imported:
        _restore(name)
    return imported


def snapshot() -> tuple:
    # Output: copies of the stage and composite registries, for restore().
    return dict(_STAGES), dict(_COMPOSITES)


def restore(saved: tuple) -> None:
    # Input: a snapshot() result. Replaces the registry contents with it.
    _STAGES.clear()
    _COMPOSITES.clear()
    _STAGES.update(saved[0])
    _COMPOSITES.update(saved[1])


@contextlib.contextmanager
def isolated_registry():
    # Context manager for tests: snapshots the stage and composite registries, yields an empty
    # registry, and restores the snapshot on exit.
    saved = snapshot()
    restore(({}, {}))
    try:
        yield
    finally:
        restore(saved)
