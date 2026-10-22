# Command-line entry point: `python -m ByteResearch.run <stage> [flags]`.
# It discovers every registered stage by importing the stage-bearing packages.
# One argparse parser is built from the global flags plus each stage's declared options, and the
# parsed arguments become the Ctx that every stage function receives.
# The built-in `list` stage prints the registered stages and composites, and `status` prints a
# done/missing grid for every stage over the units the current selections expand to.
# Any other name is a stage or a composite (`all`, `interp.<exp>`, ...) and runs in dependency
# order, skipping finished units unless --force.
# DDP stages re-launch themselves under torchrun when --gpus asks for more than one process.

from __future__ import annotations

import argparse
import sys

from .core.context import build_ctx
from .core.io import utf8_stdout
from .core.registry import BUILTIN, discover, is_known, list_composites, list_stages, requirement, run_plan, \
    unit_status, units_for
from .core.seeding import seed_everything


def build_parser(stages) -> argparse.ArgumentParser:
    # Input: registered Stage objects. Output: the argparse parser with the positional stage, the
    # global flags and every stage-declared option (the first declaration of a flag wins).
    p = argparse.ArgumentParser(prog="python -m ByteResearch.run", description="ByteResearch pipeline")
    p.add_argument("stage", help="stage, composite (all, interp.<exp>, ...), list or status")
    p.add_argument("--model", default="all", help="all | small | base | large | byt5 | mt5 | keys, comma list")
    p.add_argument("--condition", default="all", help="all | C1,C2,C3")
    p.add_argument("--target", default="all", help="all | target codes, comma list")
    p.add_argument("--testset", default="all", help="all | flores_devtest,ntrex")
    p.add_argument("--metric", default="all", help="all | sacre | neural | chrf_pp,bleu,spbleu,comet,metricx")
    p.add_argument("--lr", default=None, help="learning rate(s), comma list (default: the sweep grid)")
    p.add_argument("--artifacts", default=None, help="artifacts root (default $BYTERESEARCH_ARTIFACTS or ByteResearch/artifacts)")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run in ROOT/smoke")
    p.add_argument("--pilot", action="store_true", help="interp on fixed row subsets")
    p.add_argument("--force", action="store_true", help="rerun requested stages even if done")
    p.add_argument("--gpus", type=int, default=1, help="processes for DDP stages (torchrun relaunch)")
    p.add_argument("--keep-going", action="store_true", help="continue past failing units")
    p.add_argument("--deterministic", action="store_true", help="deterministic kernels, no TF32")
    p.add_argument("--no-deps", action="store_true", help="do not run required stages first")
    p.add_argument("--seed", type=int, default=0, help="global seed")
    p.add_argument("--device", default=None, help="torch device (default cuda:LOCAL_RANK or cpu)")
    seen = {s for a in p._actions for s in a.option_strings}
    for stage in stages:
        new = [o for o in stage.options if not seen.intersection(o.flags)]
        if not new:
            continue
        group = p.add_argument_group(f"options for {stage.name}")
        for o in new:
            group.add_argument(*o.flags, **o.kwargs)
            seen.update(o.flags)
    return p


def flags_argv(argv: list, stage: str, parser: argparse.ArgumentParser) -> list:
    # Inputs: the raw CLI arguments, the parsed stage name and the parser. Output: the arguments
    # without the stage positional (its first occurrence that is not the value of a preceding
    # option, e.g. `--model all all`), for torchrun re-launches.
    takes_value = {f for a in parser._actions if a.option_strings and a.nargs != 0 for f in a.option_strings}
    for i, arg in enumerate(argv):
        if arg == stage and not (i and argv[i - 1] in takes_value):
            return list(argv[:i]) + list(argv[i + 1:])
    return list(argv)


def print_list() -> None:
    # Prints every registered stage (order, grid, requirements, DDP flag, options) and every composite.
    stages = list_stages()
    print(f"built-in: {', '.join(BUILTIN)}")
    if not stages:
        print("no stages registered")
        return
    for s in stages:
        grid = "custom" if callable(s.grid) else (",".join(s.grid) or "-")
        opts = " ".join(f for o in s.options for f in o.flags[:1])
        reqs = ",".join(requirement(r)[0] for r in s.requires) or "-"
        print(f"{s.name:42} order={s.order:<4} grid={grid:28} requires={reqs}"
              f"{' ddp' if s.ddp else ''}{'  ' + opts if opts else ''}")
    for name, members in list_composites().items():
        print(f"composite {name}: {', '.join(members)}")


def print_status(ctx) -> None:
    # Input: Ctx. Prints, for every registered stage, done/total and one [x]/[ ] line per unit of the
    # current selections (errors from is_done are shown inline).
    stages = list_stages()
    if not stages:
        print("no stages registered")
        return
    for s in stages:
        units = units_for(ctx, s)
        states = [unit_status(ctx, s, u) for u in units]
        print(f"{s.name:42} {sum(st == 'done' for st in states)}/{len(units)}")
        for u, st in zip(units, states):
            note = "" if st in ("done", "missing") else f"  ({st})"
            print(f"    [{'x' if st == 'done' else ' '}] {u.run_id or ''} {u.label}{note}")


def main(argv=None) -> int:
    # Input: CLI arguments (default sys.argv[1:]). Discovers stages, parses arguments, builds the
    # Ctx and runs list, status or the requested stage/composite. Output: the process exit code.
    utf8_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)
    discover()
    parser = build_parser(list_stages())
    args = parser.parse_args(argv)
    ctx = build_ctx(args)
    if args.stage == "list":
        print_list()
        return 0
    if args.stage == "status":
        print_status(ctx)
        return 0
    if not is_known(args.stage):
        parser.error(f"unknown stage {args.stage!r}; try `list`")
    seed_everything(ctx.seed, ctx.deterministic)
    results = run_plan(ctx, [args.stage], keep_going=args.keep_going, no_deps=args.no_deps, gpus=args.gpus,
                       argv=flags_argv(argv, args.stage, parser))
    failed = {name: r["failed"] for name, r in results.items() if r["failed"]}
    for name, r in results.items():
        print(f"{name}: ran {len(r['ran'])}, skipped {len(r['skipped'])}, failed {len(r['failed'])}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
