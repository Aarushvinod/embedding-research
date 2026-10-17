# The run context handed to every stage function as fn(ctx, unit).
# Ctx bundles the artifact paths, the smoke/pilot/force flags, the seed, the torch device, the
# process placement, determinism, the CLI axis selections and the per-stage extra options.
# ctx.config(cls) returns a config dataclass with smoke overrides applied, so stages never
# special-case smoke mode themselves.
# ctx.rows(n) returns all row indices, or a fixed seeded subset in --pilot mode.
# build_ctx turns the parsed argparse namespace of run.py into a Ctx without side effects.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import InterpConfig, smoke_overrides
from .dist import World, world
from .paths import Paths, artifacts_root

GLOBAL_FLAGS = ("stage", "model", "condition", "target", "testset", "metric", "lr", "artifacts", "smoke",
                "pilot", "force", "gpus", "keep_going", "deterministic", "no_deps", "seed", "device")
SELECTION_KEYS = ("model", "condition", "target", "testset", "metric", "lr", "lang", "variant")


@dataclass
class Ctx:
    # Run context: paths, flags, seed, device string, extra stage options, World, determinism,
    # the CLI selections dict and the requested GPU count.
    paths: Paths
    smoke: bool = False
    pilot: bool = False
    force: bool = False
    seed: int = 0
    device: str = "cpu"
    opts: dict = field(default_factory=dict)
    world: World = field(default_factory=world)
    deterministic: bool = False
    sel: dict = field(default_factory=dict)
    gpus: int = 1

    def config(self, cls):
        # Input: a config dataclass class. Output: its default instance, smoke-overridden in smoke mode.
        cfg = cls()
        return smoke_overrides(cfg) if self.smoke else cfg

    def opt(self, name: str, default=None):
        # Inputs: an extra-option dest name and a default. Output: its value, or the default when unset.
        value = self.opts.get(name)
        return default if value is None else value

    def rows(self, n_total: int, k: int | None = None, seed: int | None = None) -> list:
        # Inputs: number of available rows, optional pilot size and seed. Output: all indices, or
        # in pilot mode a sorted seeded subset of size min(k or InterpConfig.pilot_rows, n_total).
        if not self.pilot:
            return list(range(n_total))
        size = min(int(k or self.config(InterpConfig).pilot_rows), n_total)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        return sorted(int(i) for i in rng.choice(n_total, size=size, replace=False))


def default_device(w: World) -> str:
    # Input: the World. Output: "cuda:{local_rank}" when CUDA is available, else "cpu".
    import torch
    return f"cuda:{w.local_rank}" if torch.cuda.is_available() else "cpu"


def build_ctx(args=None, **overrides) -> Ctx:
    # Inputs: an argparse namespace from run.py (or None for defaults) and field overrides.
    # Output: the Ctx: root from --artifacts/--smoke, flags, seed, device, selections and every
    # non-global argument collected into opts.
    a = vars(args) if args is not None else {}
    w = world()
    ctx = Ctx(
        paths=Paths(artifacts_root(a.get("artifacts"), bool(a.get("smoke")))),
        smoke=bool(a.get("smoke")),
        pilot=bool(a.get("pilot")),
        force=bool(a.get("force")),
        seed=int(a.get("seed") or 0),
        device=a.get("device") or default_device(w),
        opts={k: v for k, v in a.items() if k not in GLOBAL_FLAGS},
        world=w,
        deterministic=bool(a.get("deterministic")),
        sel={k: a.get(k) for k in SELECTION_KEYS if a.get(k) is not None},
        gpus=int(a.get("gpus") or 1),
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx
