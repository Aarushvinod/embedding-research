# Steering settings: which layers are edited, with which direction, by which operation.
# A Setting is base (no edit), an all-layers edit or a single-layer edit, combined with a direction
# (English, one of the language controls, or the random control) and either graded subtraction of
# c times the English norm along the direction's unit vector or the mean-centered projection.
# Setting ids are short strings such as "all:en:c0.25" or "L3:zh:proj" that round-trip exactly and
# serve as the unit keys of every steering result file; c = 0 is never enumerated because it is
# the base run, which the logprob identity check verifies bitwise.
# Grids list the settings of the select sweep (every layer) and of the devtest evaluation
# (all layers plus the chosen layer), and select_layer applies the pre-registered layer rule.
# edits_for turns a setting and a direction set into encoder or decoder block edits for a site.

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from ...core.config import CONTROLS, InterpConfig
from ...interp.readouts import edited_provider, plain_provider
from ...modeling.encode import PREFIX_LEN
from ...modeling.hooks import from_position

DIRECTIONS = ("en",) + tuple(CONTROLS) + ("rand",)
BASE_ID = "base"


@dataclass(frozen=True)
class SteerGrid:
    # Scale grids: single-layer scales (evaluation), all-layers scales, select-sweep scales, the
    # scale the layer rule reads, whether the projection endpoint is included, and the directions.
    single_scales: tuple = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    all_scales: tuple = (0.05, 0.1, 0.25, 0.5)
    select_scales: tuple = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    select_c: float = 1.0
    proj: bool = True
    directions: tuple = DIRECTIONS


PILOT_GRID = SteerGrid(single_scales=(1.0, 2.0), all_scales=(0.1, 0.25), select_scales=(1.0,))


@dataclass(frozen=True)
class Setting:
    # One steering setting: scope ("base", "all" or "L<k>"), direction code, operation ("c" for
    # graded subtraction or "proj") and the scale c (unused for proj and base).
    scope: str
    direction: str = ""
    op: str = ""
    c: float = 0.0

    @property
    def layer(self):
        # Output: the edited layer index for a single-layer scope, otherwise None.
        return int(self.scope[1:]) if self.scope.startswith("L") else None


def full_grid(cfg: InterpConfig) -> SteerGrid:
    # Input: InterpConfig. Output: the default full grid using the configured single-layer scales and
    # control languages; the select sweep scores only the scale the layer rule reads (select_c).
    dirs = ("en",) + tuple(cfg.steer_controls) + ("rand",)
    return SteerGrid(single_scales=tuple(cfg.steer_scales), select_scales=(float(cfg.steer_select_c),),
                     select_c=float(cfg.steer_select_c), directions=dirs)


def with_scales(grid: SteerGrid, single=None, all_layers=None, select=None) -> SteerGrid:
    # Inputs: a grid and optional comma-separated scale strings. Output: the grid with any given
    # scale lists replaced (negative scales add the direction instead of subtracting it).
    parse = lambda s: tuple(float(x) for x in str(s).split(",") if x.strip())
    over = {k: parse(v) for k, v in (("single_scales", single), ("all_scales", all_layers),
                                     ("select_scales", select)) if v}
    return replace(grid, **over)


def setting_id(s: Setting) -> str:
    # Input: a Setting. Output: its id string ("base", "<scope>:<dir>:c<repr c>" or "<scope>:<dir>:proj").
    if s.scope == BASE_ID:
        return BASE_ID
    return f"{s.scope}:{s.direction}:{'proj' if s.op == 'proj' else 'c' + repr(float(s.c))}"


def parse_setting(sid: str) -> Setting:
    # Input: a setting id. Output: the Setting it encodes; raises ValueError on malformed ids.
    if sid == BASE_ID:
        return Setting(BASE_ID)
    scope, direction, op = sid.split(":")
    if op == "proj":
        return Setting(scope, direction, "proj")
    if not op.startswith("c"):
        raise ValueError(f"bad setting id {sid!r}")
    return Setting(scope, direction, "c", float(op[1:]))


def scope_settings(scope: str, grid: SteerGrid, scales) -> list:
    # Inputs: a scope, the grid and the scale list. Output: Settings for every direction and nonzero
    # scale, plus the projection per direction when the grid includes it.
    subs = [Setting(scope, d, "c", float(c)) for d in grid.directions for c in scales if float(c) != 0.0]
    projs = [Setting(scope, d, "proj") for d in grid.directions] if grid.proj else []
    return subs + projs


def select_settings(n_layers: int, grid: SteerGrid) -> list:
    # Inputs: number of layers at the site and the grid. Output: base plus the single-layer sweep of
    # every layer, direction and select scale (graded subtraction only: the layer rule reads no projection).
    sub = replace(grid, proj=False)
    return [Setting(BASE_ID)] + [s for l in range(n_layers) for s in scope_settings(f"L{l}", sub, grid.select_scales)]


def eval_settings(chosen: int, grid: SteerGrid) -> list:
    # Inputs: the chosen layer and the grid. Output: base, the all-layers grid and the chosen-layer
    # grid (each with projections) for the devtest measurements.
    return [Setting(BASE_ID)] + scope_settings("all", grid, grid.all_scales) + \
        scope_settings(f"L{int(chosen)}", grid, grid.single_scales)


def check_settings(chosen: int) -> list:
    # Input: the chosen layer. Output: the settings probed by the rebuild check (chosen-layer en c=1,
    # chosen-layer en projection and all-layers en c=0.25).
    return [Setting(f"L{int(chosen)}", "en", "c", 1.0), Setting(f"L{int(chosen)}", "en", "proj"),
            Setting("all", "en", "c", 0.25)]


def group_label(s: Setting) -> str:
    # Input: a Setting. Output: its model-independent group label ("base", "all:c0.25", "best:proj"),
    # where "best" stands for the model's own chosen layer.
    if s.scope == BASE_ID:
        return BASE_ID
    scope = "all" if s.scope == "all" else "best"
    return f"{scope}:{'proj' if s.op == 'proj' else 'c' + repr(float(s.c))}"


def select_layer(table: dict) -> int:
    # Input: {layer: pooled English-specific damage}. Output: the layer with the largest value
    # (NaN counts as -inf), ties going to the shallower layer.
    layers = sorted(int(l) for l in table)
    scores = [table[l] if not math.isnan(table[l]) else -math.inf for l in layers]
    return layers[int(np.argmax(scores))]


def edit_fn(s: Setting, dset, layer: int, device):
    # Inputs: a Setting, a DirectionSet, the layer and the torch device. Output: fn(h) implementing
    # h - c*||v_en||*u_dir (subtraction) or h - ((h - mu).u)u (mean-centered projection) at that layer.
    u = torch.as_tensor(dset.unit[s.direction][layer], dtype=torch.float32, device=device)
    if s.op == "proj":
        mu = torch.as_tensor(dset.mu[layer], dtype=torch.float32, device=device)
        return lambda h: h - ((h - mu) @ u)[..., None] * u
    shift = torch.as_tensor(float(s.c) * float(dset.norm[layer]) * dset.unit[s.direction][layer],
                            dtype=torch.float32, device=device)
    return lambda h: h - shift


def edit_layers(s: Setting, n_layers: int) -> list:
    # Inputs: a Setting and the site's layer count. Output: the layers it edits (none for base).
    if s.scope == BASE_ID:
        return []
    return list(range(n_layers)) if s.scope == "all" else [s.layer]


def edits_for(trained, site: str, s: Setting, dset) -> tuple:
    # Inputs: Trained, site ("enc" or "dec"), a Setting and its target's DirectionSet. Output:
    # (encoder edits, decoder edits) as {layer: fn}; decoder edits start at the tag position.
    n = trained.n_enc if site == "enc" else trained.n_dec
    fns = {l: edit_fn(s, dset, l, trained.device) for l in edit_layers(s, n)}
    if site == "enc":
        return fns, None
    return None, {l: from_position(f, PREFIX_LEN - 1) for l, f in fns.items()}


def site_edits(trained, site: str, s: Setting, dset) -> tuple:
    # Inputs: as edits_for. Output: (provider, decoder edits) for interp.readouts: an edited encoder
    # provider at the encoder site, or the plain provider plus decoder edits at the decoder site.
    enc, dec = edits_for(trained, site, s, dset)
    return (edited_provider(trained, enc) if enc else plain_provider(trained)), dec
