# Forward-hook toolkit for reading and editing T5 internals during interpretability runs.
# Accessors locate encoder/decoder blocks, each layer's gated-FFN act module (the GELU branch),
# and the encoder block-0 module whose forward pre-hook sees the embedding output.
# attach() registers hooks inside an ExitStack so they are always removed, even after errors.
# record() captures module outputs (out[0] for tuples and ModelOutputs) in float32 or through a
# reduce function, and edit() replaces out[0] with fn(out[0]) cast back to the original dtype.
# Hooks fire on every call, so decoder edits also apply on each cached generation step.
# Hooks on one module run in registration order; a later hook sees an earlier hook's edit.

from __future__ import annotations

import contextlib
from collections.abc import Mapping

import torch

_SIDES = {"enc": "encoder", "encoder": "encoder", "dec": "decoder", "decoder": "decoder"}


def stack(model, side: str):
    # Inputs: a T5ForConditionalGeneration and side ("enc"/"encoder" or "dec"/"decoder").
    # Output: the T5Stack for that side.
    return getattr(model, _SIDES[side])


def blocks(model, side: str):
    # Inputs: model and side. Output: the ModuleList of T5Blocks on that side.
    return stack(model, side).block


def ffn_act(model, side: str, layer: int):
    # Inputs: model, side and layer index. Output: that layer's gated-FFN act module, whose output
    # is gelu(x W_i0) of shape [B, T, d_ff].
    return blocks(model, side)[layer].layer[-1].DenseReluDense.act


def act_modules(model, side: str, layers=None) -> dict:
    # Inputs: model, side and layer indices (None = all). Output: {layer: act module}.
    layers = range(len(blocks(model, side))) if layers is None else layers
    return {int(l): ffn_act(model, side, l) for l in layers}


def block_modules(model, side: str, layers=None) -> dict:
    # Inputs: model, side and layer indices (None = all). Output: {layer: T5Block}.
    layers = range(len(blocks(model, side))) if layers is None else layers
    return {int(l): blocks(model, side)[l] for l in layers}


def embed_prehook_module(model):
    # Input: model. Output: encoder block 0, whose forward pre-hook args[0] is the embedding output
    # (the shared embedding table itself must never be hooked).
    return model.encoder.block[0]


def first_tensor(out):
    # Input: a module output (tuple, ModelOutput/Mapping or tensor). Output: its first tensor.
    if torch.is_tensor(out):
        return out
    if isinstance(out, Mapping):
        return next(iter(out.values()))
    return out[0]


def replace_first(out, new):
    # Inputs: a module output and a replacement tensor. Output: the same kind of output with its
    # first tensor replaced and every other element passed through.
    if torch.is_tensor(out):
        return new
    if isinstance(out, Mapping):
        copy = type(out)(**out)
        copy[next(iter(out.keys()))] = new
        return copy
    return (new,) + tuple(out[1:])


@contextlib.contextmanager
def attach(pairs):
    # Input: iterable of (module, hook) or (module, hook, kind) with kind "forward" (default) or
    # "pre". Registers every hook, yields, and removes them all on exit, including on exceptions.
    with contextlib.ExitStack() as es:
        for item in pairs:
            module, hook, kind = (tuple(item) + ("forward",))[:3]
            reg = module.register_forward_pre_hook if kind == "pre" else module.register_forward_hook
            es.callback(reg(hook).remove)
        yield


class Recorder:
    # Holder for recorded values: out[key] is the latest captured value (or the list of all calls
    # when accumulating) and calls[key] counts how often the module fired.

    def __init__(self, accumulate: bool = False):
        # Input: whether to keep every call's value in a list instead of only the latest.
        self.accumulate = accumulate
        self.out = {}
        self.calls = {}

    def store(self, key, value) -> None:
        # Inputs: a key and a captured value. Saves it according to the accumulate mode.
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.accumulate:
            self.out.setdefault(key, []).append(value)
        else:
            self.out[key] = value


def _capture(rec: Recorder, key, reduce, pre: bool):
    # Inputs: recorder, key, optional reduce fn and pre-hook flag. Output: a hook storing the
    # detached float32 tensor (or reduce(tensor)) under key.
    def grab(t):
        # Input: the hooked tensor. Stores it (float32, detached) or its reduction under key.
        rec.store(key, reduce(t) if reduce is not None else t.detach().float())

    if pre:
        return lambda mod, args: grab(args[0])
    return lambda mod, args, out: grab(first_tensor(out))


@contextlib.contextmanager
def record(modules: dict, reduce=None, pre: bool = False, accumulate: bool = False):
    # Inputs: {key: module}, optional reduce(tensor) -> value, whether to read the module input
    # (pre-hook) instead of its output, and whether to keep all calls. Yields a Recorder whose
    # out[key] holds float32 tensors or reduced values.
    rec = Recorder(accumulate)
    kind = "pre" if pre else "forward"
    with attach((m, _capture(rec, k, reduce, pre), kind) for k, m in modules.items()):
        yield rec


def _apply(fn, h):
    # Inputs: an edit fn and a tensor. Output: fn(h as float32) cast back to h's dtype.
    return fn(h.float()).to(h.dtype)


def _edit_hook(fn, pre: bool):
    # Inputs: an edit fn and pre-hook flag. Output: a hook replacing the module's first input
    # (pre) or first output tensor with the edited tensor.
    if pre:
        return lambda mod, args: (_apply(fn, args[0]),) + tuple(args[1:])
    return lambda mod, args, out: replace_first(out, _apply(fn, first_tensor(out)))


@contextlib.contextmanager
def edit(edits: dict, pre: bool = False):
    # Inputs: {module: fn} where fn maps a float32 tensor to a same-shape tensor without editing it
    # in place, and whether to edit the module input instead of its output. Yields while the edits
    # are active on every call (including cached decoder steps); hooks are removed on exit.
    kind = "pre" if pre else "forward"
    with attach((m, _edit_hook(fn, pre), kind) for m, fn in edits.items()):
        yield


def record_blocks(model, side: str, layers=None, reduce=None, accumulate: bool = False):
    # Inputs: model, side, layer indices (None = all), optional reduce and accumulate flag.
    # Output: a record() context over {layer: block output}.
    return record(block_modules(model, side, layers), reduce=reduce, accumulate=accumulate)


def edit_blocks(model, side: str, fns: dict):
    # Inputs: model, side and {layer: fn}. Output: an edit() context over those block outputs.
    return edit({blocks(model, side)[l]: fn for l, fn in fns.items()})


def record_acts(model, side: str, layers=None, reduce=None):
    # Inputs: model, side, layer indices (None = all) and optional reduce. Output: a record()
    # context over {layer: gated-FFN act output}.
    return record(act_modules(model, side, layers), reduce=reduce)


def edit_acts(model, side: str, fns: dict):
    # Inputs: model, side and {layer: fn}. Output: an edit() context over those act outputs.
    return edit({ffn_act(model, side, l): fn for l, fn in fns.items()})


def resolve_edits(model, side: str, edits) -> dict:
    # Inputs: model, side and edits given as {layer int: fn} (block outputs) or {module: fn}.
    # Output: {module: fn} (empty for None).
    edits = edits or {}
    return {(blocks(model, side)[k] if isinstance(k, int) else k): fn for k, fn in edits.items()}


def set_units_fn(idx, values):
    # Inputs: unit indices along the last dim and replacement values (scalar or one per index).
    # Output: fn(h) returning a copy of h with h[..., idx] = values (for neuron mean ablation).
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    val_t = torch.as_tensor(values, dtype=torch.float32)

    def fn(h):
        # Input: an act output [B, T, d_ff]. Output: a copy with the chosen units replaced.
        out = h.clone()
        out[..., idx_t.to(h.device)] = val_t.to(device=h.device, dtype=h.dtype)
        return out
    return fn


def from_position(fn, start: int):
    # Inputs: an edit fn and the first position to edit. Output: fn'(h) that edits only positions
    # >= start of a full-length call and the whole tensor of a cached single-step call (T == 1),
    # e.g. to start decoder edits at the tag position (start=1).
    def wrapped(h):
        # Input: a hidden-state tensor [B, T, d]. Output: the tensor edited from position start on.
        if h.shape[1] == 1:
            return fn(h)
        return torch.cat([h[:, :start], fn(h[:, start:])], dim=1)
    return wrapped
