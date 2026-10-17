# Cross-model encoder stitching mechanics for same-architecture fine-tuned T5 models.
# A donor runs its embedding and encoder blocks 1..l on its own source, and the recipient continues
# with its blocks l+1..L, its final layer norm and (later) its decoder.
# The recipient never calls a T5Block by hand: it runs its own encoder on the donor's input ids and
# mask, so its block 0 builds its own relative position bias and mask for the donor length, and an
# edit hook replaces the output of recipient block l-1 with the donor state (l = 0 feeds the
# donor embedding through inputs_embeds instead).
# Donor and recipient must share the model class, encoder geometry and tokenizer, which
# assert_compatible enforces (ByT5 to ByT5 and mT5 to mT5 only).
# identity_check verifies that self-stitching at every l reproduces the normal forward bitwise, and
# precompute_states/cached_provider stage donor states in per-layer memmaps so large models can be
# resident one at a time.

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from ...core.io import write_json_atomic
from ...core.paths import ensure
from ...interp.scoring import encode_pass, run_batched, to_device
from ...modeling import hooks
from ...modeling.encode import collate_sources, source_rows

PROBE = "Probe: The cat sat. Habari! მე ვარ. Это тест. مرحبا. <extra_id_0> 12345"
_CONFIG_KEYS = ("model_type", "d_model", "d_kv", "d_ff", "num_heads", "num_layers", "vocab_size",
                "relative_attention_num_buckets", "relative_attention_max_distance", "feed_forward_proj",
                "layer_norm_epsilon", "is_gated_act")


class _Halt(Exception):
    # Raised by a forward hook to stop the donor encoder right after the last block it must run.
    pass


def _halt(module, args, out):
    # Inputs: a forward hook's module, inputs and output. Raises _Halt to end the forward pass early.
    raise _Halt


def _signature(trained) -> dict:
    # Input: a Trained bundle. Output: the architecture and tokenizer facts two stitched models must
    # share (model class, encoder config fields, tokenizer class, size and encoding of a probe text).
    cfg = trained.model.config
    tok = trained.tok
    return {"class": type(trained.model).__name__, "arch": trained.arch,
            **{k: getattr(cfg, k, None) for k in _CONFIG_KEYS},
            "tokenizer": type(tok).__name__, "tok_len": len(tok), "probe": source_rows(tok, [PROBE], 512)[0]}


def assert_compatible(donor, recipient) -> None:
    # Inputs: donor and recipient Trained bundles. Raises AssertionError unless both have the same
    # model class, architecture family, encoder geometry and tokenizer (so ByT5 <-> mT5 is refused).
    a, b = _signature(donor), _signature(recipient)
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    if diff:
        raise AssertionError(f"stitching needs identical architectures and tokenizers; differences: {diff}")


def _dtype(model) -> torch.dtype:
    # Input: a model. Output: its parameter dtype.
    return next(model.parameters()).dtype


def donor_state(donor, batch: dict, layer: int) -> torch.Tensor:
    # Inputs: donor Trained, an encoder batch of the donor's own source and the stitch layer l.
    # Output: float32 [B, S, d] donor state after l blocks (the embedding output for l = 0, read by
    # calling embed_tokens directly); the donor encoder stops right after block l-1.
    model, b = donor.model, to_device(batch, donor.device)
    if layer == 0:
        with torch.no_grad():
            return model.encoder.embed_tokens(b["input_ids"]).float()
    mod = hooks.blocks(model, "enc")[layer - 1]
    with torch.no_grad(), hooks.record({layer: mod}) as rec, hooks.attach([(mod, _halt)]), \
            contextlib.suppress(_Halt):
        model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    return rec.out[layer]


def _replacer(h: torch.Tensor):
    # Input: the donor state [B, S, d]. Output: an edit fn returning h in place of a block output of
    # the same shape (a shape mismatch raises ValueError).
    def fn(out):
        # Input: the recipient block output. Output: the donor state.
        if out.shape != h.shape:
            raise ValueError(f"donor state {tuple(h.shape)} does not match recipient block output {tuple(out.shape)}")
        return h
    return fn


def continue_from(recipient, batch: dict, h: torch.Tensor, layer: int):
    # Inputs: recipient Trained, the donor-source encoder batch, a donor state after l blocks and l.
    # Runs the recipient encoder on the donor ids and mask with block l-1's output replaced by h
    # (l = 0: inputs_embeds = h), so blocks l..L-1 and the final norm use the recipient's own bias.
    # Output: (BaseModelOutput with float32 last_hidden_state, attention mask on the recipient device).
    model, b = recipient.model, to_device(batch, recipient.device)
    h = h.to(device=recipient.device, dtype=_dtype(model))
    mask = b["attention_mask"]
    with torch.no_grad():
        if layer == 0:
            out = model.encoder(inputs_embeds=h, attention_mask=mask)
        else:
            with hooks.edit_blocks(model, "enc", {layer - 1: _replacer(h)}):
                out = model.encoder(input_ids=b["input_ids"], attention_mask=mask)
    return BaseModelOutput(last_hidden_state=out.last_hidden_state.float()), mask


def stitched_encoder(donor, recipient, batch: dict, layer: int):
    # Inputs: donor and recipient Trained, an encoder batch of the donor's own source and l.
    # Output: (encoder outputs, mask) of donor blocks 1..l followed by recipient blocks l+1..L.
    return continue_from(recipient, batch, donor_state(donor, batch, layer), layer)


def stitch_provider(donor, recipient, layer: int):
    # Inputs: donor and recipient Trained and l. Output: a readouts provider(batch, rows) that
    # stitches a batch of donor-source ids (the shared tokenizer makes the batch valid for both).
    return lambda batch, rows=None: stitched_encoder(donor, recipient, batch, layer)


def identity_check(trained, texts, rows, max_chars: int, factor: int) -> dict:
    # Inputs: a Trained bundle, source texts, the rows to check, character cap and token factor.
    # Self-stitches at every l in 0..L and compares last_hidden_state with the normal forward.
    # Output: {"rows", "layers": {l: bool}, "max_abs_diff": {l: float}, "ok"}.
    batch = collate_sources(source_rows(trained.tok, [texts[r] for r in rows], max_chars, factor),
                            trained.tok.pad_token_id)
    ref = encode_pass(trained, batch)[0].last_hidden_state
    equal, diff = {}, {}
    for layer in range(trained.n_enc + 1):
        out = stitched_encoder(trained, trained, batch, layer)[0].last_hidden_state
        equal[layer] = bool(torch.equal(out, ref))
        diff[layer] = float((out - ref).abs().max())
    return {"rows": [int(r) for r in rows], "layers": equal, "max_abs_diff": diff, "ok": all(equal.values())}


def _cpu(t: torch.Tensor) -> torch.Tensor:
    # Input: a hooked tensor. Output: a detached float32 CPU copy.
    return t.detach().float().cpu()


def _cache_meta(donor, texts, rows, max_chars: int, factor: int):
    # Inputs: donor Trained, source texts, rows, cap and factor. Output: (per-row token lengths,
    # per-row offsets into a flat [total, d] array, the meta dict describing the cache).
    lengths = np.array([len(x) for x in source_rows(donor.tok, [texts[r] for r in rows], max_chars, factor)])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    meta = {"rows": [int(r) for r in rows], "lengths": lengths.tolist(), "offsets": offsets.tolist(),
            "d_model": int(donor.d_model), "max_chars": max_chars, "factor": factor,
            "donor": {"key": donor.key, "condition": donor.condition, "step": donor.meta.get("step")}}
    return lengths, offsets, meta


def _cache_ok(path: Path, meta: dict) -> bool:
    # Inputs: a cached layer file and the expected meta. Output: whether the cache is complete and
    # was written for the same donor, rows, cap and factor.
    stored = Path(str(path) + ".json")
    return path.exists() and stored.exists() and json.loads(stored.read_text(encoding="utf-8")) == meta


def cache_path(out_dir, layer: int) -> Path:
    # Inputs: cache directory and l. Output: the .npy memmap holding the donor states after l blocks.
    return Path(out_dir) / f"l{int(layer)}.npy"


def precompute_states(donor, texts, rows, layers, out_dir, max_chars: int, factor: int, batching) -> list:
    # Inputs: donor Trained, its source texts, rows, stitch layers, cache directory, cap, factor and
    # Batching. Runs the donor once per batch, recording every requested layer, and writes the
    # unpadded per-row states into one float32 memmap per layer (plus a meta .json written last, so
    # an interrupted cache is rebuilt). Output: the layers that had to be computed.
    lengths, offsets, meta = _cache_meta(donor, texts, rows, max_chars, factor)
    todo = [int(l) for l in layers if not _cache_ok(cache_path(out_dir, l), meta)]
    if not todo:
        return []
    ensure(Path(out_dir))
    shape = (int(lengths.sum()), int(donor.d_model))
    mms = {l: np.lib.format.open_memmap(cache_path(out_dir, l), mode="w+", dtype=np.float32, shape=shape) for l in todo}
    pos = {int(r): p for p, r in enumerate(rows)}
    mods = {l: hooks.blocks(donor.model, "enc")[l - 1] for l in todo if l > 0}

    def step(enc, brows):
        # Inputs: a donor-source batch and its rows. Records the requested states and copies each
        # row's unpadded slice into the memmaps. Output: one None per row.
        b = to_device(enc, donor.device)
        with torch.no_grad(), hooks.record(mods, reduce=_cpu) as rec, \
                hooks.record({0: hooks.embed_prehook_module(donor.model)}, pre=True, reduce=_cpu) as emb:
            donor.model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        for l in todo:
            st = (emb.out[0] if l == 0 else rec.out[l]).numpy()
            _write_rows(mms[l], st, [pos[int(r)] for r in brows], offsets, lengths)
        return [None] * len(brows)

    run_batched(donor, texts, list(rows), max_chars, factor, batching, step)
    for l in todo:
        mms[l].flush()
        write_json_atomic(Path(str(cache_path(out_dir, l)) + ".json"), meta)
    return todo


def _write_rows(mm, states: np.ndarray, positions, offsets, lengths) -> None:
    # Inputs: a [total, d] memmap, batch states [B, S, d], each batch row's position in the cached
    # row list, and the per-row offsets and lengths. Copies each row's unpadded states into place.
    for i, p in enumerate(positions):
        mm[offsets[p]:offsets[p] + lengths[p]] = states[i, :lengths[p]]


def cached_provider(recipient, out_dir, layer: int):
    # Inputs: recipient Trained, a cache directory written by precompute_states and l. Output: a
    # readouts provider(batch, rows) that pads the cached donor states of the batch rows (zeros at
    # padding, which the mask hides) and continues them through the recipient.
    path = cache_path(out_dir, layer)
    meta = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    mm = np.load(path, mmap_mode="r")
    pos = {int(r): p for p, r in enumerate(meta["rows"])}
    offsets, lengths = meta["offsets"], meta["lengths"]

    def provide(batch, rows):
        # Inputs: a donor-source batch and its rows. Output: (encoder outputs, mask) for the rows.
        mask = batch["attention_mask"]
        h = torch.zeros((mask.shape[0], mask.shape[1], meta["d_model"]), dtype=torch.float32)
        for i, r in enumerate(rows):
            p = pos[int(r)]
            if int(mask[i].sum()) != lengths[p]:
                raise ValueError(f"cached length {lengths[p]} != batch length for row {r}")
            h[i, :lengths[p]] = torch.from_numpy(np.array(mm[offsets[p]:offsets[p] + lengths[p]]))
        return continue_from(recipient, batch, h, layer)
    return provide
