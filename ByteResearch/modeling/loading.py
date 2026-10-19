# Model and tokenizer loading for training, evaluation and interpretability.
# load_pretrained fetches a Hugging Face ByT5 or mT5 checkpoint by registry key.
# load_trained opens one of our fine-tuned checkpoints (best, latest complete, or a given step)
# together with its byteresearch_meta.json, whose stored tag ids are re-verified on load.
# Both paths produce a Trained bundle with the model, tokenizer, tag ids, generation suppression
# ids, checkpoint meta and the architecture sizes interpretability code needs.
# prepare_for_interp sets eval mode, disables gradient checkpointing and TF32, enables the KV
# cache, casts to float32 and asserts gated-GELU FFNs with distinct act modules.
# Models load through AutoModelForSeq2SeqLM (mT5 keeps its untied LM head), and free_memory, release
# and loaded are the shared way to drop a model, while checkpoint_info describes a checkpoint.

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..core.config import ALL_LANGS, MODELS
from ..core.io import read_json, sha256_file
from ..core.paths import META_FILE, run_id
from .tags import suppress_ids, tag_ids


@dataclass
class Trained:
    # A loaded model bundle: model, tokenizer, {lang: tag id}, generation suppression ids,
    # checkpoint meta, architecture family, encoder/decoder layer counts, d_model and d_ff, plus
    # the registry key and condition when known.
    model: object
    tok: object
    tags: dict
    suppress: list
    meta: dict = field(default_factory=dict)
    arch: str | None = None
    n_enc: int = 0
    n_dec: int = 0
    d_model: int = 0
    d_ff: int = 0
    key: str | None = None
    condition: str | None = None

    @property
    def device(self) -> torch.device:
        # Output: the device holding the model parameters.
        return next(self.model.parameters()).device

    def tag(self, lang: str) -> int:
        # Input: language code. Output: its tag id; KeyError when the bundle has no tag for it.
        return self.tags[lang]

    def tags_for(self, langs) -> list:
        # Input: one language code or a per-row sequence of codes. Output: the per-row tag ids
        # (a single code yields a one-element list).
        return [self.tags[langs]] if isinstance(langs, str) else [self.tags[c] for c in langs]


def arch_of(model) -> str:
    # Input: a (M)T5ForConditionalGeneration. Output: "byt5" for the 384-token byte vocabulary,
    # otherwise "mt5".
    return "byt5" if model.config.vocab_size <= 384 else "mt5"


def make_trained(model, tok, tags=None, meta=None, arch=None, key=None, condition=None) -> Trained:
    # Inputs: model, tokenizer, optional tag dict (default: looked up for all study and control languages), meta,
    # arch, registry key and condition. Output: the Trained bundle with suppression ids and sizes.
    cfg = model.config
    return Trained(model=model, tok=tok, tags=dict(tags or tag_ids(tok, ALL_LANGS)),
                   suppress=suppress_ids(tok, cfg.vocab_size), meta=dict(meta or {}),
                   arch=arch or arch_of(model), n_enc=cfg.num_layers, n_dec=cfg.num_decoder_layers,
                   d_model=cfg.d_model, d_ff=cfg.d_ff, key=key, condition=condition)


def load_tokenizer(source):
    # Input: a registry key, Hugging Face id or local directory. Output: its tokenizer
    # (ByT5Tokenizer for ByT5, the T5 tokenizer for mT5).
    return AutoTokenizer.from_pretrained(MODELS[source].hf_id if source in MODELS else str(source))


def load_pretrained(model_key: str, dtype=torch.float32, device=None):
    # Inputs: registry model key, torch dtype and optional device. Output: (model in eval mode,
    # tokenizer) from the Hugging Face hub.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODELS[model_key].hf_id, dtype=dtype)
    if device is not None:
        model.to(device)
    return model.eval(), load_tokenizer(model_key)


def check_meta_tags(meta: dict, tok) -> dict:
    # Inputs: checkpoint meta and tokenizer. Output: the stored {lang: id} tags after verifying that
    # encoding each language's sentinel still yields the stored id; raises ValueError otherwise.
    stored = {k: int(v) for k, v in (meta.get("tags") or {}).items()}
    if not stored:
        raise ValueError("checkpoint meta has no tags")
    fresh = tag_ids(tok, list(stored))
    if fresh != stored:
        raise ValueError(f"tag ids in meta {stored} differ from tokenizer lookup {fresh}")
    return stored


def load_trained(ctx, model_key: str, condition: str, which="best", dtype=torch.float32, device=None,
                 lr=None, for_interp: bool = True) -> Trained:
    # Inputs: Ctx, model key, condition, which checkpoint ("best", "latest" or a step), dtype,
    # device (default ctx.device), optional sweep lr and whether to apply prepare_for_interp.
    # Output: the Trained bundle with tags read from byteresearch_meta.json.
    ckpt = ctx.paths.checkpoint_for(run_id(model_key, condition, lr), which)
    meta = read_json(ckpt / META_FILE)
    if meta is None:
        raise FileNotFoundError(f"missing or unreadable {ckpt / META_FILE}")
    tok = load_tokenizer(ckpt if (ckpt / "tokenizer_config.json").exists() else model_key)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt), dtype=dtype)
    model.to(device or ctx.device)
    model = prepare_for_interp(model, float32=dtype == torch.float32) if for_interp else model.eval()
    return make_trained(model, tok, check_meta_tags(meta, tok), meta, MODELS[model_key].arch, model_key, condition)


def gated_ffns(model) -> list:
    # Input: a (M)T5ForConditionalGeneration. Output: every block's FFN module, encoder blocks first.
    stacks = (model.encoder.block, model.decoder.block)
    return [blk.layer[-1].DenseReluDense for stack in stacks for blk in stack]


def prepare_for_interp(model, float32: bool = True):
    # Inputs: a (M)T5ForConditionalGeneration and whether to cast to float32. Puts it in eval mode,
    # disables gradient checkpointing and TF32, enables the KV cache, and asserts every FFN is a
    # gated-act FFN (wi_0, wi_1, wo, act) with its own act module instance. Output: the same model.
    model.eval()
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    if float32:
        model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ffns = gated_ffns(model)
    if not model.config.is_gated_act or not all(hasattr(f, "wi_0") and hasattr(f, "wi_1") for f in ffns):
        raise AssertionError("expected gated-gelu FFNs (wi_0, wi_1, act) in every block")
    if len({id(f.act) for f in ffns}) != len(ffns):
        raise AssertionError("FFN act modules are shared between layers; per-layer hooks would be ambiguous")
    return model


def free_memory() -> None:
    # Runs the garbage collector and empties the CUDA cache, so dropped models release their memory
    # (e.g. before a metrics subprocess needs the GPU).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release(trained) -> None:
    # Input: a Trained bundle (or None). Drops its model and frees memory. No output.
    if trained is not None:
        trained.model = None
    free_memory()


@contextlib.contextmanager
def loaded(ctx, model_key: str, condition: str, **kwargs):
    # Inputs: Ctx, model key, condition and load_trained keyword arguments. Yields the fine-tuned
    # Trained bundle and releases its model on exit, even after an error.
    trained = load_trained(ctx, model_key, condition, **kwargs)
    try:
        yield trained
    finally:
        release(trained)


def checkpoint_info(ctx, model_key: str, condition: str, which="best") -> dict:
    # Inputs: Ctx, model key, condition and which checkpoint. Output: {"path", "which", "step", "lr",
    # "meta_sha256"} from its byteresearch_meta.json (step/lr/hash None when the meta is missing).
    ckpt = ctx.paths.checkpoint_for(run_id(model_key, condition), which)
    meta_path = ckpt / META_FILE
    meta = read_json(meta_path) or {}
    return {"path": ctx.paths.relative(ckpt), "which": which, "step": meta.get("step"), "lr": meta.get("lr"),
            "meta_sha256": sha256_file(meta_path) if meta_path.exists() else None}
