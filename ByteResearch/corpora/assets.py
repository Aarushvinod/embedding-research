# Auxiliary model assets cached under ROOT/data: Hugging Face model snapshots, LaBSE and the sacreBLEU SPM model.
# build_hf_asset downloads a repository, resolved to an exact commit, into ROOT/data/models/<name>.
# Its manifest records that revision and the hash of every file, and ensure_hf_asset reuses a complete one.
# LaBSE (sentence-transformers/LaBSE, safetensors weights only) scores pair similarity as the dot product of
# normalized sentence embeddings, batched on the GPU when one is available, in fp32 or (--labse-dtype) fp16.
# use_sacrebleu_cache points sacreBLEU at ROOT/data/sacrebleu, also when sacreBLEU is already imported.
# build_sacrebleu instantiates BLEU(tokenize="flores200") once so that its SPM model is cached there.

from __future__ import annotations

import functools
import os
from pathlib import Path, PurePosixPath

import numpy as np

from ..core.config import PINS, DataConfig
from .http import hf_fetch
from .manifest import begin_build, finish_build, manifest_complete, read_manifest, verified_path

LABSE_PATTERNS = ("*.json", "*.txt", "model.safetensors", "2_Dense/model.safetensors")
LABSE_DTYPES = ("fp32", "fp16")


def asset_dir(ctx, name: str) -> Path:
    # Inputs: Ctx and an asset name. Output: ROOT/data/models/<name>.
    return ctx.paths.models_dir() / name


def build_hf_asset(ctx, name: str, repo: str, files=None, patterns=None, revision=None) -> dict:
    # Inputs: Ctx, asset name, repository id, explicit files or allow-patterns and an optional pinned
    # revision. Downloads the files into ROOT/data/models/<name> and writes the manifest (refusing to
    # overwrite a complete one without --rebuild-data). Output: the manifest.
    out = begin_build(ctx, asset_dir(ctx, name))
    info = hf_fetch(repo, out, files=files, patterns=patterns, revision=revision)
    config = {"repo": repo, "files": files, "patterns": patterns, "pinned_revision": revision}
    return finish_build(ctx, out, sources=[info], config=config, outputs=info["files"])


def ensure_hf_asset(ctx, name: str, repo: str, files=None, patterns=None, revision=None) -> Path:
    # Inputs: as build_hf_asset. Output: the asset directory, downloading it first when it has no
    # complete manifest.
    if not manifest_complete(asset_dir(ctx, name)):
        build_hf_asset(ctx, name, repo, files=files, patterns=patterns, revision=revision)
    return asset_dir(ctx, name)


def verify_asset(ctx, name: str) -> Path:
    # Inputs: Ctx and an asset name. Output: the asset directory after checking every recorded output's
    # SHA-256 (once per process and file version).
    directory = asset_dir(ctx, name)
    for output in read_manifest(directory)["outputs"]:
        verified_path(directory, output)
    return directory


def build_labse(ctx) -> dict:
    # Input: Ctx. Downloads the LaBSE sentence-transformers snapshot (safetensors only). Output: manifest.
    return build_hf_asset(ctx, "labse", PINS["labse"]["model"], patterns=list(LABSE_PATTERNS),
                          revision=PINS["labse"].get("revision"))


@functools.lru_cache(maxsize=2)
def _sentence_transformer(path: str, device: str, dtype: str = "fp32"):
    # Inputs: a local model directory, a torch device string and "fp32" or "fp16". Output: the loaded
    # SentenceTransformer in eval mode, cast to half precision for fp16 (cached per process).
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(path, device=device).eval()
    return model.half() if dtype == "fp16" else model


def labse_dtype(ctx) -> str:
    # Input: Ctx. Output: the LaBSE inference dtype, --labse-dtype or DataConfig.labse_dtype ("fp32").
    dtype = ctx.opt("labse_dtype") or ctx.config(DataConfig).labse_dtype
    if dtype not in LABSE_DTYPES:
        raise ValueError(f"labse dtype must be one of {LABSE_DTYPES}, got {dtype!r}")
    return dtype


def load_labse(ctx):
    # Input: Ctx. Output: the LaBSE SentenceTransformer on ctx.device in the selected dtype, downloaded
    # and verified first.
    ensure_hf_asset(ctx, "labse", PINS["labse"]["model"], patterns=list(LABSE_PATTERNS),
                    revision=PINS["labse"].get("revision"))
    return _sentence_transformer(str(verify_asset(ctx, "labse")), ctx.device, labse_dtype(ctx))


def labse_cosines(model, left, right, batch_size: int = 256) -> np.ndarray:
    # Inputs: a SentenceTransformer, two equal-length text lists and the batch size. Output: float32
    # cosine similarities of the aligned pairs (dot products of normalized embeddings).
    if not left:
        return np.zeros(0, dtype=np.float32)
    enc = functools.partial(model.encode, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    return (enc(list(left)) * enc(list(right))).sum(axis=1).astype(np.float32)


def use_sacrebleu_cache(ctx) -> Path:
    # Input: Ctx. Sets $SACREBLEU and the already-imported sacreBLEU module constants to
    # ROOT/data/sacrebleu so SPM models are read from and saved there. Output: that directory.
    directory = ctx.paths.sacrebleu_dir()
    os.environ["SACREBLEU"] = str(directory)
    import sacrebleu.tokenizers.tokenizer_spm as spm
    import sacrebleu.utils as utils
    utils.SACREBLEU_DIR = str(directory)
    spm.SACREBLEU_DIR = str(directory)
    return directory


def build_sacrebleu(ctx) -> dict:
    # Input: Ctx. Instantiates BLEU(tokenize="flores200") once with the cache redirected, so the SPM model
    # lands in ROOT/data/sacrebleu/models, and records its SHA-256 in a manifest. Output: the manifest.
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from sacrebleu.tokenizers.tokenizer_spm import SPM_MODELS
    out = begin_build(ctx, ctx.paths.sacrebleu_dir())
    use_sacrebleu_cache(ctx)
    BLEU(tokenize=PINS["sacrebleu"]["bleu_tokenize"])
    url = SPM_MODELS["flores200"]["url"]
    name = f"models/{PurePosixPath(url).name}"
    source = {"url": url, "sacrebleu": sacrebleu.__version__, "tokenize": PINS["sacrebleu"]["bleu_tokenize"]}
    return finish_build(ctx, out, sources=[source], config=PINS["sacrebleu"], outputs=[name])
