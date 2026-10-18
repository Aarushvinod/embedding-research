# Language identification with GlotLID, shared by the NLLB and Wikipedia filters and the DecoderLens experiment.
# load_glotlid downloads cis-lmu/glotlid model.bin into ROOT/data/models/glotlid, records the resolved
# revision in its manifest, verifies its SHA-256 and loads it with fasttext.
# predict calls model.f.predict directly, because fasttext's Python wrapper breaks under numpy 2.
# Every language is identified by exactly one GlotLID label, the one of its FLORES variety (e.g. arb_Arab for
# Modern Standard Arabic, swh_Latn for Swahili), except LID_LABEL_OVERRIDES (fil_Latn for Tagalog, fas_Arab for
# Persian, ekk_Latn for Estonian);
# lid_labels checks that every needed label exists in the model.
# A text counts as language L when L's label is GlotLID's top label with a probability of at least L's
# threshold (lid_ok); own_top_prob gives that probability (0 when another label is on top) and label_prob
# reads L's probability from a ranked prediction even when it is not on top (used to calibrate thresholds).

from __future__ import annotations

import functools

from ..core.config import LID_LABEL_OVERRIDES, LID_LANGS, PINS, language
from .assets import asset_dir, build_hf_asset, ensure_hf_asset
from .manifest import verified_path

LABEL_PREFIX = "__label__"
LID_RULE = "the language's FLORES-variety GlotLID label is the top label with probability >= the threshold"


def build_glotlid(ctx) -> dict:
    # Input: Ctx. Downloads GlotLID model.bin at the resolved (or pinned) revision. Output: the manifest.
    pin = PINS["glotlid"]
    return build_hf_asset(ctx, "glotlid", pin["repo"], files=[pin["file"]], revision=pin.get("revision"))


def glotlid_path(ctx):
    # Input: Ctx. Output: the verified local model.bin path, downloading it first when missing.
    pin = PINS["glotlid"]
    ensure_hf_asset(ctx, "glotlid", pin["repo"], files=[pin["file"]], revision=pin.get("revision"))
    return verified_path(asset_dir(ctx, "glotlid"), pin["file"])


@functools.lru_cache(maxsize=2)
def _fasttext_model(path: str):
    # Input: a fasttext model path. Output: the loaded fasttext model (cached per process).
    import fasttext
    return fasttext.load_model(path)


def load_glotlid(ctx):
    # Input: Ctx. Output: the GlotLID fasttext model (downloaded, SHA-checked and cached per process).
    return _fasttext_model(str(glotlid_path(ctx)))


def model_labels(model) -> list:
    # Input: a fasttext model. Output: its labels without the "__label__" prefix.
    return [label.removeprefix(LABEL_PREFIX) for label in model.get_labels()]


def lid_label(code: str) -> str:
    # Input: a study or candidate language code. Output: its single GlotLID label (the FLORES variety's code,
    # or the LID_LABEL_OVERRIDES entry).
    return LID_LABEL_OVERRIDES.get(code) or language(code).flores


def check_labels(labels) -> dict:
    # Input: the labels present in the GlotLID model. Output: {code: label} for every LID language; raises
    # ValueError naming every language whose label the model lacks.
    table = {code: lid_label(code) for code in LID_LANGS}
    missing = sorted(c for c, label in table.items() if label not in set(labels))
    if missing:
        raise ValueError(f"GlotLID has no label for {missing}; add them to LID_LABEL_OVERRIDES in core.config")
    return table


def lid_labels(ctx) -> dict:
    # Input: Ctx. Output: {code: GlotLID label} for every study and candidate language, checked against the model.
    return check_labels(model_labels(load_glotlid(ctx)))


def lid_record(langs) -> dict:
    # Input: language codes. Output: the LID rule for manifests and result configs: rule text, each language's
    # label and the overrides (the thresholds are recorded with each filter spec).
    return {"rule": LID_RULE, "labels": {c: lid_label(c) for c in langs},
            "overrides": dict(LID_LABEL_OVERRIDES)}


def predict(model, texts, k: int = 1) -> list:
    # Inputs: a fasttext model, a list of texts and the number of labels per text. Output: one
    # (label, prob) tuple per text when k == 1, else one list of k such tuples per text; labels have no
    # "__label__" prefix and newlines are replaced by spaces before prediction.
    out = []
    for text in texts:
        pairs = model.f.predict(text.replace("\n", " ").replace("\r", " "), k, 0.0, "replace")
        ranked = [(label.removeprefix(LABEL_PREFIX), float(prob)) for prob, label in pairs]
        out.append((ranked[0] if ranked else ("", 0.0)) if k == 1 else ranked)
    return out


def ranked_pairs(pred) -> list:
    # Input: one text's prediction, either a single (label, prob) pair (k == 1) or a ranked list of them.
    # Output: the ranked list of (label, prob) pairs.
    return [tuple(pred)] if pred and isinstance(pred[0], str) else [tuple(p) for p in pred]


def top_label(pred) -> tuple:
    # Input: one text's prediction (a pair or a ranked list). Output: the top (label, prob), or ("", 0.0).
    pairs = ranked_pairs(pred)
    return pairs[0] if pairs else ("", 0.0)


def own_top_prob(pred, label: str) -> float:
    # Inputs: one text's prediction and a language's label. Output: the label's probability when it is the
    # top label, else 0.0 (so `own_top_prob >= threshold` is the LID filter).
    top, prob = top_label(pred)
    return float(prob) if top == label else 0.0


def lid_ok(pred, label: str, threshold: float) -> bool:
    # Inputs: one text's prediction, a language's label and its threshold. Output: whether the label is the top
    # label with probability >= threshold (the LID filter).
    top, prob = top_label(pred)
    return top == label and prob >= threshold


def label_prob(pred, label: str) -> float:
    # Inputs: one text's ranked prediction and a language's label. Output: that label's probability (0.0 when it
    # is not among the predicted labels).
    return float(next((p for lab, p in ranked_pairs(pred) if lab == label), 0.0))
