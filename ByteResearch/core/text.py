# Text utilities shared by data filtering, formatting and evaluation.
# normalize_for_match gives the canonical form used for FLORES/NTREX exclusion (NFKC, casefold,
# collapsed whitespace), and truncate_chars implements char-first truncation by code points.
# script_ratio measures which fraction of the letters and marks in a string fall inside the
# Unicode ranges of one script, taken from core.config.SCRIPT_RANGES.
# split_sentences cuts paragraphs on the sentence terminators listed in core.config and on
# newlines, and length_ok applies the character-length window.
# Everything here is pure Python with no model or tokenizer dependency.

from __future__ import annotations

import bisect
import re
import unicodedata

from .config import SCRIPT_ALIASES, SCRIPT_RANGES, SENTENCE_TERMINATORS, UNSPACED_TERMINATORS

_STARTS = {k: [a for a, _ in v] for k, v in SCRIPT_RANGES.items()}
SCRIPTS = tuple(SCRIPT_RANGES) + tuple(SCRIPT_ALIASES)

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(rf"(?<=[{re.escape(SENTENCE_TERMINATORS)}])\s+|(?<=[{re.escape(UNSPACED_TERMINATORS)}])"
                         r"|\s*\n+\s*")
_SPECIAL_MARKUP = re.compile(r"<extra_id_\d+>|</s>|<pad>|<unk>")


def normalize_for_match(text: str) -> str:
    # Input: any string. Output: NFKC-normalised, casefolded text with whitespace runs collapsed
    # to single spaces and the ends stripped (the key for exclusion matching).
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def truncate_chars(text: str, max_chars) -> str:
    # Inputs: a string and a code-point cap (None = no cap). Output: the first max_chars code points.
    return text if max_chars is None else text[: int(max_chars)]


def _in_script(cp: int, script: str) -> bool:
    # Inputs: a code point and a canonical script key. Output: whether it lies in the script's ranges.
    ranges = SCRIPT_RANGES[script]
    i = bisect.bisect_right(_STARTS[script], cp) - 1
    return i >= 0 and cp <= ranges[i][1]


def canonical_script(script: str) -> str:
    # Input: an ISO 15924 script code. Output: its canonical key (SCRIPT_ALIASES applied, e.g. Hans -> Hani).
    return SCRIPT_ALIASES.get(script, script)


def script_ratio(text: str, script: str) -> float:
    # Inputs: a string and an ISO 15924 script code with an entry in SCRIPT_RANGES (or an alias in
    # SCRIPT_ALIASES). Output: the fraction of letter and mark characters that belong to that script
    # (0.0 when the text has no letters or marks); an unknown script raises KeyError.
    key = canonical_script(script)
    if key not in SCRIPT_RANGES:
        raise KeyError(f"unsupported script {script!r}")
    letters = [ord(ch) for ch in text if unicodedata.category(ch)[0] in "LM"]
    if not letters:
        return 0.0
    return sum(_in_script(cp, key) for cp in letters) / len(letters)


def split_sentences(text: str) -> list:
    # Input: a paragraph or document. Output: stripped, non-empty sentences, split after a
    # SENTENCE_TERMINATORS character followed by whitespace, after an UNSPACED_TERMINATORS character
    # unconditionally, and on newlines.
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def length_ok(text: str, min_chars: int, max_chars: int) -> bool:
    # Inputs: a string and an inclusive character window. Output: whether the stripped text's
    # code-point length lies within [min_chars, max_chars].
    return min_chars <= len(text.strip()) <= max_chars


def has_special_markup(text: str) -> bool:
    # Input: a string. Output: whether it contains literal tokenizer markup (<extra_id_N>, </s>,
    # <pad>, <unk>), which the mT5 tokenizer would map to special ids even with split_special_tokens.
    return bool(_SPECIAL_MARKUP.search(text))
