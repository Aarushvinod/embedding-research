# The ordered filter chain that turns a score-sorted stream of mined pairs into training pairs.
# Filters run in a fixed order with one rejection counter each: malformed line, exact-pair duplicate,
# 20-300 characters on both sides, the x/hub character-length ratio band, literal tokenizer markup,
# FLORES/NTREX exclusion, script consistency per side, GlotLID per side (the language's label must be the top
# label with at least the side's minimum probability) and LaBSE cosine; the data stage supplies the thresholds.
# Score-order violations are counted but do not reject a line.
# For speed the cheap filters run line by line and their survivors are collected into blocks, on which LID
# and then LaBSE run in batches; lines are then tallied back in stream order.
# The result is therefore exactly the first `cap` survivors, and every counter covers precisely the lines
# read up to that point; LID and LaBSE are injected as functions, which keeps this module model-free.

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..core.text import has_special_markup, length_ok, normalize_for_match, script_ratio
from .lid import LID_RULE, lid_ok, own_top_prob, top_label

REASONS = ("malformed", "duplicate", "length", "ratio", "markup", "excluded", "script", "lid", "labse")
STEPS = ("malformed", "dedupe", "length", "ratio", "markup", "exclusion", "script", "lid", "labse")


@dataclass(frozen=True)
class FilterSpec:
    # Thresholds for one pair: hub and target codes and scripts, the inclusive character window, the x/hub
    # length-ratio band, each side's minimum script ratio, the GlotLID rule (each side's label, which must be
    # the top label with at least that side's minimum probability), the LaBSE minimum cosine and the exclusion
    # set. The defaults accept everything; the data stage fills in its thresholds.
    hub: str
    x: str
    hub_script: str
    x_script: str
    min_chars: int = 20
    max_chars: int = 512
    ratio_min: float = 0.0
    ratio_max: float = float("inf")
    hub_min_script: float = 0.0
    x_min_script: float = 0.0
    hub_lid_min: float = 0.0
    x_lid_min: float = 0.0
    lid_rule: str = LID_RULE
    hub_lid: str = ""
    x_lid: str = ""
    labse_min: float = 0.0
    exclusion: frozenset = frozenset()


@dataclass
class Entry:
    # One streamed line: its 1-based line number, the parsed (hub, x, score) or None, the rejection
    # reason (None while it survives), the top LID predictions, each side's own-label top probability (0 when
    # another label is on top) and the LaBSE cosine once computed.
    line_no: int
    parsed: tuple | None
    reason: str | None
    lid_hub: tuple | None = None
    lid_x: tuple | None = None
    lid_hub_p: float | None = None
    lid_x_p: float | None = None
    labse: float | None = None


@dataclass
class FilterState:
    # Running state of one filter pass: the cap, kept rows, rejection counters, lines tallied,
    # score-order violations, the previous score, the dedupe key set and the number of blocks scored.
    cap: int
    rows: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: dict.fromkeys(REASONS, 0))
    lines: int = 0
    violations: int = 0
    last_score: float | None = None
    seen: set = field(default_factory=set)
    blocks: int = 0

    def add(self, entry: Entry) -> bool:
        # Input: an Entry whose filters have all run. Tallies it (line count, score order, rejection
        # counter or kept row). Output: whether the cap has been reached.
        self.lines += 1
        score = entry.parsed[2] if entry.parsed else None
        self.violations += int(score is not None and self.last_score is not None and score > self.last_score)
        self.last_score = score if score is not None else self.last_score
        if entry.reason:
            self.counts[entry.reason] += 1
        else:
            self.rows.append(make_row(len(self.rows), entry))
        return len(self.rows) >= self.cap


def pair_key(hub: str, x: str) -> bytes:
    # Inputs: the two texts of a pair. Output: a 16-byte digest identifying the exact pair (for dedupe).
    return hashlib.blake2b(f"{hub}\t{x}".encode("utf-8"), digest_size=16).digest()


def cheap_reason(parsed, seen: set, spec: FilterSpec):
    # Inputs: a parsed (hub, x, score) or None, the dedupe key set (updated in place) and the spec.
    # Output: the first failing cheap filter's reason, or None when the pair survives them all.
    if parsed is None:
        return "malformed"
    hub, x, _ = parsed
    key = pair_key(hub, x)
    if key in seen:
        return "duplicate"
    seen.add(key)
    if not (length_ok(hub, spec.min_chars, spec.max_chars) and length_ok(x, spec.min_chars, spec.max_chars)):
        return "length"
    if not spec.ratio_min <= len(x) / len(hub) <= spec.ratio_max:
        return "ratio"
    if has_special_markup(hub) or has_special_markup(x):
        return "markup"
    if normalize_for_match(hub) in spec.exclusion or normalize_for_match(x) in spec.exclusion:
        return "excluded"
    if script_ratio(hub, spec.hub_script) < spec.hub_min_script or script_ratio(x, spec.x_script) < spec.x_min_script:
        return "script"
    return None


def score_block(cands: list, spec: FilterSpec, lid_fn, labse_fn) -> None:
    # Inputs: Entries that passed the cheap filters, the spec, lid_fn(texts) -> one prediction per text
    # (a (label, prob) pair or a ranked list) and labse_fn(left, right) -> cosines. Runs LID on both sides,
    # then LaBSE on the LID survivors, and records predictions, own-label top probabilities, cosines and
    # rejection reasons on the entries in place.
    if not cands:
        return
    hub_preds = lid_fn([e.parsed[0] for e in cands])
    x_preds = lid_fn([e.parsed[1] for e in cands])
    for e, ph, px in zip(cands, hub_preds, x_preds):
        e.lid_hub, e.lid_x = top_label(ph), top_label(px)
        e.lid_hub_p, e.lid_x_p = own_top_prob(ph, spec.hub_lid), own_top_prob(px, spec.x_lid)
        ok = lid_ok(ph, spec.hub_lid, spec.hub_lid_min) and lid_ok(px, spec.x_lid, spec.x_lid_min)
        e.reason = None if ok else "lid"
    passed = [e for e in cands if e.reason is None]
    cosines = labse_fn([e.parsed[0] for e in passed], [e.parsed[1] for e in passed]) if passed else []
    for e, cos in zip(passed, cosines):
        e.labse = float(cos)
        e.reason = None if e.labse >= spec.labse_min else "labse"


def make_row(i: int, e: Entry) -> dict:
    # Inputs: the output index and a surviving Entry. Output: the stored pair row
    # {i, line_no, score, hub, x, labse, lid_hub, lid_x, lid_hub_p, lid_x_p} (top label and probability
    # per side, and each side's own-label top probability).
    hub, x, score = e.parsed
    return {"i": i, "line_no": e.line_no, "score": score, "hub": hub, "x": x, "labse": e.labse,
            "lid_hub": list(e.lid_hub), "lid_x": list(e.lid_x), "lid_hub_p": e.lid_hub_p, "lid_x_p": e.lid_x_p}


def flush_block(pending: list, state: FilterState, spec: FilterSpec, lid_fn, labse_fn) -> bool:
    # Inputs: the block's Entries in stream order, the state, spec and model functions. Scores the
    # cheap-filter survivors, then tallies every entry in order until the cap is hit. Output: whether
    # the cap was reached.
    score_block([e for e in pending if e.reason is None], spec, lid_fn, labse_fn)
    state.blocks += 1
    for e in pending:
        if state.add(e):
            return True
    return False


def run_filters(entries, spec: FilterSpec, lid_fn, labse_fn, cap: int, block: int = 5000, progress=None) -> dict:
    # Inputs: an iterable of (line_no, parsed-or-None) in stream order, the spec, the LID and LaBSE
    # functions, the number of pairs to keep, the candidate block size and an optional
    # progress(state) callback called after each block. Output: {rows, counts, lines_read,
    # order_violations, exhausted, blocks}, where exhausted means the input ended before the cap.
    state = FilterState(cap=cap)
    pending, n_cand, full = [], 0, False
    for line_no, parsed in entries:
        entry = Entry(line_no, parsed, cheap_reason(parsed, state.seen, spec))
        pending.append(entry)
        n_cand += entry.reason is None
        if n_cand >= block:
            full = flush_block(pending, state, spec, lid_fn, labse_fn)
            pending, n_cand = [], 0
            _report(progress, state)
        if full:
            break
    if not full:
        full = flush_block(pending, state, spec, lid_fn, labse_fn)
        _report(progress, state)
    return {"rows": state.rows, "counts": dict(state.counts), "lines_read": state.lines,
            "order_violations": state.violations, "exhausted": not full, "blocks": state.blocks}


def _report(progress, state: FilterState) -> None:
    # Inputs: an optional progress callback and the state. Calls the callback when given.
    if progress is not None:
        progress(state)


def survival(counts: dict, lines_read: int) -> dict:
    # Inputs: rejection counters and lines read. Output: {"read": n, <step>: remaining after that filter
    # for each step in order, "kept": final count}; "kept" equals the remainder after the last filter.
    out, left = {"read": lines_read}, lines_read
    for step, reason in zip(STEPS, REASONS):
        left -= counts.get(reason, 0)
        out[step] = left
    out["kept"] = left
    return out
