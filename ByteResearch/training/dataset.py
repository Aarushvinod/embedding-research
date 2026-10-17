# Training and dev data for one multilingual (architecture, size, condition) model.
# The training set concatenates the condition hub's pair files for every target in registry order
# (only the smoke targets under --smoke); each example keeps its hub text, target text, target
# index and a stable example index, so batches are formed from raw text and tokenized in the collator.
# The dev set is the fixed FLORES-dev subset from the corpora package (the same ids for every model),
# laid out as (target, source, reference) items for greedy chrF++ evaluation.
# The thin corpora access points below are the only places that touch the data agent's modules,
# and tests replace them with synthetic data.
# data_requires names the data stages (FLORES and NLLB) the training stages depend on.

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from ..core.config import CONDITIONS, TARGETS, resolve_langs
from ..core.registry import is_known

DATA_STAGES = ("data.flores", "data.nllb")


@dataclass(frozen=True)
class DevItem:
    # One dev row: target language code, hub-language source text and target reference.
    target: str
    src: str
    ref: str


class PairDataset(Dataset):
    # Map-style dataset over (src, tgt, target_idx) tuples. Items are dicts with the raw texts, the
    # target index (position in the run's target list) and the example index (row position).

    def __init__(self, rows):
        # Input: a list of (src, tgt, target_idx) tuples.
        self.rows = list(rows)

    def __len__(self) -> int:
        # Output: number of examples.
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        # Input: example index. Output: {"src", "tgt", "target_idx", "example_idx"}.
        src, tgt, t = self.rows[i]
        return {"src": src, "tgt": tgt, "target_idx": int(t), "example_idx": int(i)}


def load_pairs(ctx, hub: str, x: str) -> list:
    # Inputs: Ctx, hub and target codes. Output: corpora.nllb.load_pairs rows (dicts with "hub" and
    # "x" texts).
    from ..corpora import nllb
    return nllb.load_pairs(ctx, hub, x)


def load_flores(ctx, split: str, langs) -> dict:
    # Inputs: Ctx, FLORES split and language codes. Output: {code: [sentences]} from corpora.flores.
    from ..corpora import flores
    return flores.load_flores(ctx, split, langs)


def dev_subset_ids(ctx, n_per_target: int) -> list:
    # Inputs: Ctx and rows per target. Output: the fixed FLORES-dev row ids from corpora.flores.
    from ..corpora import flores
    return flores.dev_subset_ids(ctx, n_per_target=n_per_target)


def data_hash(ctx, condition: str) -> str:
    # Inputs: Ctx and condition key. Output: the corpora manifest hash of that condition's data.
    from ..corpora import manifest
    return manifest.data_hash_for_condition(ctx, condition)


def condition_targets(ctx, condition: str) -> list:
    # Inputs: Ctx and condition key. Output: the condition's targets in registry order, restricted to
    # the smoke targets under --smoke.
    allowed = resolve_langs("all", TARGETS, ctx.smoke)
    return [t for t in CONDITIONS[condition].targets if t in allowed]


def train_rows(ctx, condition: str, targets) -> list:
    # Inputs: Ctx, condition and its target list. Output: (hub text, target text, target index)
    # tuples, the hub's pair files concatenated in target order.
    hub = CONDITIONS[condition].hub
    rows = []
    for t, x in enumerate(targets):
        rows.extend((str(p["hub"]), str(p["x"]), t) for p in load_pairs(ctx, hub, x))
    return rows


def trim_to_batches(rows: list, global_batch: int) -> list:
    # Inputs: training rows and the global batch. Output: the rows minus the tail beyond the last full
    # global batch, so every epoch is a whole number of equal optimizer steps on any rank layout.
    return rows[: len(rows) // int(global_batch) * int(global_batch)]


def _ids_per_target(ids: list, targets, n: int) -> dict:
    # Inputs: dev ids from the corpora package, targets and rows per target. Output: {target: ids};
    # n ids are shared by all targets, n x 8 ids are split per target in registry order.
    if len(ids) == n:
        return {t: list(ids) for t in targets}
    if len(ids) == n * len(TARGETS):
        return {t: list(ids[TARGETS.index(t) * n:(TARGETS.index(t) + 1) * n]) for t in targets}
    raise ValueError(f"dev_subset_ids returned {len(ids)} ids; expected {n} or {n * len(TARGETS)}")


def dev_items(ctx, condition: str, targets, n_per_target: int) -> list:
    # Inputs: Ctx, condition, targets and rows per target. Output: DevItems grouped by target in
    # target order, sources in the condition's hub language.
    hub = CONDITIONS[condition].hub
    text = load_flores(ctx, "dev", [hub, *targets])
    per = _ids_per_target(list(dev_subset_ids(ctx, n_per_target)), targets, n_per_target)
    return [DevItem(t, text[hub][i], text[t][i]) for t in targets for i in per[t]]


def data_requires() -> tuple:
    # Output: the registered data stages training needs (FLORES for the dev subset, NLLB pairs for the
    # condition; corpora is discovered before training), so training also resolves in isolation.
    return tuple(name for name in DATA_STAGES if is_known(name))
