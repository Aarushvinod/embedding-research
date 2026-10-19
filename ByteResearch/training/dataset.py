# Training and dev data for one per-direction (model, condition, target) run.
# The training set is the single NLLB pair file of the condition's hub and that target; each example keeps
# its hub text, target text and a stable example index, so batches are formed from raw text and tokenized
# in the collator.
# The dev set is the fixed seeded FLORES-dev subset from the corpora package (the same ids for every model
# and direction), laid out as (target, source, reference) items for greedy chrF++ evaluation.
# The thin corpora access points below are the only places that touch the data agent's modules,
# and tests replace them with synthetic data.
# data_requires names the data stages (FLORES and NLLB) the training stages depend on.

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from ..core.config import CONDITIONS
from ..core.registry import is_known

DATA_STAGES = ("data.flores", "data.nllb")


@dataclass(frozen=True)
class DevItem:
    # One dev row: target language code, hub-language source text and target reference.
    target: str
    src: str
    ref: str


class PairDataset(Dataset):
    # Map-style dataset over (src, tgt) tuples. Items are dicts with the raw texts and the example index
    # (row position).

    def __init__(self, rows):
        # Input: a list of (src, tgt) tuples.
        self.rows = list(rows)

    def __len__(self) -> int:
        # Output: number of examples.
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        # Input: example index. Output: {"src", "tgt", "example_idx"}.
        src, tgt = self.rows[i]
        return {"src": src, "tgt": tgt, "example_idx": int(i)}


def load_pairs(ctx, hub: str, x: str) -> list:
    # Inputs: Ctx, hub and target codes. Output: corpora.nllb.load_pairs rows (dicts with "hub" and
    # "x" texts).
    from ..corpora import nllb
    return nllb.load_pairs(ctx, hub, x)


def load_flores(ctx, split: str, langs) -> dict:
    # Inputs: Ctx, FLORES split and language codes. Output: {code: [sentences]} from corpora.flores.
    from ..corpora import flores
    return flores.load_flores(ctx, split, langs)


def dev_subset_ids(ctx, n: int) -> list:
    # Inputs: Ctx and the subset size. Output: the fixed FLORES-dev row ids from corpora.flores.
    from ..corpora import flores
    return flores.dev_subset_ids(ctx, n_per_target=n)


def data_hash(ctx, condition: str, target: str) -> str:
    # Inputs: Ctx, condition key and target code. Output: the corpora manifest hash of that direction's data.
    from ..corpora import manifest
    return manifest.data_hash_for_direction(ctx, condition, target)


def direction_hub(condition: str, target: str) -> str:
    # Inputs: condition key and target code. Output: the condition's hub; raises ValueError when the target
    # is not one of the condition's targets.
    if target not in CONDITIONS[condition].targets:
        raise ValueError(f"{target} is not a target of {condition}")
    return CONDITIONS[condition].hub


def train_rows(ctx, condition: str, target: str) -> list:
    # Inputs: Ctx, condition and target. Output: (hub text, target text) tuples from the single pair file
    # of the condition's hub and the target.
    return [(str(p["hub"]), str(p["x"])) for p in load_pairs(ctx, direction_hub(condition, target), target)]


def trim_to_batches(rows: list, global_batch: int) -> list:
    # Inputs: training rows and the global batch. Output: the rows minus the tail beyond the last full
    # global batch, so every epoch is a whole number of equal optimizer steps on any rank layout.
    return rows[: len(rows) // int(global_batch) * int(global_batch)]


def dev_items(ctx, condition: str, target: str, n: int) -> list:
    # Inputs: Ctx, condition, target and subset size. Output: the n DevItems of the fixed FLORES-dev subset,
    # sources in the condition's hub language and references in the target.
    hub = direction_hub(condition, target)
    text = load_flores(ctx, "dev", [hub, target])
    ids = list(dev_subset_ids(ctx, n))
    if len(ids) != int(n):
        raise ValueError(f"dev_subset_ids returned {len(ids)} ids; expected {n}")
    return [DevItem(target, text[hub][i], text[target][i]) for i in ids]


def data_requires() -> tuple:
    # Output: the registered data stages training needs (FLORES for the dev subset, NLLB pairs for the
    # direction; corpora is discovered before training), so training also resolves in isolation.
    return tuple(name for name in DATA_STAGES if is_known(name))
