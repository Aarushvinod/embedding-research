# The Seq2SeqTrainer subclass used for every ByteResearch fine-tuning run.
# Its loss is a per-sequence average: float32 token cross-entropy, summed per row, divided by that
# row's own label count and averaged over rows, so long byte sequences are not up-weighted.
# Every training log entry gains per-target losses, a digest of the exact examples seen per step,
# examples seen, throughput, peak memory and timestamps, aggregated over all DDP ranks.
# Evaluation is replaced by greedy decoding of a fixed FLORES-dev subset through the shared
# interp.scoring generation path, sharded over ranks and gathered so every rank gets identical
# macro and per-target chrF++, all inside an RNG-preserving block so resume stays exact.
# VOLATILE_KEYS lists the timing and memory fields that legitimately differ between runs.

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

from ..core.config import DecodingConfig
from ..core.dist import gather_objects, world
from ..core.seeding import rng_preserved
from ..interp.scoring import Batching, generate
from ..modeling.encode import IGNORE
from ..modeling.loading import make_trained
from ..stats.bootstrap import chrf_metric

VOLATILE_KEYS = frozenset({
    "time", "interval_s", "examples_per_s", "peak_mem_gb", "eval_runtime", "train_runtime",
    "train_samples_per_second", "train_steps_per_second", "train_loss",
})
MODEL_KEYS = ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")


def per_sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple:
    # Inputs: logits [B, T, V] and labels [B, T] (-100 ignored). Output: (scalar loss, per-row loss
    # [B]); token CE in float32, summed per row and divided by that row's label count, mean over rows.
    tok = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                          ignore_index=IGNORE, reduction="none").view(labels.shape)
    mask = labels.ne(IGNORE)
    per_row = (tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_row.mean(), per_row


def dev_chrf(items, hyps, targets) -> dict:
    # Inputs: DevItems, hypotheses aligned to them and the target order. Output: {"chrf_pp": macro
    # mean over targets, "chrf_pp_<t>": corpus chrF++ of target t}.
    metric = chrf_metric()
    per = {}
    for t in targets:
        pairs = [(h, it.ref) for it, h in zip(items, hyps) if it.target == t]
        per[t] = metric.corpus_score([p[0] for p in pairs], [[p[1] for p in pairs]]).score
    return {"chrf_pp": sum(per.values()) / len(per), **{f"chrf_pp_{t}": v for t, v in per.items()}}


def examples_digest(steps: dict) -> str:
    # Input: {global step: example ids}. Output: 16 hex characters of the SHA-1 of the per-step sorted
    # id lists in step order, a fingerprint of the exact data order that is invariant to rank layout.
    blob = json.dumps([[s, sorted(steps[s])] for s in sorted(steps)])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class ByteTrainer(Seq2SeqTrainer):
    # Seq2SeqTrainer with per-sequence loss, enriched logs and greedy dev chrF++ evaluation. The
    # eval_dataset is the list of DevItems. Extra keyword arguments: targets (codes by target index),
    # tags ({lang: id}), arch, global_batch and gen_batching (Batching for dev generation).

    def __init__(self, *args, targets=(), tags=None, arch=None, global_batch=1,
                 gen_batching=Batching(max_batch=32), **kwargs):
        # Inputs: the usual Seq2SeqTrainer arguments plus the extras above. Sets
        # model_accepts_loss_kwargs=False so the loss is never rescaled by token counts.
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.dev, self.targets, self.tags, self.arch = list(self.eval_dataset), list(targets), dict(tags or {}), arch
        self.global_batch, self.gen_batching = int(global_batch), gen_batching
        self.decoding = dataclasses.replace(DecodingConfig(), num_beams=1)
        self._bundle = None
        self._reset_window()
        self._clock = {"log": time.time(), "train": time.time(), "eval_s": 0.0}

    def _reset_window(self) -> None:
        # Clears the per-target loss sums/counts and the per-step example ids of the logging window.
        self._sums = torch.zeros(max(len(self.targets), 1), dtype=torch.float64)
        self._counts = torch.zeros_like(self._sums)
        self._ids = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Inputs: model, a collated batch, whether to return outputs, and the (unused) token count.
        # Output: the per-sequence loss (and model outputs); in training mode also accumulates the
        # per-target loss sums and the example ids of the current step.
        out = model(**{k: inputs[k] for k in MODEL_KEYS}, use_cache=False)
        loss, per_row = per_sequence_loss(out.logits, inputs["labels"])
        if model.training:
            idx = inputs["target_idx"].to(per_row.device)
            self._sums = self._sums.to(per_row.device).index_add_(0, idx, per_row.detach().double())
            self._counts = self._counts.to(per_row.device).index_add_(0, idx, torch.ones_like(per_row).double())
            self._ids[self.state.global_step + 1].extend(inputs["example_idx"])
        return (loss, out) if return_outputs else loss

    def train(self, *args, **kwargs):
        # Inputs: Trainer.train arguments. Starts this session's clocks, then trains. Output: TrainOutput.
        now = time.time()
        self._clock = {"log": now, "train": now, "eval_s": 0.0}
        return super().train(*args, **kwargs)

    def _window_stats(self) -> dict:
        # Output: log fields of the window since the last training log, gathered over ranks: loss per
        # target, examples digest, examples seen, examples/s (eval time excluded) and peak memory.
        parts = gather_objects((self._sums.tolist(), self._counts.tolist(), dict(self._ids)))
        sums = [sum(p[0][i] for p in parts) for i in range(len(self._sums))]
        counts = [sum(p[1][i] for p in parts) for i in range(len(self._counts))]
        steps = defaultdict(list)
        for p in parts:
            for s, ids in p[2].items():
                steps[s].extend(ids)
        self._reset_window()
        busy = max(time.time() - self._clock["train"] - self._clock["eval_s"], 1e-9)
        stats = {f"loss_{t}": s / c for t, s, c in zip(self.targets, sums, counts) if c > 0}
        stats.update(examples_digest=examples_digest(steps), examples_seen=self.state.global_step * self.global_batch,
                     examples_per_s=len(steps) * self.global_batch / busy)
        if torch.cuda.is_available() and self.args.device.type == "cuda":
            stats["peak_mem_gb"] = torch.cuda.max_memory_allocated(self.args.device) / 2 ** 30
            torch.cuda.reset_peak_memory_stats(self.args.device)
        return stats

    def log(self, logs: dict, start_time=None) -> None:
        # Inputs: the log dict and optional start time. Adds the window statistics to training logs
        # and a timestamp plus the seconds since the previous log to every entry, then logs.
        now = time.time()
        if "loss" in logs:
            logs.update(self._window_stats())
            self._clock.update(train=now, eval_s=0.0)
        logs.update(time=round(now, 3), interval_s=now - self._clock["log"])
        self._clock["log"] = now
        super().log(logs, start_time)

    def bundle(self):
        # Output: the cached Trained bundle around the live (unwrapped) model, used for dev decoding
        # with the run's tags and generation suppression ids.
        if self._bundle is None:
            model = self.accelerator.unwrap_model(self.model)
            self._bundle = make_trained(model, self.processing_class, self.tags, arch=self.arch)
        return self._bundle

    def dev_scores(self) -> dict:
        # Output: dev chrF++ (macro and per target). Each rank greedily decodes its shard of the dev
        # items through interp.scoring.generate, the shards are gathered, and every rank scores all.
        w = world()
        rows = list(range(w.rank, len(self.dev), w.size))
        model = self.bundle().model
        was_training = model.training
        model.eval()
        hyps = generate(self.bundle(), [d.src for d in self.dev], [d.target for d in self.dev], self.decoding,
                        self.gen_batching, rows=rows) if rows else []
        model.train(was_training)
        merged = dict(pair for part in gather_objects(list(zip(rows, hyps))) for pair in part)
        return dev_chrf(self.dev, [merged[i] for i in range(len(self.dev))], self.targets)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval", **gen_kwargs) -> dict:
        # Inputs: Trainer.evaluate arguments (only the prefix is used). Output: the prefixed dev
        # metrics plus runtime; logs them and fires on_evaluate, leaving every RNG stream untouched.
        t0 = time.time()
        with rng_preserved(), self.accelerator.autocast():
            scores = self.dev_scores()
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in scores.items()}
        metrics[f"{metric_key_prefix}_runtime"] = time.time() - t0
        self._clock["eval_s"] += metrics[f"{metric_key_prefix}_runtime"]
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
