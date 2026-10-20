# The Seq2SeqTrainer subclass used for every ByteResearch fine-tuning run (one hub-to-target direction).
# Its loss is the standard token-level cross-entropy: summed over every non-ignored label token in float32 and
# divided by the label-token count of the whole global batch, which the Trainer counts across gradient-
# accumulation micro-batches and, with average_tokens_across_devices, all-reduces over DDP ranks.
# The Trainer then scales each rank's loss by the world size (DDP averages gradients) and does not divide by
# the accumulation steps again, so any rank x accumulation layout gives the model's own mean loss over the
# global batch, which a plain per-micro-batch mean (T5's built-in loss under the Trainer) would not.
# Training logs gain a digest of the exact examples seen per step, examples seen, throughput, peak memory
# and timestamps, aggregated over all DDP ranks, and VOLATILE_KEYS lists the fields that differ between runs.
# Evaluation is replaced by greedy decoding of a fixed FLORES-dev subset from the [start] prefix through the
# shared interp.scoring generation path, sharded over ranks and gathered so every rank gets the same chrF++,
# all inside an RNG-preserving block so resume stays exact.
# Only evaluations at checkpoint steps can become the best model, so the best step always has a saved
# checkpoint while dev chrF++ (and early stopping) still runs at every evaluation.

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


def token_loss(outputs, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
    # Inputs: model outputs with logits [B, T, V], labels [B, T] (-100 ignored) and the label-token count of
    # the global batch (None: this batch's own count). Output: the scalar token-level cross-entropy, the
    # float32 sum over label tokens divided by that count (the model's own mean loss for a single batch).
    logits = outputs["logits"]
    labels = labels.to(logits.device)
    total = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    count = labels.ne(IGNORE).sum() if num_items_in_batch is None else num_items_in_batch
    return total / count


def dev_chrf(items, hyps) -> dict:
    # Inputs: DevItems and hypotheses aligned to them. Output: {"chrf_pp": corpus chrF++}.
    score = chrf_metric().corpus_score(list(hyps), [[it.ref for it in items]]).score
    return {"chrf_pp": score}


def examples_digest(steps: dict) -> str:
    # Input: {global step: example ids}. Output: 16 hex characters of the SHA-1 of the per-step sorted
    # id lists in step order, a fingerprint of the exact data order that is invariant to rank layout.
    blob = json.dumps([[s, sorted(steps[s])] for s in sorted(steps)])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class ByteTrainer(Seq2SeqTrainer):
    # Seq2SeqTrainer with the token-level loss, enriched logs and greedy dev chrF++ evaluation. The
    # eval_dataset is the list of DevItems. Extra keyword arguments: target (the direction's target code),
    # arch, global_batch and gen_batching (Batching for dev generation).

    def __init__(self, *args, target=None, arch=None, global_batch=1, gen_batching=Batching(max_batch=32),
                 **kwargs):
        # Inputs: the usual Seq2SeqTrainer arguments plus the extras above. Installs token_loss as the
        # compute_loss_func, so the Trainer counts label tokens over the global batch and never rescales by
        # accumulation steps, and keeps the token count out of the model's forward call.
        super().__init__(*args, compute_loss_func=token_loss, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.dev, self.target, self.arch = list(self.eval_dataset), target, arch
        self.global_batch, self.gen_batching = int(global_batch), gen_batching
        self.decoding = dataclasses.replace(DecodingConfig(), num_beams=1)
        self._bundle = None
        self._ids = defaultdict(list)
        self._clock = {"log": time.time(), "train": time.time(), "eval_s": 0.0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Inputs: model, a collated batch, whether to return outputs, and the global label-token count.
        # Output: the token-level loss (and model outputs) from the Trainer's own compute_loss on the model
        # inputs and labels; in training mode also records the example ids of the current step.
        if model.training:
            self._ids[self.state.global_step + 1].extend(inputs["example_idx"])
        batch = {**{k: inputs[k] for k in MODEL_KEYS}, "labels": inputs["labels"], "use_cache": False}
        return super().compute_loss(model, batch, return_outputs, num_items_in_batch)

    def train(self, *args, **kwargs):
        # Inputs: Trainer.train arguments. Starts this session's clocks, then trains. Output: TrainOutput.
        now = time.time()
        self._clock = {"log": now, "train": now, "eval_s": 0.0}
        return super().train(*args, **kwargs)

    def _window_stats(self) -> dict:
        # Output: log fields of the window since the last training log, gathered over ranks: examples digest,
        # examples seen, examples/s (eval time excluded) and peak memory.
        steps = defaultdict(list)
        for part in gather_objects(dict(self._ids)):
            for s, ids in part.items():
                steps[s].extend(ids)
        self._ids = defaultdict(list)
        busy = max(time.time() - self._clock["train"] - self._clock["eval_s"], 1e-9)
        stats = {"examples_digest": examples_digest(steps), "examples_seen": self.state.global_step * self.global_batch,
                 "examples_per_s": len(steps) * self.global_batch / busy}
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
        # Output: the cached tagless Trained bundle around the live (unwrapped) model, used for dev decoding
        # from the [start] prefix with the generation suppression ids.
        if self._bundle is None:
            model = self.accelerator.unwrap_model(self.model)
            self._bundle = make_trained(model, self.processing_class, arch=self.arch)
        return self._bundle

    def dev_scores(self) -> dict:
        # Output: dev chrF++. Each rank greedily decodes its shard of the dev items through
        # interp.scoring.generate, the shards are gathered, and every rank scores all of them.
        w = world()
        rows = list(range(w.rank, len(self.dev), w.size))
        model = self.bundle().model
        was_training = model.training
        model.eval()
        hyps = generate(self.bundle(), [d.src for d in self.dev], self.target, self.decoding, self.gen_batching,
                        rows=rows) if rows else []
        model.train(was_training)
        merged = dict(pair for part in gather_objects(list(zip(rows, hyps))) for pair in part)
        return dev_chrf(self.dev, [merged[i] for i in range(len(self.dev))])

    def _determine_best_metric(self, metrics: dict, trial) -> bool:
        # Inputs: the evaluation metrics and the (unused) trial. Output: whether they set a new best; only
        # evaluations at a checkpoint step (global_step a multiple of save_steps) are considered.
        if self.args.save_strategy == "no" or self.state.global_step % self.args.save_steps:
            return False
        return super()._determine_best_metric(metrics, trial)

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
