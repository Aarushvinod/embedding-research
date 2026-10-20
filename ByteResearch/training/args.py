# Training-argument construction for every fine-tuning run (sweep and full training).
# batch_plan turns the global batch, measured in examples, into a per-device batch and a
# gradient-accumulation count, and refuses any geometry where world x per-device x accumulation
# differs from the global batch, so ByT5 and mT5 always see the same number of examples per step.
# train_config applies the CLI and programmatic overrides on top of the (smoke-aware) TrainConfig.
# steps_plan derives the number of optimizer steps from the dataset size and the epoch budget.
# training_args builds the Seq2SeqTrainingArguments with Adafactor, constant-with-warmup, bf16 where
# supported, label-token counts averaged across devices for the token-level loss, step-aligned
# eval/save/logging and exact-resume settings.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from transformers import Seq2SeqTrainingArguments

from ..core.config import TrainConfig

METRIC = "chrf_pp"
OVERRIDE_FLAGS = ("per_device_batch", "max_steps")


@dataclass(frozen=True)
class BatchPlan:
    # Batch geometry of one run: world size, per-device micro-batch and gradient accumulation.
    world: int
    per_device: int
    grad_accum: int

    @property
    def global_batch(self) -> int:
        # Output: examples per optimizer step (world x per-device x accumulation).
        return self.world * self.per_device * self.grad_accum


def batch_plan(global_batch: int, world: int, per_device: int) -> BatchPlan:
    # Inputs: global batch in examples, world size and the requested per-device batch (capped at
    # global_batch / world). Output: the BatchPlan; raises ValueError unless world x per-device x
    # accumulation equals the global batch exactly.
    per = min(int(per_device), int(global_batch) // int(world))
    if per < 1 or int(global_batch) % (int(world) * per):
        raise ValueError(f"global batch {global_batch} is not divisible into world {world} x per-device {per}")
    plan = BatchPlan(int(world), per, int(global_batch) // (int(world) * per))
    if plan.global_batch != int(global_batch):
        raise ValueError(f"batch plan {plan} does not give global batch {global_batch}")
    return plan


def train_config(ctx) -> TrainConfig:
    # Input: Ctx. Output: the smoke-aware TrainConfig with --per-device-batch/--max-steps and any
    # programmatic ctx.opts["train_overrides"] dict applied; save steps must be a multiple of eval steps (every
    # checkpoint has a dev score) and eval steps of the logging interval, so resumed logs line up with
    # uninterrupted ones.
    cfg = ctx.config(TrainConfig)
    flags = {k: ctx.opt(k) for k in OVERRIDE_FLAGS if ctx.opt(k) is not None}
    cfg = dataclasses.replace(cfg, **flags, **dict(ctx.opt("train_overrides", {})))
    if cfg.save_steps % cfg.eval_steps or cfg.eval_steps % cfg.logging_steps:
        raise ValueError(f"save_steps ({cfg.save_steps}) must be a multiple of eval_steps ({cfg.eval_steps}), and "
                         f"eval_steps a multiple of logging_steps ({cfg.logging_steps})")
    return cfg


def steps_plan(n_examples: int, cfg: TrainConfig, sweep: bool = False) -> int:
    # Inputs: training-set size, TrainConfig and whether this is an LR-sweep run. Output: the number
    # of optimizer steps: sweep_steps for sweeps, max_steps when set, else epochs x full batches.
    per_epoch = int(n_examples) // cfg.global_batch
    if per_epoch < 1:
        raise ValueError(f"{n_examples} training examples cannot fill one global batch of {cfg.global_batch}")
    if sweep:
        return int(cfg.sweep_steps)
    return int(cfg.max_steps) if cfg.max_steps else int(cfg.epochs) * per_epoch


def train_seed(ctx, cfg: TrainConfig) -> int:
    # Inputs: Ctx and TrainConfig. Output: the run seed (TrainConfig.seed offset by --seed), used for
    # model-side randomness and the seedable data sampler alike.
    return int(cfg.seed) + int(ctx.seed)


def precision(ctx, cfg: TrainConfig) -> dict:
    # Inputs: Ctx and TrainConfig. Output: {"use_cpu", "bf16", "tf32"}: bf16 when requested and the
    # GPU supports it, TF32 on GPU unless --deterministic, nothing special on CPU.
    cuda = str(ctx.device).startswith("cuda") and torch.cuda.is_available()
    bf16 = bool(cfg.bf16 and cuda and torch.cuda.is_bf16_supported())
    return {"use_cpu": not cuda, "bf16": bf16, "tf32": (not ctx.deterministic) if cuda else None}


def training_args(ctx, cfg: TrainConfig, run_dir, lr: float, plan: BatchPlan, max_steps: int,
                  sweep: bool = False) -> Seq2SeqTrainingArguments:
    # Inputs: Ctx, TrainConfig, checkpoint directory, learning rate, BatchPlan, step count and the
    # sweep flag. Output: Seq2SeqTrainingArguments for one run. Sweeps save no checkpoints and track no best
    # model; full runs save every save_steps, keep the last three plus the best by dev chrF++ and load it at
    # the end.
    seed = train_seed(ctx, cfg)
    return Seq2SeqTrainingArguments(
        output_dir=str(run_dir), per_device_train_batch_size=plan.per_device,
        gradient_accumulation_steps=plan.grad_accum, max_steps=max_steps, learning_rate=float(lr),
        lr_scheduler_type=cfg.lr_scheduler, warmup_steps=cfg.warmup_steps, optim=cfg.optim,
        weight_decay=cfg.weight_decay, max_grad_norm=cfg.max_grad_norm, **precision(ctx, cfg),
        logging_strategy="steps", logging_steps=cfg.logging_steps, eval_strategy="steps",
        eval_steps=cfg.eval_steps, save_strategy="no" if sweep else "steps", save_steps=cfg.save_steps,
        save_total_limit=None if sweep else cfg.save_total_limit, load_best_model_at_end=not sweep,
        metric_for_best_model=None if sweep else METRIC, greater_is_better=True,
        restore_callback_states_from_checkpoint=True, ignore_data_skip=False,
        average_tokens_across_devices=True, dataloader_drop_last=True,
        dataloader_num_workers=cfg.dataloader_workers, remove_unused_columns=False, label_names=["labels"],
        seed=seed, data_seed=seed, report_to="none", disable_tqdm=True, ddp_find_unused_parameters=False,
        prediction_loss_only=True)
