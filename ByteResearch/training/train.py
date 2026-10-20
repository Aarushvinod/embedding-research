# The full fine-tuning stage and the shared run driver used by the LR sweep as well.
# run_training builds one model per direction (model, condition, target[, lr]): it loads the pretrained
# checkpoint, reads that hub-target pair file only, plans the batch in examples, and runs ByteTrainer
# with checkpoint meta, live train.json results and (for full runs) early stopping on dev chrF++.
# The model is tagless (no target tag, no prompt): checkpoint meta records "tags": {} and
# "decoder_prefix": "start", so loading and generation use the [start] prefix everywhere.
# Resumption is automatic: half-written checkpoints and ones whose files do not match their meta are deleted,
# the newest intact one is checked for compatibility and resumed, and one that already ended training goes
# straight to export. Full runs export the best checkpoint to ROOT/checkpoints/{run_id}/best/ (with the Drive
# upload watchdog, after Drive confirms the previous checkpoint at every save and best/ at the end) and mark
# train.json complete.
# The learning rate comes from --lr, else from the model's sweep selection, else a clear error is raised.

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback

from ..core.config import MODELS, InterpConfig, config_hash, resolve_lrs
from ..core.dist import barrier, is_main
from ..core.io import checkpoint_intact
from ..core.paths import run_id
from ..core.registry import opt, register_stage
from ..core.results import git_commit, library_versions, result_done, result_file
from ..interp.scoring import batching_from
from ..modeling.loading import TAGLESS_PREFIX, load_pretrained
from ..modeling.tags import suppress_ids
from . import drive_watch
from .args import batch_plan, steps_plan, train_config, train_seed, training_args
from .callbacks import CheckpointMetaCallback, DriveWatchdogCallback, ResultsCallback, raise_first, train_results
from .collator import PairCollator
from .dataset import PairDataset, data_hash, data_requires, dev_items, direction_hub, train_rows, \
    trim_to_batches
from .resume import check_resume_compat, export_best, finished_state, latest_complete_checkpoint, load_state, \
    read_meta, remove_incomplete
from .trainer import ByteTrainer

EXP = "train"
SWEEP_EXP = "sweep"
MEASUREMENT = "train"
HASH_EXCLUDE = ("per_device_batch", "dataloader_workers", "lrs", "logging_steps")
OPTIONS = (
    opt("--per-device-batch", type=int, default=None, help="per-device micro-batch (default TrainConfig)"),
    opt("--max-steps", type=int, default=None, help="override the optimizer step budget"),
)


@dataclass(frozen=True)
class RunSpec:
    # One fine-tuning run: model key, condition, target, learning rate, result experiment ("train" or
    # "sweep") and whether it is an LR-sweep run (fixed step budget, no early stopping, no export).
    model: str
    condition: str
    target: str
    lr: float
    experiment: str = EXP
    sweep: bool = False

    @property
    def run_id(self) -> str:
        # Output: model__condition__target, with an lr suffix for sweep runs.
        return run_id(self.model, self.condition, self.target, self.lr if self.sweep else None)


def selected_lr(ctx, model: str) -> float:
    # Inputs: Ctx and model key. Output: the single --lr value, else the LR stored by sweep.select;
    # raises ValueError for several --lr values and RuntimeError when neither source exists.
    if ctx.sel.get("lr"):
        lrs = resolve_lrs(ctx.sel["lr"])
        if len(lrs) != 1:
            raise ValueError(f"train needs exactly one --lr, got {lrs}")
        return lrs[0]
    sel = result_file(ctx, SWEEP_EXP, model, "selection").load()
    if not sel or not sel.get("status", {}).get("complete"):
        raise RuntimeError(f"no learning rate for {model}: pass --lr, or run `sweep` and `sweep.select` first")
    return float(sel["aggregate"]["lr"])


def build_meta(ctx, spec: RunSpec, cfg, plan, tok, model, n_train: int, max_steps: int) -> dict:
    # Inputs: Ctx, RunSpec, TrainConfig, BatchPlan, tokenizer, model, training-set size and step budget.
    # Output: the checkpoint meta (config + hash, direction data hash, the tagless marker, suppress ids, ids
    # of tokenizer and model, versions, commit, batch geometry, seeds) shared by every checkpoint.
    seed = train_seed(ctx, cfg)
    config = {"train": {k: v for k, v in dataclasses.asdict(cfg).items() if k not in HASH_EXCLUDE},
              "model": spec.model, "condition": spec.condition, "target": spec.target, "lr": spec.lr,
              "sweep": spec.sweep, "max_steps": max_steps, "decoder_prefix": TAGLESS_PREFIX, "seed": seed}
    return {"model": spec.model, "condition": spec.condition, "target": spec.target, "lr": spec.lr,
            "run_id": spec.run_id, "config": config, "config_hash": config_hash(config),
            "data_hash": data_hash(ctx, spec.condition, spec.target), "tags": {}, "decoder_prefix": TAGLESS_PREFIX,
            "suppress": suppress_ids(tok, model.config.vocab_size), "hub": direction_hub(spec.condition, spec.target),
            "tokenizer": MODELS[spec.model].hf_id,
            "model_id": MODELS[spec.model].hf_id, "versions": library_versions(), "git_commit": git_commit(),
            "world_size": plan.world, "per_device_batch": plan.per_device, "grad_accum": plan.grad_accum,
            "global_batch": plan.global_batch, "seeds": {"seed": seed, "data_seed": seed}, "n_train": n_train,
            "max_steps": max_steps, "deterministic": ctx.deterministic, "smoke": ctx.smoke,
            "lm_head_tied": model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr()}


def prepare_run_dir(ctx, spec: RunSpec):
    # Inputs: Ctx and RunSpec. Rank 0 wipes the run (checkpoints and train.json) under --force and
    # deletes half-written checkpoints; after a barrier every rank looks up the newest complete
    # checkpoint. Output: that checkpoint directory or None.
    if is_main() and ctx.force:
        shutil.rmtree(ctx.paths.checkpoints_dir(spec.run_id), ignore_errors=True)
        ctx.paths.results(spec.experiment, spec.run_id, MEASUREMENT).unlink(missing_ok=True)
    if is_main():
        remove_incomplete(ctx.paths, spec.run_id)
    barrier()
    return latest_complete_checkpoint(ctx.paths, spec.run_id)


def results_writer(ctx, spec: RunSpec, meta: dict, info: dict):
    # Inputs: Ctx, RunSpec, run meta and run info. Output: write(state, complete=False,
    # stop_reason=None) -> aggregate, which rebuilds ROOT/results/{experiment}/{run_id}/train.json.
    rf = result_file(ctx, spec.experiment, spec.run_id, MEASUREMENT,
                     unit={"model": spec.model, "condition": spec.condition, "target": spec.target, "lr": spec.lr})
    extra = {"provenance": {"config_hash": meta["config_hash"], "data_hash": meta["data_hash"],
                            "checkpoints": ctx.paths.relative(ctx.paths.checkpoints_dir(spec.run_id))}}

    def write(state, complete: bool = False, stop_reason=None) -> dict:
        # Inputs: TrainerState, complete flag and stop reason. Writes train.json. Output: the aggregate.
        per_item, aggregate = train_results(state, info, stop_reason)
        rf.write(per_item, aggregate, config=meta["config"], complete=complete, extra=extra)
        return aggregate
    return write


def trainer_dir(ctx, spec: RunSpec) -> Path:
    # Inputs: Ctx and RunSpec. Output: the Trainer's output folder: ROOT/checkpoints/{run_id} for full runs, and
    # for sweeps (which save nothing, though the Trainer always creates its output folder) a fresh local
    # temporary folder, so no empty folder appears under ROOT.
    return Path(tempfile.mkdtemp(prefix="byteresearch_sweep_")) if spec.sweep else ctx.paths.checkpoints_dir(spec.run_id)


def build_trainer(ctx, spec: RunSpec, cfg, plan, model, tok, data: dict, meta: dict, write, max_steps: int,
                  checker=None, out_dir=None):
    # Inputs: Ctx, RunSpec, TrainConfig, BatchPlan, model, tokenizer, data {"train", "dev"}, meta, results
    # writer, step budget, the Drive upload checker (None off rank 0) and the Trainer output folder (default
    # ROOT/checkpoints/{run_id}). Output: the configured ByteTrainer
    # (callbacks: checkpoint meta, live results, and for full runs early stopping plus, when the Drive
    # watchdog is on, the upload check at every save).
    callbacks = [CheckpointMetaCallback(lambda st: {**meta, "step": st.global_step}), ResultsCallback(write)]
    if not spec.sweep:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.patience))
    if not spec.sweep and drive_watch.enabled():
        callbacks.append(DriveWatchdogCallback(checker, drive_watch.timeout_s()))
    collator = PairCollator(tok, cfg.max_chars, cfg.token_factor)
    out_dir = out_dir or ctx.paths.checkpoints_dir(spec.run_id)
    args = training_args(ctx, cfg, out_dir, spec.lr, plan, max_steps, spec.sweep)
    return ByteTrainer(model=model, args=args, data_collator=collator, train_dataset=data["train"],
                       eval_dataset=data["dev"], processing_class=tok, callbacks=callbacks, target=spec.target,
                       arch=MODELS[spec.model].arch, global_batch=plan.global_batch,
                       gen_batching=batching_from(ctx.config(InterpConfig), generation=True))


def fit(ctx, spec: RunSpec, trainer, meta: dict, max_steps: int, patience):
    # Inputs: Ctx, RunSpec, the built trainer, run meta, step budget and early-stopping patience (None for
    # sweeps). Resumes from the newest intact compatible checkpoint (or starts fresh) and trains unless that
    # checkpoint already ended training. Output: the final TrainerState.
    ckpt = prepare_run_dir(ctx, spec)
    if ckpt is not None:
        check_resume_compat(read_meta(ckpt), meta)
    if ckpt is not None and finished_state(load_state(ckpt), max_steps, patience):
        return load_state(ckpt)
    trainer.train(resume_from_checkpoint=str(ckpt) if ckpt is not None else None)
    return trainer.state


def run_training(ctx, spec: RunSpec) -> dict:
    # Inputs: Ctx and RunSpec. Trains (or resumes, or finalizes an already finished run), exports the
    # best checkpoint for full runs (confirming with Drive that best/ arrived when the watchdog is on) and
    # writes the complete train.json on rank 0. Output: the aggregate.
    cfg = train_config(ctx)
    plan = batch_plan(cfg.global_batch, ctx.world.size, cfg.per_device_batch)
    model, tok = load_pretrained(spec.model, dtype=torch.float32)
    rows = train_rows(ctx, spec.condition, spec.target)
    data = {"train": PairDataset(trim_to_batches(rows, plan.global_batch)),
            "dev": dev_items(ctx, spec.condition, spec.target, cfg.dev_per_target)}
    max_steps = steps_plan(len(data["train"]), cfg, spec.sweep)
    meta = build_meta(ctx, spec, cfg, plan, tok, model, len(data["train"]), max_steps)
    info = {"max_steps": max_steps, "global_batch": plan.global_batch, "n_train": len(data["train"]),
            "n_trimmed": len(rows) - len(data["train"]), "n_dev": len(data["dev"]), "lr": spec.lr,
            "target": spec.target, "world_size": plan.world,
            "per_device_batch": plan.per_device, "grad_accum": plan.grad_accum}
    write = results_writer(ctx, spec, meta, info)
    checker = drive_watch.checker_from_env(ctx.paths.root) if is_main() and not spec.sweep else None
    out_dir = trainer_dir(ctx, spec)
    try:
        trainer = build_trainer(ctx, spec, cfg, plan, model, tok, data, meta, write, max_steps, checker, out_dir)
        state = fit(ctx, spec, trainer, meta, max_steps, None if spec.sweep else cfg.patience)
    finally:
        if spec.sweep:
            shutil.rmtree(out_dir, ignore_errors=True)
    aggregate = None
    if is_main():
        if not spec.sweep:
            best = export_best(ctx.paths, spec.run_id, state, meta)
            raise_first([e for e in [drive_watch.confirm(checker, best, drive_watch.timeout_s())] if e])
        aggregate = write(state, complete=True, stop_reason="max_steps" if state.global_step >= max_steps
                          else "early_stopping")
    barrier()
    return aggregate


def train_done(ctx, unit) -> bool:
    # Inputs: Ctx and Unit. Output: whether train.json is complete and best/ is complete with every file at
    # its recorded size.
    rid = unit.run_id
    return result_done(ctx, EXP, rid, MEASUREMENT) and checkpoint_intact(ctx.paths.best_checkpoint(rid), full=False)


@register_stage("train", grid=("model", "condition", "target"), is_done=train_done, requires=data_requires(),
                ddp=True, options=OPTIONS, order=300)
def train(ctx, unit) -> None:
    # Inputs: Ctx and Unit (model, condition, target). Runs the full fine-tuning run of that direction at the
    # model's selected LR.
    run_training(ctx, RunSpec(unit.model, unit.condition, unit.target, selected_lr(ctx, unit.model)))
