# Standalone COMET/MetricX scoring worker, run by metrics/neural.py inside the metrics venv.
# It must never import ByteResearch: it uses only the standard library plus comet, torch and the
# cloned MetricX repository, because the metrics venv pins transformers below version 5.
# Input is a JSONL file of {source, hypothesis, reference} rows and output is a JSONL file with
# one {"score": float} line per input row, in the same order.
# COMET uses the reference-based Unbabel/wmt22-comet-da model through download_model,
# load_from_checkpoint and predict, on one GPU when CUDA is available.
# MetricX runs the official metricx24.predict module (no --qe) as `python -m` would, with the
# repository as cwd, plus a padding collator so batches larger than one work and no Trainer logging
# integrations; TensorFlow, JAX and torchvision are blocked from importing (block_tf_jax): system copies built
# for numpy 2 (or, for torchvision, newer than datasets 3.x expects) crash this pinned environment, and
# neither metric needs them.

import argparse
import json
import os
import sys

BLOCKED = ("tensorflow", "jax", "jaxlib", "flax", "torchvision")


def block_tf_jax():
    # Marks TensorFlow, JAX, Flax and torchvision as absent (a None entry in sys.modules makes `import x` raise
    # ImportError and find_spec return None), so every library takes its path without them. No output.
    for name in BLOCKED:
        sys.modules[name] = None


def parse_args(argv):
    # Input: the argument list (without the program name). Output: the parsed argparse namespace.
    p = argparse.ArgumentParser(description="COMET/MetricX scoring worker")
    p.add_argument("--metric", required=True, choices=sorted(SCORERS))
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--metricx-repo", default=".")
    p.add_argument("--metricx-model", default="google/metricx-24-hybrid-xl-v2p6")
    p.add_argument("--metricx-tokenizer", default="google/mt5-xl")
    p.add_argument("--max-input-length", type=int, default=1536)
    return p.parse_args(argv)


def read_rows(path):
    # Input: a JSONL path. Output: the list of row dicts (blank lines skipped).
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_scores(path, scores):
    # Inputs: an output path and a list of floats. Writes one {"score": x} JSON line per value.
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"score": float(s)}) + "\n" for s in scores)


def score_comet(rows, args):
    # Inputs: input rows and the parsed arguments. Output: COMET segment scores (higher is better),
    # computed on one GPU when available, otherwise on CPU.
    import torch
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(args.comet_model))
    data = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows]
    gpus = 1 if torch.cuda.is_available() else 0
    out = model.predict(data, batch_size=args.batch_size, gpus=gpus, progress_bar=False)
    return [float(s) for s in out.scores]


def metricx_argv(args, out_path):
    # Inputs: the parsed arguments and the MetricX prediction path. Output: the sys.argv for the
    # official reference-based metricx24.predict entry point (no --qe).
    return ["metricx24.predict", "--tokenizer", args.metricx_tokenizer, "--model_name_or_path", args.metricx_model,
            "--max_input_length", str(args.max_input_length), "--batch_size", str(args.batch_size),
            "--input_file", os.path.abspath(args.input), "--output_file", out_path]


def pad_collate(features):
    # Input: a list of tokenized MetricX examples. Output: {input_ids, attention_mask} right-padded
    # with 0 to the longest row, replacing the Trainer's default collator, which can only stack
    # equal-length rows; padding is masked, so batch size 1 and larger batches give the same scores.
    import torch
    width = max(len(f["input_ids"]) for f in features)
    return {k: torch.stack([torch.nn.functional.pad(torch.as_tensor(f[k]), (0, width - len(f[k]))) for f in features])
            for k in ("input_ids", "attention_mask")}


def score_metricx(rows, args):
    # Inputs: input rows (already in MetricX's {source, hypothesis, reference} format) and the
    # parsed arguments. Runs metricx24.predict as `python -m` would (runpy, __main__) with the
    # repository as cwd and on sys.path, after installing the padding collator and turning off the Trainer's
    # reporting integrations (TensorBoard, W&B, ...). Output: MetricX
    # scores in [0, 25] (lower is better), in input order.
    import runpy
    import transformers.integrations
    import transformers.trainer
    repo, here = os.path.abspath(args.metricx_repo), os.getcwd()
    out_path = os.path.abspath(args.output) + ".metricx.jsonl"
    os.environ.update(WANDB_DISABLED="true", WANDB_MODE="disabled")
    transformers.integrations.get_available_reporting_integrations = lambda: []
    transformers.trainer.default_data_collator = pad_collate
    sys.path.insert(0, repo)
    sys.argv = metricx_argv(args, out_path)
    os.chdir(repo)
    try:
        runpy.run_module("metricx24.predict", run_name="__main__", alter_sys=True)
    finally:
        os.chdir(here)
    preds = [float(r["prediction"]) for r in read_rows(out_path)]
    os.remove(out_path)
    if len(preds) != len(rows):
        raise RuntimeError(f"metricx returned {len(preds)} predictions for {len(rows)} rows")
    return preds


SCORERS = {"comet": score_comet, "metricx": score_metricx}


def main(argv=None):
    # Input: the argument list (default sys.argv[1:]). Blocks TensorFlow/JAX, reads the rows, scores them with
    # the chosen metric and writes the scores. Output: the process exit code (0).
    block_tf_jax()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = read_rows(args.input)
    write_scores(args.output, SCORERS[args.metric](rows, args) if rows else [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
