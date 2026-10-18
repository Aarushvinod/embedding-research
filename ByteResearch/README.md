# ByteResearch

ByteResearch fine-tunes ByT5 and mT5 encoder-decoders (small, base, large) for translation from a hub
language (C1 English, C2 Arabic, C3 Russian) into a set of target languages. It then evaluates the fine-tuned
models and runs five interpretability experiments on them. Everything runs through one CLI made of small,
resumable stages, and every result is a JSON file with the same envelope.

All languages (targets, hubs, controls) and everything tied to them live in `core/config.py`: FLORES,
NTREX, CCMatrix and Wikipedia codes, scripts and their Unicode ranges, sentence terminators, decoder-tag
sentinels, the NLLB source rules and the GlotLID extra codes. The registry is validated on import, so a
changed language list fails early with a clear message.

## Install

The code needs two Python environments.

**Main environment** (data, training, evaluation, interp). Install torch>=2.7 first (on Blackwell GPUs, use
a cu128 or newer build; Colab's preinstalled torch works). Then, from the repository root:

```bash
pip install -r ByteResearch/requirements.txt
```

**Metrics environment** (COMET and MetricX need transformers<5, so they run in a subprocess):

```bash
python -m ByteResearch.run setup-metrics
```

This stage:
- creates a `--system-site-packages` venv at `METRICS_DIR/metrics` (with `uv` if available), so it reuses
  the main torch build;
- installs `requirements-metrics.txt`;
- clones MetricX at the pinned commit into `METRICS_DIR/metricx`;
- runs a COMET/MetricX self-check.

`METRICS_DIR` is `--metrics-dir PATH`, else `$BYTERESEARCH_METRICS_DIR`, else `ROOT/envs`. Setup and scoring
read the same setting. A venv needs symlinks, so on Colab (where `ROOT` is on Google Drive) set it to a local
disk, e.g. `BYTERESEARCH_METRICS_DIR=/content/envs`.

Stage options are `--metrics-base-python PATH`, `--selfcheck-metrics none` and `--metrics-dir PATH`. To use
an interpreter elsewhere, set `BYTERESEARCH_METRICS_PYTHON`.

## Artifacts root

`ROOT` is chosen in this order: `--artifacts PATH`, then `$BYTERESEARCH_ARTIFACTS`, then
`ByteResearch/artifacts/` (gitignored). On Colab, point it at Google Drive.

```
ROOT/data/{flores,ntrex,nllb/{hub}__{x},wiki/{lang},sacrebleu,models}/   cached data + manifest.json
ROOT/data/DATA_MANIFEST.json, ROOT/data/nllb/FILTER_TABLE.json, ROOT/data/nllb/CANDIDATE_TABLE.json
ROOT/checkpoints/{run_id}/checkpoint-{step}/, ROOT/checkpoints/{run_id}/best/
ROOT/results/{experiment}/{run_id}/{measurement}[__{variant}].json
ROOT/arrays/{experiment}/{run_id}/...                                    large .npy/.npz
ROOT/logs/{stage}/{run_id}.log                                           console output per unit
ROOT/envs/metrics, ROOT/envs/metricx (default METRICS_DIR), ROOT/tmp
ROOT/smoke/...                                                           same layout, --smoke only
```

`run_id` is `{model}__{condition}` (e.g. `byt5-small__C2`). Sweep cells use
`{model}__{condition}__lr{lr}`, and ByT5-vs-mT5 comparisons use `{size}__{condition}`.

## CLI

```bash
python -m ByteResearch.run <stage | composite | list | status> [selections] [flags] [stage options]
```

| Selection | Values (comma lists; default `all`) |
|---|---|
| `--model` | `all`, `small`, `base`, `large`, `byt5`, `mt5`, or keys such as `byt5-small,mt5-small` |
| `--condition` | `C1,C2,C3` |
| `--target` | target codes from `core/config.py` |
| `--testset` | `all`, `main` (= `flores_devtest,ntrex`), `flores_devtest`, `ntrex`, `flores_dev` |
| `--metric` | `all`, `sacre`, `neural`, or `chrf_pp,spbleu,comet,metricx` |
| `--lr` | learning rates (default: the sweep grid 1e-4, 3e-4, 1e-3) |
| `--lang` | study languages for `data.wiki` (default: all 11) |
| `--variant` | LAPE variant tokens such as `k1`, `shared`, `k0.5__en_only` |

Global flags: `--force`, `--no-deps`, `--keep-going`, `--smoke`, `--pilot`, `--gpus N`, `--seed`,
`--deterministic`, `--device`, `--artifacts`.

`list` prints every stage with its grid, requirements and options. `status` prints a done/missing grid for
the current selections.

## Stages

### Data (`data`)

| Stage | Grid | What it builds |
|---|---|---|
| `data.flores` | - | FLORES dev/devtest (pinned revision) and the dev subsets |
| `data.ntrex` | - | NTREX-128 (pinned commit) |
| `data.exclusion` | - | normalized FLORES/NTREX exclusion set |
| `data.lid_model` | - | GlotLID |
| `data.labse_model` | - | LaBSE |
| `data.sacrebleu` | - | the flores200 SPM model for spBLEU |
| `data.thresholds` | - | FLORES dev + devtest calibration of the filter thresholds (`ROOT/data/thresholds/`) |
| `data.nllb` | condition x target | filtered NLLB pairs (hub to target) |
| `data.wiki` | lang | Wikipedia sentences (LAPE only) |
| `data.table` | - | prints and saves `ROOT/data/nllb/FILTER_TABLE.json` (pairs left after each filter) |
| `data.candidates` | condition x candidate | filtered NLLB pairs from each hub to each candidate target |
| `data.candidate_wiki` | candidate | Wikipedia sentences for each candidate target |
| `data.candidate_table` | - | prints and saves `ROOT/data/nllb/CANDIDATE_TABLE.json` |

Candidate targets (`CANDIDATE_LANGUAGES` in `core/config.py`: Bulgarian, Urdu, Bengali, Malay, Albanian,
Ukrainian) get all their data prepared exactly as targets do, but nothing trains on them:
- FLORES and NTREX include them, and so does the exclusion set.
- Their pairs from every hub go through the same filters and cap, into the same `ROOT/data/nllb/` folders.
- Their Wikipedia sentences are sampled too.

To promote a candidate, move its row from `_CANDIDATE_ROWS` to `_LANG_ROWS` (as a target, replacing a
dropped one). Its pairs, FLORES/NTREX rows and Wikipedia sample are then already cached.

To discard the candidates you did not choose, run `python -m ByteResearch.run data.purge --purge sq,uk`. This
permanently deletes their pairs from every hub and their Wikipedia samples, and removes their languages and
directions from the calibrated thresholds. It refuses study and control languages, and it works before or
after you remove them from `_CANDIDATE_ROWS`. Their FLORES/NTREX rows stay in the small shared caches, so the
exclusion set and every data hash are unchanged.

Options: `--pairs N` (pairs kept per NLLB file), `--rebuild-data` and `--labse-dtype {fp32,fp16}` (LaBSE
precision for the NLLB filter; default fp32, fp16 is about 3x faster). The dtype is recorded in each pair
manifest, so it enters the data hashes.

**Data filters.** `data.thresholds` scores every FLORES dev + devtest sentence (about 2,000 per language) with
the measures the filters use and takes the 1st percentile (`DataConfig.calib_percentile`), with no floor:

| Filter | Rule | Used by |
|---|---|---|
| GlotLID | the language's label is the top label with probability >= its FLORES 1st percentile (per language) | NLLB pairs (each side), Wikipedia |
| LaBSE cosine | >= the FLORES 1st percentile (per hub -> target direction) | NLLB pairs |
| x/hub character-length ratio | inside the FLORES 1st-99th percentile band (per direction) | NLLB pairs |
| script ratio | >= 0.8 of letters in the expected script (fixed; the FLORES 1st percentile is saved for reference) | NLLB pairs (each side), Wikipedia |
| length | 20-512 characters per side (fixed; 512 is also the training and decoding cap) | NLLB pairs, Wikipedia |

Exact-pair dedupe, exact-match FLORES/NTREX exclusion, markup and malformed lines are fixed rules. The
per-sentence FLORES scores, the thresholds and their distribution statistics are saved in
`ROOT/data/thresholds/` and printed with the statistics when built, and each pair manifest records the exact
values it was filtered with.

Each language is identified by exactly one GlotLID label, the one of its FLORES variety (`arb_Arab` for Modern
Standard Arabic, `swh_Latn`, `zsm_Latn`, `als_Latn`, `hye_Armn`, ...), so dialect and variant labels do not
count. `LID_LABEL_OVERRIDES` maps Tagalog to GlotLID's `fil_Latn` (GlotLID has no Tagalog label), and a missing
label is an error. `corpora.thresholds.lid_passes(ctx, texts, lang)` applies the same check to any texts.

DecoderLens buckets each lens output by GlotLID's top label: en, src (the hub) or x when that language's label
is on top with probability >= `InterpConfig.lid_threshold` (0.5, a fixed value to revisit before the interp
runs), other for any other confident label, und for low confidence or empty output. It reports each bucket's
fraction of all outputs and its share among identified (non-und) outputs.

### Training

| Stage | Grid | What it does |
|---|---|---|
| `sweep` | model x condition x lr | short fixed-step run per LR (DDP) |
| `sweep.select` | model | picks the LR with the best mean dev chrF++ over C1-C3, saved to `results/sweep/{model}/selection.json` |
| `train` | model x condition | full fine-tuning with early stopping on dev chrF++ (DDP) |

`train` uses `--lr` if given, otherwise the `sweep.select` result. Options are `--per-device-batch`,
`--max-steps` and, for `sweep`, `--keep-checkpoints`.

### Translation evaluation (`eval`)

| Stage | Grid | What it does |
|---|---|---|
| `translate` | model x condition x testset | beam-search hypotheses for all targets of the condition |
| `score` | model x condition x testset x metric | adds one metric's per-sentence scores and aggregates to the translate file |
| `compare` | size x condition x testset | ByT5 vs mT5 paired bootstrap per target and pooled |

Options:
- `translate`: `--eval-rows N`, `--gen-batch N`.
- `score`: `--comet-batch-size`, `--metricx-batch-size`, `--metricx-model`, `--metrics-batch-rows N`.
  Score units requested together share COMET/MetricX worker calls (at most N rows each, default 60000),
  so the neural model loads once per batch of files rather than once per file.
- `compare` always requires both architectures of the size, whatever `--model` selects.
- `compare`: `--did`, which adds the C2/C3-vs-C1 difference-in-differences once all conditions are scored.

### Interpretability (`interp`, or `interp.<experiment>`)

All interp stages run on the `best/` checkpoints of the fine-tuned models.

| Experiment | Measurements, in order | Options |
|---|---|---|
| `interp.hubness` | `states`, `similarity`, `ranks` | |
| `interp.decoderlens` | `logprob`, `lid`, `summary` | `--chunk-rows` |
| `interp.lape` | `activation_probs`, `scores`, `selection`, `overlap`, `correlation`, `ablation`, `ablation_compare` | `--variant`, `--lape-chunk`, `--lape-rows-chunk` |
| `interp.steering` | `directions`, `select`, `logprob`, `generate`, `metrics`, `check`, `summary`, `compare` | `--steer-scales`, `--steer-all-scales`, `--steer-select-scales` |
| `interp.stitching` | `logprob`, `generate`, `metrics`, `summary`, `compare` | `--stitch-layers`, `--stitch-rows`, `--stitch-precompute` |

Notes:
- Each measurement is its own stage, e.g. `interp.lape.overlap`.
- LAPE needs `data.wiki`, and its selection, overlap and ablation units run per k (`--variant k1`).
- `interp.steering.estimate` times one scoring and generation pass, prints projected stage times and writes
  nothing. It is not part of the composite.
- Stitching runs on the C2 and C3 recipients and needs the C1 to C3 models of the same architecture and size;
  it requires (and trains) all three, whatever `--condition` selects.
- The steering and stitching `compare` stages require both architectures of the size.
- Steering, stitching and LAPE ablation score generations with all four metrics, so they need the metrics
  environment. Units run together share the COMET/MetricX worker calls (`--metrics-batch-rows`).
- Generation stops each beam once it has decoded the output character cap (512, or 128 for DecoderLens
  LID), with `4 x chars + 1` tokens as the hard cap.

## Granular runs, resume and `--force`

- A stage runs over the cartesian product of its grid axes, restricted by the selections. Units that are
  already finished are skipped.
- Requirements run first and skip their own finished units. A requirement covers the units its dependents
  need: the same axes as each dependent unit, or an explicit mapping (stitching needs C1 to C3, comparisons
  need both architectures). `--no-deps` turns this off.
- Composites:
  - `all`, `data`, `eval`, `interp` and `interp.<experiment>`;
  - any dotted prefix of a stage name.
- Partial work resumes.
  - **Training** resumes from the newest checkpoint that has a `COMPLETE` marker; half-written checkpoints
    are deleted. Resume refuses if the config, data, batch geometry or tags have changed.
  - **Translations** count as done only while their data, decoding settings and best checkpoint are
    unchanged; after retraining, `translate` refuses the old file unless `--force` is given. The LR selection
    is likewise recomputed when a sweep cell changes.
  - **Interp measurements** flush finished units periodically (`status.complete=false` plus
    `status.units_done`) and continue from there. A result written under a different config fingerprint is
    refused unless `--force` is given.
- `--force` reruns only the stages you named; their requirements skip or resume as usual. For `train`, it
  deletes that run's checkpoints first.
- Data caches are immutable. To replace one, pass `--force --rebuild-data`; the old directory moves to
  `ROOT/data/_superseded/`. `--force` alone leaves a complete cache in place.
- `--keep-going` continues past units that fail.

## `--smoke` and `--pilot`

`--smoke` runs a tiny end-to-end check in `ROOT/smoke/`, with this scope:
- models: byt5-small and mt5-small;
- conditions: C1 and C2;
- targets: `SMOKE_TARGETS` from `core/config.py` (sw and ka);
- data: 2k pairs per NLLB file and 500 Wikipedia sentences per language;
- training: 60 steps (eval and save every 20), global batch 8;
- evaluation: 64 rows; bootstrap: 200 resamples.

`--pilot` runs interp stages on fixed, seeded row subsets, with reduced grids for steering, stitching and
LAPE (k = 1 only, unless `--variant` names other k values). Pilot results never count as, or overwrite, full
results: every measurement file gains a `__pilot` suffix (`{measurement}__pilot.json`, or
`{measurement}__{variant}__pilot.json`) and arrays go to `ROOT/arrays/{experiment}/{run_id}/pilot/`. The same
applies to `translate`, `score` and `compare` under `--pilot`.

## Single GPU and multi-GPU

`sweep` and `train` are DDP stages. With `--gpus N` (N > 1), the runner first runs the requirements in the
current process. It then relaunches the stage under `torch.distributed.run --standalone --nproc_per_node=N`.

The global batch (64 examples) is always world x per-device batch x gradient accumulation, so 1-GPU and
N-GPU runs train on the same batches. Use `--per-device-batch` to fit GPU memory.

All other stages run in one process, on `cuda:0` (or `--device`).

```bash
python -m ByteResearch.run train --model byt5-small --condition C2            # one GPU
python -m ByteResearch.run train --model byt5-small --condition C2 --gpus 4   # four GPUs
```

## Results

Each (experiment, run_id, measurement[, variant]) writes one file:

```
ROOT/results/{experiment}/{run_id}/{measurement}[__{variant}].json
```

Every file has this envelope:

```
{"schema": "byteresearch/result/v1", "experiment", "run_id", "measurement", "variant",
 "unit": {...}, "status": {"complete", "units_done", "fingerprint"}, "config": {...},
 "provenance": {"git_commit", "versions", "data_manifest_hash", "host", "gpu", "created", "updated",
                "world_size", "smoke", "pilot", "seed", ...},
 "per_item": {...}, "aggregate": {...}}
```

- `per_item` holds the granular data: columnar arrays aligned to an id list, per sentence, layer, neuron or
  grid point.
- `aggregate` holds statistics computed from `per_item`: means, pooled scores, bootstrap CIs and p-values.
- Large arrays (e.g. pooled hidden states) are saved under `ROOT/arrays/...` and referenced from the JSON.
- Writes are atomic and file-locked. An unreadable file counts as missing.

Main files:

| File | Contents |
|---|---|
| `results/train/{run_id}/train.json` | training log history, best step, stop reason |
| `results/sweep/{model}__{condition}__lr{lr}/train.json` | one sweep cell |
| `results/sweep/{model}/selection.json` | selected LR |
| `results/translate/{run_id}/{testset}.json` | hypotheses, plus each scored metric |
| `results/compare/{size}__{condition}/{testset}.json` | ByT5 vs mT5 cells; `aggregate.pooled` and `aggregate.wins` per metric |
| `results/compare/{size}__did/{testset}.json` | difference-in-differences (`--did`) |
| `results/interp_{experiment}/{run_id}/{measurement}[__{variant}].json` | interp measurements |

## Checkpoints

- Location: `ROOT/checkpoints/{run_id}/checkpoint-{step}/`.
- Training saves every 500 steps and keeps the last 3 plus the best (`save_total_limit=4`).
- Each checkpoint holds the weights, the tokenizer, the trainer/RNG state, `byteresearch_meta.json`
  (config, data hash, language-tag ids, seeds, world size, versions) and a `COMPLETE` marker.
- When training ends, the best checkpoint is exported to `ROOT/checkpoints/{run_id}/best/`, which evaluation
  and interp load.
- Sweep cells checkpoint under `ROOT/checkpoints/{model}__{condition}__lr{lr}/`. These are deleted once the
  cell finishes unless `--keep-checkpoints` is given.
