#!/bin/bash
# Interpretability analyses on the TRAINED students — no retraining. One job per (experiment, model)
# reading checkpoints/ + the cached 20k eval pools; every script skips finished work, so re-running
# this is free. Five experiments (RETRIEVAL_EXPERIMENT.md, "Interpretability analyses"):
#   params    exp 1  parameter allocation (capacity)             minutes / model
#   alignuni  exp 2  alignment / uniformity (objective)          minutes / model
#   langgeom  exp 3  language geometry -> erasure rerank         ~1h small .. ~4h byte-large
#   script    exp 4  script vs language (transliteration)        ~1h; needs `pip install uroman`
#   segment   exp 5  emergent-segmentation probes (byte only)    ~0.5-2h
#
#   bash slurm/submit_interp.sh                                   # clip A6000s (inference only)
#   PARTITION=scavenger ACCOUNT=scavenger QOS=scavenger bash slurm/submit_interp.sh
#   EXPS="params alignuni" MODELS="byte-small subword-small" bash slurm/submit_interp.sh   # subset
set -euo pipefail

python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" 2>/dev/null || {
  echo "ERROR: this shell's 'python' has no torch. Activate the env first:"
  echo "  source <scratch>/miniconda3/bin/activate && conda activate byteembed"
  exit 1
}
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('uroman') else 1)" 2>/dev/null \
  || echo "WARN: uroman not installed -> the 'script' experiment will fail (pip install uroman)"

PARTITION="${PARTITION-clip}"; ACCOUNT="${ACCOUNT-clip}"; QOS="${QOS-huge-long}"
GRES_BYTE="${GRES_BYTE-gpu:rtxa6000:1}"     # byte-large per-position extraction wants 48GB
GRES_SUB="${GRES_SUB-gpu:1}"; CONSTRAINT="${CONSTRAINT-Ampere}"
EXPS="${EXPS-params alignuni langgeom script segment}"
MODELS="${MODELS-byte-small subword-small byte-base subword-base byte-large subword-large}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFLAGS=()
[ -n "$PARTITION" ] && SFLAGS+=(--partition="$PARTITION")
[ -n "$ACCOUNT" ]   && SFLAGS+=(--account="$ACCOUNT")
[ -n "$QOS" ]       && SFLAGS+=(--qos="$QOS")
sb() { sbatch --parsable "${SFLAGS[@]}" --cpus-per-task=8 --mem=48G --requeue \
      --output=slurm-%x-%j.out "$@"; }
gres_for()  { case "$1" in byte-*) echo "$GRES_BYTE" ;; *) echo "$GRES_SUB" ;; esac; }
hours_for() { case "$1" in langgeom) echo 08:00:00 ;; script|segment) echo 06:00:00 ;; *) echo 02:00:00 ;; esac; }

total=0
for exp in $EXPS; do
  IDS=()
  for m in $MODELS; do
    if [ "$exp" = segment ] && [[ "$m" != byte-* ]]; then continue; fi        # exp 5 is byte-only
    cflag=(); [ -n "$CONSTRAINT" ] && [[ "$m" != byte-* ]] && cflag=(--constraint="$CONSTRAINT")
    jid=$(sb --gres="$(gres_for "$m")" ${cflag[@]+"${cflag[@]}"} --job-name="in-$exp-$m" \
          --time="$(hours_for "$exp")" --wrap "python -u -m byte_embed.interp_$exp --only $m")
    IDS+=("$jid"); total=$((total + 1)); echo "in-$exp-$m: $jid"
  done
  [ "${#IDS[@]}" -eq 0 ] && continue
  DEP=$(IFS=:; echo "${IDS[*]}")
  sb --gres="$GRES_SUB" --job-name="in-$exp-merge" --time=00:30:00 --dependency=afterany:"$DEP" \
     --wrap "python -u -m byte_embed.interp_$exp --merge" >/dev/null && echo "merge ($exp) queued"
done
echo "submitted: $total interp jobs + one merge per experiment"
