#!/bin/bash
# Interpretability experiments on the TRAINED students — no retraining. Reads checkpoints/ + the cached
# 20k eval pools; every script skips finished work (per stage / per language), so re-running this is
# free and a --requeue'd (preempted) job resumes where it stopped. Three experiments
# (RETRIEVAL_EXPERIMENT.md, "Interpretability experiments"):
#   segment   exp 5  emergent segmentation probes (byte only; trained + pretrained + transfer)  ~1-3h
#   script    exp 4  native vs romanized script (RR / RN conditions)      ~1-3h; needs `pip install uroman`
#   english   exp 3  English erased inside the encoder (LEACE hook)      ~1h small .. ~4h byte-base
#
# MODE=model (default): ONE job per model running the experiments back to back -> 6 GPU jobs + 1 merge.
# MODE=exp:             one job per (experiment, model).
#
#   PARTITION=scavenger ACCOUNT=scavenger QOS=scavenger bash slurm/submit_interp.sh
#   MODELS="byte-small subword-small" bash slurm/submit_interp.sh          # subset of models
#   EXPS="english" MODE=exp bash slurm/submit_interp.sh                    # subset of experiments
#   PARTITION=tron ACCOUNT=nexus QOS=default CPUS=4 MEM=32G bash slurm/submit_interp.sh   # capped QOS
set -euo pipefail

python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" 2>/dev/null || {
  echo "ERROR: this shell's 'python' has no torch. Activate the env first:"
  echo "  source <scratch>/miniconda3/bin/activate && conda activate byteembed"
  exit 1
}
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('uroman') else 1)" 2>/dev/null \
  || echo "WARN: uroman not installed -> the 'script' experiment will fail (pip install uroman)"

PARTITION="${PARTITION-clip}"; ACCOUNT="${ACCOUNT-clip}"; QOS="${QOS-huge-long}"
GRES_BYTE="${GRES_BYTE-gpu:rtxa6000:1}"     # byte per-position extraction wants 48GB
GRES_SUB="${GRES_SUB-gpu:1}"; CONSTRAINT="${CONSTRAINT-Ampere}"
MODE="${MODE-model}"
CPUS="${CPUS-8}"; MEM="${MEM-48G}"                 # per-job CPU / RAM; some QOS cap these (tron default: 4 / 32G)
EXPS="${EXPS-segment script english}"
MODELS="${MODELS-byte-small subword-small byte-base subword-base byte-large subword-large}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFLAGS=()
[ -n "$PARTITION" ] && SFLAGS+=(--partition="$PARTITION")
[ -n "$ACCOUNT" ]   && SFLAGS+=(--account="$ACCOUNT")
[ -n "$QOS" ]       && SFLAGS+=(--qos="$QOS")
sb() { sbatch --parsable "${SFLAGS[@]}" --cpus-per-task="$CPUS" --mem="$MEM" --requeue \
      --output=slurm-%x-%j.out "$@"; }
gres_for()  { case "$1" in byte-*) echo "$GRES_BYTE" ;; *) echo "$GRES_SUB" ;; esac; }
hours_for() { case "$1" in english) echo 12:00:00 ;; segment) echo 10:00:00 ;; *) echo 08:00:00 ;; esac; }
hours_all() { case "$1" in byte-large) echo 30:00:00 ;; byte-*) echo 16:00:00 ;; *-large) echo 12:00:00 ;; *) echo 08:00:00 ;; esac; }
exps_for()  { local out=(); for e in $EXPS; do [ "$e" = segment ] && [[ "$1" != byte-* ]] && continue; out+=("$e"); done; echo "${out[@]}"; }

total=0; ALL_IDS=()
if [ "$MODE" = model ]; then
  for m in $MODELS; do
    cmd=""
    for exp in $(exps_for "$m"); do
      cmd+="python -u -m byte_embed.interp_$exp --only $m; "     # ';' so one failure does not stop the rest
    done
    [ -z "$cmd" ] && continue
    cflag=(); [ -n "$CONSTRAINT" ] && [[ "$m" != byte-* ]] && cflag=(--constraint="$CONSTRAINT")
    jid=$(sb --gres="$(gres_for "$m")" ${cflag[@]+"${cflag[@]}"} --job-name="in-all-$m" \
          --time="$(hours_all "$m")" --wrap "$cmd")
    ALL_IDS+=("$jid"); total=$((total + 1)); echo "in-all-$m: $jid  [$(exps_for "$m")]"
  done
else
  for exp in $EXPS; do
    for m in $MODELS; do
      if [ "$exp" = segment ] && [[ "$m" != byte-* ]]; then continue; fi        # exp 5 is byte-only
      cflag=(); [ -n "$CONSTRAINT" ] && [[ "$m" != byte-* ]] && cflag=(--constraint="$CONSTRAINT")
      jid=$(sb --gres="$(gres_for "$m")" ${cflag[@]+"${cflag[@]}"} --job-name="in-$exp-$m" \
            --time="$(hours_for "$exp")" --wrap "python -u -m byte_embed.interp_$exp --only $m")
      ALL_IDS+=("$jid"); total=$((total + 1)); echo "in-$exp-$m: $jid"
    done
  done
fi
[ "${#ALL_IDS[@]}" -eq 0 ] && { echo "nothing to submit"; exit 0; }
DEP=$(IFS=:; echo "${ALL_IDS[*]}")
mcmd=""; for exp in $EXPS; do mcmd+="python -u -m byte_embed.interp_$exp --merge; "; done
sb --gres="$GRES_SUB" --job-name="in-merge" --time=00:30:00 --dependency=afterany:"$DEP" \
   --wrap "$mcmd" >/dev/null && echo "merge (all experiments) queued after the $total jobs"
echo "submitted: $total interp jobs (MODE=$MODE) + 1 merge"
