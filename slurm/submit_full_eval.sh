#!/bin/bash
# FULL-CORPUS final evaluation — run AFTER all trainings finish. One job per model (13 total:
# 12 students + the BGE-M3 teacher baseline) + a final merge job that prints the big table.
#
#   PARTITION=clip ACCOUNT=clip QOS=huge-long GRES=gpu:1 CONSTRAINT=Ampere \
#     bash slurm/submit_full_eval.sh
#
# Pool text caches are shared (first job to need a pool builds it; en streams its 33M-row corpus
# once — the big one-time build). Each model then pays only its own encoding time: subword models
# minutes-to-an-hour, byte-large up to ~12-20h (the 500k anchor pools dominate). Re-running skips
# models whose full_eval part-file already exists.
set -euo pipefail

TEACHER="bge-m3"; POOLING="attn"
PARTITION="${PARTITION-clip}"
ACCOUNT="${ACCOUNT-clip}"
QOS="${QOS-huge-long}"
GRES="${GRES-gpu:1}"
CONSTRAINT="${CONSTRAINT-Ampere}"
SFLAGS=()
[ -n "$PARTITION" ]  && SFLAGS+=(--partition="$PARTITION")
[ -n "$ACCOUNT" ]    && SFLAGS+=(--account="$ACCOUNT")
[ -n "$QOS" ]        && SFLAGS+=(--qos="$QOS")
[ -n "$GRES" ]       && SFLAGS+=(--gres="$GRES")
[ -n "$CONSTRAINT" ] && SFLAGS+=(--constraint="$CONSTRAINT")

sb() { sbatch --parsable "${SFLAGS[@]}" --cpus-per-task=8 --mem=48G --requeue \
      --output=slurm-%x-%j.out "$@"; }

IDS=()
submit_model() {  # <label> <results-file> <model>
  jid=$(sb --job-name="fe-$1-$3" --time=36:00:00 --wrap \
    "python -u -m byte_embed.full_eval --results $2 --label $1 --only $3 --pooling $POOLING")
  IDS+=("$jid"); echo "full-eval $1/$3: $jid"
}

for m in byte-small subword-small byte-base subword-base byte-large subword-large; do
  submit_model main results/retrieval_bgem3.json "$m"
done
for m in byte-small byte-base byte-large; do
  submit_model bteacher results/retrieval_bgem3_bteacher.json "$m"
  submit_model brandom  results/retrieval_bgem3_brandom.json  "$m"
done

jid=$(sb --job-name=fe-baseline --time=24:00:00 --wrap \
  "python -u -m byte_embed.full_eval --teacher-baseline")
IDS+=("$jid"); echo "full-eval baseline/BGE-M3: $jid"

DEP=$(IFS=:; echo "${IDS[*]}")
sb --job-name=fe-merge --time=00:30:00 --dependency=afterany:"$DEP" --wrap \
  "python -u -m byte_embed.full_eval --merge" >/dev/null && echo "merge queued (afterany)"
echo "submitted: 13 full-eval jobs + 1 merge"
