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

python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" 2>/dev/null || {
  echo "ERROR: this shell's 'python' has no torch. Activate the env first:"
  echo "  source <scratch>/miniconda3/bin/activate && conda activate byteembed"
  exit 1
}

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

# LANGS restricts the deep MIRACL pool. The anchors en/zh/ar carry a 500k cap each and dominate the
# cost, while the 20k battery is already saturated there (the four comparable models sit within 0.011
# nDCG on ar and en); LANGS="te,bn,sw,yo" runs the low-resource cells, whose corpora are small.
# Belebele, the QA benchmarks and AfriQA run either way -- they are the low-resource evals already.
LANGS="${LANGS-}"
LFLAG=""; [ -n "$LANGS" ] && LFLAG=" --langs $LANGS"
MODELS="${MODELS-byte-small subword-small byte-base subword-base byte-large subword-large}"
HOURS="${HOURS-36}"          # wall scales with the pool: 36h is the anchor case, not low-resource

IDS=()
submit_model() {  # <label> <results-file> <model>
  jid=$(sb --job-name="fe-$1-$3" --time="$HOURS:00:00" --wrap \
    "python -u -m byte_embed.full_eval --results $2 --label $1 --only $3 --pooling $POOLING$LFLAG")
  IDS+=("$jid"); echo "full-eval $1/$3: $jid${LANGS:+  [langs $LANGS]}"
}

for m in $MODELS; do
  submit_model main results/retrieval_bgem3.json "$m"
done
# ARMS=0 skips the boundary-injection arms; they are not part of the byte-vs-subword comparison.
if [ "${ARMS-1}" = 1 ]; then
  for m in byte-small byte-base byte-large; do
    submit_model bteacher results/retrieval_bgem3_bteacher.json "$m"
    submit_model brandom  results/retrieval_bgem3_brandom.json  "$m"
  done
fi

jid=$(sb --job-name=fe-baseline --time="$HOURS:00:00" --wrap \
  "python -u -m byte_embed.full_eval --teacher-baseline$LFLAG")
IDS+=("$jid"); echo "full-eval baseline/BGE-M3: $jid"

DEP=$(IFS=:; echo "${IDS[*]}")
sb --job-name=fe-merge --time=00:30:00 --dependency=afterany:"$DEP" --wrap \
  "python -u -m byte_embed.full_eval --merge" >/dev/null && echo "merge queued (afterany)"
echo "submitted: 13 full-eval jobs + 1 merge"
