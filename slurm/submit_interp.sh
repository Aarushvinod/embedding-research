#!/bin/bash
# Interpretability experiments on the TRAINED students — no retraining. Reads checkpoints/ + the cached
# 20k eval pools; every script skips finished work (per stage / per language), so re-running this is
# free and a --requeue'd (preempted) job resumes where it stopped. Three experiments
# (RETRIEVAL_EXPERIMENT.md, "Interpretability experiments"):
#   segment   exp 5  segmentation probes (byte only; trained + pretrained + transfer)   budget 10h
#   script    exp 4  native vs romanized (NN / RR / RN x 5 langs)          budget  8h; needs uroman
#             ALSO runs BGE-M3 itself as the ceiling arm -- the students' distillation target, so
#             their romanized drops can be read as inherited-or-not instead of in the abstract.
#   english   exp 3  English erased inside the encoder (LEACE hook)        budget 12h
#   heads     exp 6  language heads: patch a head with another language's signal   budget  8h
# The budgets are deliberately generous: they are wall-clock CEILINGS for the slowest model on the
# slowest card, not estimates (a subword-small pass is minutes to ~2h). MODE=model asks for the SUM
# of the experiments it chains, capped by MAX_HOURS (36 by default) — raise MAX_HOURS only if the QOS
# allows it, and lower it when a QOS MaxWall would otherwise REJECT the job at submit time.
#
# MODE=model (default): ONE job per model running the experiments back to back -> 6 GPU jobs + 1 merge.
# MODE=exp:             one job per (experiment, model).
# Jobs self-requeue 5 minutes before the wall clock (--signal + scontrol requeue, as
# train_model.sbatch does): --requeue alone covers preemption and node failure, NOT a TIMEOUT, and
# every script resumes from its part file, so a wall-clock kill costs only the stage in flight.
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
GRES_BYTE="${GRES_BYTE-gpu:rtxa6000:1}"     # byte per-position extraction wants >=48GB
GRES_SUB="${GRES_SUB-gpu:1}"; CONSTRAINT="${CONSTRAINT-Ampere}"
# byte-base and every *-large are compute-bound; submit_all.sh routes them to Hopper on measured
# throughput (byte-base crawls at ~5.8s/step on an A6000) and H100/H200 also has the memory the
# per-position extraction wants. BIG_CONSTRAINT= (empty) falls back to the A6000 route everywhere.
BIG_CONSTRAINT="${BIG_CONSTRAINT-Hopper}"; GRES_BIG="${GRES_BIG-gpu:1}"
is_big()    { case "$1" in *-large|byte-base) return 0 ;; *) return 1 ;; esac; }
MODE="${MODE-model}"
CPUS="${CPUS-8}"; MEM="${MEM-48G}"                 # per-job CPU / RAM; some QOS cap these (tron default: 4 / 32G)
MAX_HOURS="${MAX_HOURS-36}"                        # ceiling for the MODE=model sum; lower it if a QOS caps wall time
EXPS="${EXPS-segment script english heads}"
MODELS="${MODELS-byte-small subword-small byte-base subword-base byte-large subword-large BGE-M3}"
TEACHER=BGE-M3                                     # exp 4's ceiling arm; no checkpoint, loaded from the hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFLAGS=()
[ -n "$PARTITION" ] && SFLAGS+=(--partition="$PARTITION")
[ -n "$ACCOUNT" ]   && SFLAGS+=(--account="$ACCOUNT")
[ -n "$QOS" ]       && SFLAGS+=(--qos="$QOS")
# --open-mode=append: a --requeue'd job keeps its JobID, so %j resolves to the SAME file; without
# this SLURM truncates it and the pre-preemption half of the log (and status.sh's view of it) is
# gone. train_model.sbatch sets it for the same reason.
sb() { sbatch --parsable "${SFLAGS[@]}" --cpus-per-task="$CPUS" --mem="$MEM" --requeue \
      --signal=B:USR1@300 --open-mode=append --output=slurm-%x-%j.out "$@"; }
# Wrap the payload so SIGUSR1 (5 min before the wall) requeues the job instead of killing it. The
# payload is backgrounded and waited on, because `wait` is interruptible by a trap and a foreground
# command is not. Single-quoted body: $SLURM_JOB_ID must expand on the node, not here.
requeue_wrap() { printf '%s' "trap 'echo \"[wall] SIGUSR1 -> requeue \$SLURM_JOB_ID (resumes from the part file)\"; scontrol requeue \$SLURM_JOB_ID || true' USR1; { $1 } & wait"; }
gres_for()  { if [ -n "$BIG_CONSTRAINT" ] && is_big "$1"; then echo "$GRES_BIG"
              elif [[ "$1" == byte-* ]]; then echo "$GRES_BYTE"; else echo "$GRES_SUB"; fi; }
# --constraint: Hopper for the big models, Ampere for the small subword ones, none for a typed gres.
cons_for()  { if [ -n "$BIG_CONSTRAINT" ] && is_big "$1"; then echo "$BIG_CONSTRAINT"
              elif [[ "$1" == byte-* ]]; then echo ""; else echo "$CONSTRAINT"; fi; }
# Per-experiment ceilings for the SLOWEST model, scaled down for the smaller ones (a
# subword-small pass is minutes, byte-large's per-position extraction is hours): asking 30h for
# byte-small only delays scheduling, and on a capped QOS gets the job rejected outright.
# english now fits 2 + len(CONTROL_LANGS) sequential chains and scores an arm each, so its
# ceiling grew with the control set; override HOURS_ENGLISH to trade wall for scheduling speed.
hours_base() { case "$1" in english) echo "${HOURS_ENGLISH-16}" ;; segment) echo 10 ;; heads) echo 8 ;;
                              *) echo 8 ;; esac; }
size_pct()   { case "$1" in *-large|BGE-M3) echo 100 ;; *-base) echo 75 ;; *) echo 50 ;; esac; }
hours_for()  { local h=$(( ($(hours_base "$1") * $(size_pct "${2-x-large}") + 99) / 100 ))
               [ "$h" -lt 2 ] && h=2; echo "$h"; }
# $1 = experiment, $2 = model. Exp 5 probes byte positions, so it is byte-only; the teacher has no
# student block structure for the activation hooks exps 3 and 5 install, so it runs exp 4 alone.
can_run()   { if [ "$1" = segment ] && [[ "$2" != byte-* ]]; then return 1; fi
              # exp 6 needs the student block structure for its head hooks, like exps 3 and 5
              if [ "$2" = "$TEACHER" ] && [ "$1" != script ]; then return 1; fi
              return 0; }
exps_for()  { local out=(); for e in $EXPS; do if can_run "$e" "$1"; then out+=("$e"); fi; done; echo "${out[@]}"; }
# MODE=model runs exps_for() back to back in ONE job, so its wall clock is the SUM of the same
# per-experiment budgets MODE=exp hands out — never a separate hand-maintained table, which drifted
# into requesting 8h for {script,english} while MODE=exp gave `english` alone 12h. Also follows EXPS.
hours_all() { local t=0 e; for e in $(exps_for "$1"); do t=$((t + $(hours_for "$e" "$1"))); done
              [ "$t" -gt "$MAX_HOURS" ] && t="$MAX_HOURS"; echo "$t"; }
hms()       { printf '%02d:00:00\n' "$1"; }

total=0; ALL_IDS=(); REJECTED=()
if [ "$MODE" = model ]; then
  for m in $MODELS; do
    cmd=""
    # Run every experiment even if an earlier one fails, but REMEMBER the failures and exit
     # non-zero at the end. Chaining with a bare ';' made the job's exit status that of the LAST
     # command only, so an experiment that died reported COMPLETED and sacct showed nothing wrong.
    cmd="FAILED_EXPS=;"
    for exp in $(exps_for "$m"); do
      cmd+=" python -u -m byte_embed.interp_$exp --only $m || FAILED_EXPS=\"\$FAILED_EXPS $exp\";"
    done
    cmd+=' if [ -n "$FAILED_EXPS" ]; then echo "[job] FAILED experiments:$FAILED_EXPS"; exit 1; fi;'
    cmd="${cmd# }"
    [ -z "$cmd" ] && continue
    cflag=(); c="$(cons_for "$m")"; [ -n "$c" ] && cflag=(--constraint="$c")
    # `|| true`: under `set -e` a single QOS rejection (MaxWall, CPU/mem caps) would abort the whole
    # loop, leaving the already-submitted models running with no merge job behind them.
    jid=$(sb --gres="$(gres_for "$m")" ${cflag[@]+"${cflag[@]}"} --job-name="in-all-$m" \
          --time="$(hms "$(hours_all "$m")")" --wrap "$(requeue_wrap "$cmd")") || jid=""
    if [ -z "$jid" ]; then REJECTED+=("$m"); continue; fi
    ALL_IDS+=("$jid"); total=$((total + 1)); echo "in-all-$m: $jid  [$(exps_for "$m")]"
  done
else
  for exp in $EXPS; do
    for m in $MODELS; do
      if ! can_run "$exp" "$m"; then continue; fi        # exp 5 is byte-only; the teacher is exp 4 only
      cflag=(); c="$(cons_for "$m")"; [ -n "$c" ] && cflag=(--constraint="$c")
      jid=$(sb --gres="$(gres_for "$m")" ${cflag[@]+"${cflag[@]}"} --job-name="in-$exp-$m" \
            --time="$(hms "$(hours_for "$exp" "$m")")" \
            --wrap "$(requeue_wrap "python -u -m byte_embed.interp_$exp --only $m;")") || jid=""
      if [ -z "$jid" ]; then REJECTED+=("$exp/$m"); continue; fi
      ALL_IDS+=("$jid"); total=$((total + 1)); echo "in-$exp-$m: $jid"
    done
  done
fi
report_rejected() {
  [ "${#REJECTED[@]}" -eq 0 ] && return 0
  echo "REJECTED by the scheduler (nothing queued for these): ${REJECTED[*]}"
  echo "  read the sbatch error above: 'QOS'/'accounting policy' -> lower MAX_HOURS or CPUS/MEM;"
  echo "  'node configuration is not available' -> this partition has no node matching the GPU or"
  echo "  constraint asked for (e.g. no Hopper on tron): set BIG_CONSTRAINT= or GRES_BYTE=, or use"
  echo "  another PARTITION. 'sinfo -p <part> -o \"%.20N %.5c %.9m %.30f %.30G\" | sort -u' lists them."
  return 1
}
if [ "${#ALL_IDS[@]}" -eq 0 ]; then echo "nothing submitted"; report_rejected; exit $?; fi
DEP=$(IFS=:; echo "${ALL_IDS[*]}")
mcmd=""; for exp in $EXPS; do mcmd+="python -u -m byte_embed.interp_$exp --merge; "; done
sb --gres="$GRES_SUB" --job-name="in-merge" --time=00:30:00 --dependency=afterany:"$DEP" \
   --wrap "$mcmd" >/dev/null && echo "merge (all experiments) queued after the $total jobs"
echo "submitted: $total interp jobs (MODE=$MODE) + 1 merge"
report_rejected
