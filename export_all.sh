#!/bin/bash
# One-shot off-ramp from the cluster (or Colab terminal): commit the results JSONs to git and push
# the optimizer-stripped model exports to a PRIVATE Hugging Face Hub repo. Safe to re-run.
#
#   bash export_all.sh <hf-write-token> [hf-repo-id]
#   bash export_all.sh hf_abc123...                      # repo defaults to <you>/byteembed-retrieval
#   HF_TOKEN=hf_abc123... bash export_all.sh             # env var instead of argv (safer: argv is
#                                                        # visible in `ps` and shell history)
#
# Git push needs GitHub auth on this machine (gh auth login, or a PAT in the remote URL); if the
# push fails the script warns and CONTINUES to the Hub upload — nothing is lost, push git later.
set -uo pipefail

TOKEN="${1:-${HF_TOKEN:-}}"
[ -z "$TOKEN" ] && { echo "usage: bash export_all.sh <hf-write-token> [hf-repo-id]"; exit 1; }
export HF_TOKEN="$TOKEN"     # huggingface_hub picks this up automatically

REPO="${2:-}"
if [ -z "$REPO" ]; then
  HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])") \
    || { echo "token rejected by the Hub (need a WRITE token)"; exit 1; }
  REPO="$HF_USER/byteembed-retrieval"
fi
echo "== exporting to Hub repo: $REPO (private) =="

# 1) results JSONs -> git (results/ is gitignored by design; -f adds exactly these three)
STAGED=0
for f in results/retrieval_bgem3.json results/retrieval_bgem3_bteacher.json results/retrieval_bgem3_brandom.json; do
  [ -f "$f" ] && git add -f "$f" && STAGED=1 && echo "  staged $f"
done
if [ "$STAGED" = 1 ] && ! git diff --cached --quiet; then
  git commit -m "Run results (auto-export)" && git push \
    && echo "== results committed + pushed ==" \
    || echo "WARN: git push failed (no GitHub auth on this machine?) — commit is local; push later."
else
  echo "== no new/changed results JSONs to commit =="
fi

# 2) checkpoints -> strip optimizer state -> private HF Hub repo (resumable upload)
python export_checkpoints.py --push "$REPO" --private \
  && echo "== models exported: https://huggingface.co/$REPO ==" \
  || { echo "Hub upload failed — checkpoints untouched in checkpoints/; re-run after fixing."; exit 1; }

echo "DONE. Scratch is now safe to lose: results are in git (or committed locally), models on the Hub."
