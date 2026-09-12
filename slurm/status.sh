#!/usr/bin/env bash
# Experiment status: which artifacts exist (training part files + steps, merged results, full-eval
# parts, interp parts) and which byteembed jobs are running/pending, with training progress pulled
# from the live logs. Read-only. Run from the repo root on the cluster:
#   bash slurm/status.sh
set -uo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import glob, json, os, re, subprocess
from pathlib import Path

MAIN = ["byte-small", "subword-small", "byte-base", "subword-base", "byte-large", "subword-large"]
ARMS = ["byte-small", "byte-base", "byte-large"]
LABELS = [("main", MAIN, "results/retrieval_bgem3"), ("bteacher", ARMS, "results/retrieval_bgem3_bteacher"),
          ("brandom", ARMS, "results/retrieval_bgem3_brandom")]
INTERP = [("english", MAIN), ("script", MAIN), ("segment", ARMS)]

def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def train_cell(base, label, m):
    d = load(f"{base}_part_{m}.json")
    bm = ((d or {}).get("models") or {}).get(m)
    if not bm:
        return "-"
    return f"{bm.get('steps_run', '?')}/{bm.get('steps', '?')}"

def ckpt_cell(label, m):
    suf = "" if label == "main" else f"_b-{label[1:]}"
    p = f"checkpoints/{m}_attn_bge-m3{suf}.pt"
    return f"{os.path.getsize(p) / 1e9:.1f}G" if os.path.exists(p) else "-"

def full_cell(label, m):
    return "done" if os.path.exists(f"results/full_eval_part_{label}_{m}.json") else "-"

def interp_state(x, m):
    d = load(f"results/interp_{x}_part_{m}.json")
    if not d:
        return "-"
    if x == "english":
        bat = d.get("battery") or {}
        if all(e in bat for e in ("en", "random")):
            return "done"
        stage = "latent" if "latent" in d else "-"
        stage = "shift" if "shift" in d else stage
        nb = len(d.get("belebele") or {})
        return f"{stage}, belebele {nb}/17" if nb else stage
    if x == "script":
        conds = d.get("cond") or {}
        done = sum(len(v.get("langs") or []) for v in conds.values())
        return "done" if done >= 14 else (f"{done}/14 cond-langs" if done else ("shift" if d.get("shift") else "-"))
    if x == "segment":
        n = len(d.get("langs") or {})
        return "done" if n >= 10 and d.get("transfer") else f"{n}/10 langs" + (" +transfer" if d.get("transfer") else "")
    return "done"

print("== training parts (steps_run/steps) | checkpoint | full-eval part ==")
print(f"  {'label':9}{'model':15}{'train':>16}{'ckpt':>8}{'full-eval':>11}")
for label, models, base in LABELS:
    for m in models:
        print(f"  {label:9}{m:15}{train_cell(base, label, m):>16}{ckpt_cell(label, m):>8}{full_cell(label, m):>11}")
    merged = load(base + ".json")
    n_m = len((merged or {}).get("models") or {})
    extra = " + baselines" if merged and merged.get("baselines") else ""
    print(f"  {label:9}{'(merged)':15}{(f'{n_m} models{extra}' if merged else '-'):>16}")
base_fe = "done" if glob.glob("results/full_eval_part_baseline_*.json") else "-"
fe_merged = [p for p in glob.glob("results/full_eval*.json") if "_part_" not in p]
print(f"  {'baseline':9}{'bge-m3':15}{'':>16}{'':>8}{base_fe:>11}   full-eval merged: {fe_merged or '-'}")

print("\n== interp parts (results/interp_<x>_part_<model>.json) ==")
print(f"  {'model':15}" + "".join(f"{x:>18}" for x, _ in INTERP))
for m in MAIN:
    print(f"  {m:15}" + "".join(f"{(interp_state(x, m) if m in ms else ''):>18}" for x, ms in INTERP))
merged_i = [x for x, _ in INTERP if os.path.exists(f"results/interp_{x}.json")]
print(f"  merged: {merged_i or '-'}")

print("\n== jobs (squeue: byteembed / be-* / fe-* / in-*) ==")
try:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%j|%T|%M|%R"],
                         capture_output=True, text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as e:
    out = ""
    print(f"  (squeue unavailable: {e})")
rows = [l.split("|") for l in out.splitlines() if re.match(r"^\d+\|(byteembed|be-|fe-|in-)", l)]
if not rows and out is not None:
    print("  none queued or running")
for jid, name, state, t, reason in rows:
    info = ""
    logs = sorted(glob.glob(f"slurm-{name}-{jid}.out"))
    if logs:
        txt = Path(logs[-1]).read_text(encoding="utf-8", errors="replace")
        m = re.search(r"model=(\S+)\s+boundary=(\S+)", txt)
        steps = re.findall(r"step (\d+)/(\d+)", txt)
        if m:
            info += f"  {m.group(1)}" + ("" if m.group(2) == "none" else f" (arm {m.group(2)})")
        if steps:
            info += f"  step {steps[-1][0]}/{steps[-1][1]}"
    print(f"  {jid:>9} {name:22} {state:9} {t:>11} {reason:24}{info}")
PY
