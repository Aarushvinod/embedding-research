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
SCRIPT_M = MAIN + ["BGE-M3"]           # exp 4 also runs the teacher (BGE-M3) as its ceiling arm
INTERP = [("english", MAIN), ("script", SCRIPT_M), ("segment", ARMS)]
# Ask each experiment for its own plan size instead of hardcoding it here (numpy-only imports).
try:
    from byte_embed.interp_script import cond_cells
    SCRIPT_CELLS = cond_cells()
except Exception:                                        # noqa: BLE001 — status must never fail
    SCRIPT_CELLS = 19
try:
    from byte_embed.interp_english import n_belebele_cells
    EN_CELLS = n_belebele_cells()
except Exception:                                        # noqa: BLE001
    EN_CELLS = 18

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
        if all(e in bat for e in ("none", "en", "random")):
            return "done"
        stage = "latent" if "latent" in d else "-"
        stage = "shift" if "shift" in d else stage
        nb = len(d.get("belebele") or {})
        ic = (d.get("identity_check") or {}).get("verdict", "")
        tag = " LOADER-WARN" if ic.startswith("WARN") else ""
        return (f"{stage}, belebele {nb}/{EN_CELLS}{tag}" if nb else stage)
    if x == "script":
        conds = d.get("cond") or {}
        done = sum(len(v.get("langs") or []) for v in conds.values())
        tot = SCRIPT_CELLS
        return "done" if done >= tot else (f"{done}/{tot} cond-langs" if done else ("shift" if d.get("shift") else "-"))
    if x == "segment":
        n = len(d.get("langs") or {})
        return "done" if n >= 10 and d.get("transfer") else f"{n}/10 langs" + (" +transfer" if d.get("transfer") else "")
    return "done"

def interp_detail(x, m):
    """One line per model naming the finished units, so a partial run says WHERE it stopped."""
    d = load(f"results/interp_{x}_part_{m}.json")
    if not d:
        return None
    if x == "english":
        bel = d.get("belebele") or {}
        cb = d.get("chosen_block")
        depths = sorted({int(k.split(":")[0]) for k in bel if ":" in k})
        cols = sorted(k.split(":")[1] for k in bel if k.startswith(f"{cb}:"))
        # erasure_check is the whole point of the stage-B2 rerun, so it has to be visible here:
        # without it a part file with a full battery reads as finished while the verification that
        # the intervention actually removes English -- and is not rebuilt downstream -- is absent.
        ec = d.get("erasure_check")
        eck = (f"erasure_check b{ec['block']} {ec['at_intervention']}"
               + (f" -> last b{ec['last_block']} {ec['at_last_block']}"
                  if ec["last_block"] != ec["block"] else " (== last block: says nothing about rebuild)")
               if ec else "erasure_check MISSING")
        rs = d.get("reinstatement")
        eck += ("; reinstatement b%s %s" % (rs["block"], {b: v["erased"] for b, v in rs["at"].items()})
                if rs else "; reinstatement MISSING")
        return (f"block {cb} of {d.get('n_blocks')}; belebele cells {len(bel)}/{EN_CELLS} "
                f"[{'none ' if 'none' in bel else ''}depths {depths}; columns at cb: {' '.join(cols)}]; "
                f"battery {sorted(d.get('battery') or {})}; {eck}")
    if x == "script":
        cond = d.get("cond") or {}
        parts = [f"{k}:{'+'.join(v.get('langs') or []) or '-'}"
                 + (f" (skipped {sorted((v.get('skipped') or {}))})" if v.get("skipped") else "")
                 for k, v in sorted(cond.items())]
        return f"shift {len(d.get('shift') or {})} cells; " + "  ".join(parts)
    if x == "segment":
        langs = d.get("langs") or {}
        done = [l for l, v in langs.items() if v.get("layers") and v.get("surface") and v.get("pretrained")]
        partial = [l for l in langs if l not in done]
        t = d.get("transfer")
        return (f"langs complete {'+'.join(done) or '-'}"
                + (f"; partial {'+'.join(partial)}" if partial else "")
                + (f"; transfer {len(t.get('langs') or t.get('acc') or [])} langs @ layer {t['layer']}"
                   if t else "; transfer not yet run"))
    return None


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
for m in SCRIPT_M:
    print(f"  {m:15}" + "".join(f"{(interp_state(x, m) if m in ms else ''):>18}" for x, ms in INTERP))
merged_i = [x for x, _ in INTERP if os.path.exists(f"results/interp_{x}.json")]
print(f"  merged: {merged_i or '-'}")
for x, ms in INTERP:
    rows = [(m, interp_detail(x, m)) for m in ms]
    rows = [(m, t) for m, t in rows if t]
    if rows:
        print(f"\n  -- {x} detail --")
        for m, t in rows:
            print(f"     {m:15}{t}")

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
STATE_NOTE = {"TIMEOUT": "hit the wall clock — resubmit, it resumes from the part file",
              "FAILED": "non-zero exit — read slurm-<name>-<id>.out",
              "OUT_OF_MEMORY": "raise --mem or lower the batch size",
              "CANCELLED": "cancelled (scavenger preemption without requeue, or by hand)",
              "NODE_FAIL": "node died — resubmit"}
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

print("\n== finished interp/eval jobs in the last 3 days (sacct) ==")
try:
    fin = subprocess.run(["sacct", "-u", os.environ.get("USER", ""), "-S", "now-3days", "-X", "-P", "-n",
                          "--format=JobID,JobName%30,State,Elapsed,ExitCode"],
                         capture_output=True, text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as e:
    fin = ""
    print(f"  (sacct unavailable: {e})")
seen = set()
for line in fin.splitlines():
    f = line.split("|")
    if len(f) < 5 or not re.match(r"^(in-|fe-|be-|byteembed)", f[1]):
        continue
    if f[0] in {r[0] for r in rows}:            # still queued/running: already listed above
        continue
    key = (f[1], f[0])
    if key in seen:
        continue
    seen.add(key)
    note = STATE_NOTE.get(f[2].split()[0], "")
    print(f"  {f[0]:>9} {f[1]:30} {f[2]:14} {f[3]:>10} exit={f[4]:8}{note}")
if fin and not seen:
    print("  none finished in this window")
PY
