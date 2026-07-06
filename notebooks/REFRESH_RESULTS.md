# Getting all results (incl. the new deep-retrieval benchmarks) on your latest models

The eval suite now includes deep-retrieval benchmarks that earlier runs did not score:
**IndicQA** (mr/ta/te), **Mr.TyDi** (te), **2AIRTC** (am, formal), **Amharic-PR** (am),
and **AfriCLIRMatrix** (am/ha, cross-lingual), on top of Belebele (all 9), MIRACL, SIB, FLORES, STS.

The orchestrators **skip models that are already in the results JSON**, so simply re-running a training
cell will **not** add the new benchmarks to models you already trained. To backfill them, use the
**re-eval** path: it loads each saved checkpoint and re-scores it — **no retraining**.

## Fresh runs (training new models)
Nothing to do — the new benchmarks are wired into the default eval, so any new `run(...)` /
`parallel(...)` scores them automatically. Just `git pull` first (see step 2 below).

## Already-trained models — backfill in 4 steps

1. **Open the notebook for those models** in Colab (a GPU helps encoding but an A100 is not required
   for re-eval; a T4 is fine):
   - byte / subword → `byteembed_lowresource_a100.ipynb`
   - matched-transformer → `matched_transformer_a100.ipynb`

2. **Run the setup cells** (GPU check → clone+`git pull` → SONAR reachability → Drive mount). The
   `git pull` is what pulls the new benchmark code. **Point Drive at the same folder** that holds your
   `checkpoints/` and `results/` (the notebooks default to `/content/drive/MyDrive/byteembed_lowres`).

3. **Run the re-eval cell** (step **7b** in the byte notebook, step **5b** in the matched
   notebook). It calls, with the pooling that matches the run:
   ```python
   from byte_embed.reeval import reeval
   reeval('results/byte_lowresource_attn.json', pooling='attn')   # byte + subword (50k, attn)
   reeval('results/matched.json',               pooling='mean')   # matched-transformer
   ```
   Use `qa_only=True` to add **only** the new QA-retrieval benchmarks and keep the existing
   SIB/Belebele/FLORES/STS/MIRACL numbers (fast); omit it to refresh the whole battery.

4. **Re-run the Results cell** (`_summary(...)`). The table now ends with a **RAG-RETRIEVAL** block:
   `IndicQA · Mr.TyDi · Amharic-PR · 2AIRTC · AfriCLIR (am/ha)`.

## Notes
- **`pooling` must match how the model was trained**: byte/subword 50k attn → `'attn'`;
  matched → `'mean'`. The re-eval reloads `checkpoints/{name}_{pooling}.pt` (or `matched_{name}_...`).
- **First build streams a corpus** per new benchmark (2AIRTC ≈ 68 MB; AfriCLIR streams the am/ha
  Wikipedia corpora; IndicQA/Mr.TyDi are small). Pools are cached to `checkpoints/qa_*.json`, so every
  later model reuses them — the download happens once.
- **No new pip dependencies** — corpus fetch/stream uses the standard library.
- **Kinyarwanda** has no public deep-retrieval corpus, so it stays on Belebele (shallow) + FLORES
  bitext; that is expected, not a missing result.
- Re-eval **saves after each model**, so a disconnect mid-way loses nothing — just re-run the cell.
