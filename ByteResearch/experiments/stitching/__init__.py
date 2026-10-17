# Cross-model encoder stitching experiment (interp_stitching) on the fine-tuned C1-C3 models.
# A donor encoder runs blocks 1..l on its own source and the recipient's blocks l+1..L, final norm and
# decoder continue, for every l in 0..L, with recipients and donors from InterpConfig.stitch_donors.
# mechanics.py holds the hook-based stitching itself, the identity check and the memmap donor cache.
# measure.py registers the logprob, generate and metrics stages, summary.py the per-layer gap
# summary and compare.py the ByT5-vs-mT5 comparison plus the interp.stitching composite.
# Self-stitching at every l must reproduce the normal forward pass exactly.
