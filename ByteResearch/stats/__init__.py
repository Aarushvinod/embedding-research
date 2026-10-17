# Statistical utilities for ByteResearch comparisons.
# bootstrap implements the paired bootstrap over sentence ids used for every model comparison.
# Mean metrics resample per-sentence values and corpus metrics resample sufficient statistics.
# Pooled statistics resample FLORES rows jointly across targets because the rows are parallel.
# chrF++ and spBLEU corpus scores are recomputed from summed statistics in vectorised form.
# Seeds are fixed so every reported interval is reproducible.
