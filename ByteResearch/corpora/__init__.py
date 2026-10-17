# Data acquisition and filtering for ByteResearch, cached immutably under ROOT/data.
# flores, ntrex and exclusion provide the evaluation sets and the normalized exclusion set built from them.
# nllb streams the NLLB-mined pair files and applies the ordered filters in filters, using lid (GlotLID)
# and assets (LaBSE, sacreBLEU's SPM model) for the model-based checks.
# wiki samples Wikipedia sentences for LAPE, and manifest owns the immutable-cache and data-hash logic.
# http holds the resumable gzip line stream and the download helpers.
# stages registers every data stage with core.registry when the runner discovers this package.
