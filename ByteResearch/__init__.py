# ByteResearch: ByT5 versus mT5 encoder-decoder machine translation and interpretability study.
# The package fine-tunes byte-level and subword T5 models from English, Persian and Russian into
# eight low-resource targets and then measures how English-centric each model has become.
# Every unit of work is a registered stage run through `python -m ByteResearch.run <stage>`.
# Shared foundations live in core (config, paths, IO, registry, results), modeling (tags,
# formatting, loading, hooks), stats (bootstrap) and interp (scoring, readouts, unit files).
# Data, training, evaluation, metrics and experiments packages register their stages on import.
