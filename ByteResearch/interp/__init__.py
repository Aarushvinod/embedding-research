# Shared interpretability machinery used by all five experiments.
# scoring runs the encoder, applies the lens layer norm, scores references and generates.
# readouts provides the provider/edit abstraction used by steering, stitching and LAPE ablation.
# units provides resumable unit-keyed result files with config fingerprints.
# lid (added by the DecoderLens stage) wraps GlotLID language identification.
# Experiments import from here instead of touching models or hooks directly.
