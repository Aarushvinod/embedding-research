# Core infrastructure shared by every ByteResearch stage.
# config holds the fixed study constants, registries and default configuration dataclasses.
# paths defines the artifact layout and run identifiers, and io provides atomic file helpers.
# seeding, text and dist cover randomness, text utilities and distributed helpers.
# registry, results and context implement stages, result envelopes and the run context.
# Nothing in this package depends on models, data downloads or experiment code.
