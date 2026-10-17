# Model-facing helpers shared by training, evaluation and interpretability.
# tags looks up the decoder language tags and the generation suppression list.
# encode is the single source of truth for truncation, tokenisation and decoder formatting.
# loading opens pretrained and fine-tuned checkpoints into a Trained bundle.
# hooks records and edits encoder and decoder internals through forward hooks.
# Code elsewhere must never re-implement any of these steps.
