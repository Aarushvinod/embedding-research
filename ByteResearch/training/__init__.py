# Fine-tuning stages built on the Hugging Face Seq2SeqTrainer.
# The package holds the dataset and collator (formatting via modeling.encode), the trainer subclass
# with the standard token-level loss and greedy dev chrF++, the checkpoint callbacks and the resume logic.
# Stage `sweep` runs the learning-rate grid on a few fixed representative directions per model, and
# `sweep.select` picks one LR per model from them.
# Stage `train` runs one tagless model per direction (model, condition, target) on that single pair file with
# early stopping, and exports the best checkpoint, whose COMPLETE marker and byteresearch_meta.json carry
# the tagless decoder prefix, hashes and seeds.
# Modules here register their stages with core.registry on import.
