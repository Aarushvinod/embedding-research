# Fine-tuning stages built on the Hugging Face Seq2SeqTrainer.
# The package holds the dataset and collator (formatting via modeling.encode), the trainer subclass
# with per-sequence loss and greedy dev chrF++, the checkpoint callbacks and the resume logic.
# Stage `sweep` runs the per-model learning-rate grid and `sweep.select` picks the LR per model.
# Stage `train` runs one multilingual model per (model, condition) with early stopping and exports
# the best checkpoint, whose COMPLETE marker and byteresearch_meta.json carry tags, hashes and seeds.
# Modules here register their stages with core.registry on import.
