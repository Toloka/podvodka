# Step 1: Supervised Fine-Tuning

We fine-tune a `gpt2-large` in the following setting:

1. We constuct a dataset containig strings `image description</s>prompt<|endoftext|>`
2. We use a standard LM fine-tuning pipeline from the HuggingFace Transformers examples.

You can find the modified version of the fine-tuning script in the `run_clm.py` file.

For hyperparameter search we use W&B Sweep to find the best values of `learing_rate` and `weight_decay`. The Sweep's config is written in the `sweep.yml` file.

Finally, to reproduce the training with the best params, run
```bash
sh run_training.sh
```

We used a single NVIDIA A100 80 GB GPU, the full training takes roughly 90 mins.
