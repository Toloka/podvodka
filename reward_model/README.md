# Reward Model Training

This folder contains the reward model training scripts.

Download the results from Toloka, build a Docker image (don't forget to put your HuggingFace token into `dockerfile`), and run `run.py`.

This script trains a `distilroberta-base` on a pairwise ranking loss. Additionaly, auxillary loss of comparisons with obviously wrong prompts is used.

The final model: https://huggingface.co/toloka/prompts_reward_model

W&B report: https://wandb.ai/toloka-research/prompts_reward_model/runs/bbq7xxbh. The model achieves 0.62 accuracy of comparisons.
