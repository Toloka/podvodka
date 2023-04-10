# BestPromptsV2

# Step 1:

Fine-tuned GPT-2 Large: https://huggingface.co/toloka/gpt2-large-supervised-prompt-writing.

Use the following template: `image description</s>`

W&B report of the final model: https://wandb.ai/toloka-research/gpt-2-self-supervised-prompt-writing/runs/byxqftxf


# Step 2:

Collect the annotation using pairwise comparisons of images generated from prompts written by the base model using `preference_collection`

Train a reward model using `reward_model`


The final model: https://huggingface.co/toloka/prompts_reward_model

W&B report: https://wandb.ai/toloka-research/prompts_reward_model/runs/bbq7xxbh.
