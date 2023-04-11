# Preference Collection

This folder contains scripts for the preference collection for the reward model training.

1. Write prompts for training image descriptions using `write_prompts.py`
2. Generate images for prompts using `gen_images.py`
3. Upload images to S3 using `upload_images.py`
4. Create tasks for Toloka using `create_tasks.py`
5. Create a Toloka project using the interface from `interface.json` and instructions from `instructions.html`
6. Create a pool and upload golden tasks from `golden_tasks.tsv` and the created tasks
