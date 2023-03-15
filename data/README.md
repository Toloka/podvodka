# Prompts Collection and Labeling

This folder contains scripts and data to collect and label prompts for Self-Supervised Pretraining, the first step in RLHF (Reinforcement Learning with Human Feedback).

The objective is to train a model to solve the problem of generating prompts: ''image description -> **prompt**''. For example, ''a cat -> **a photorealistic image of a cat, trending on artstation, cinematic lightning**''. Bold text indicates what the model needs to write. However, there is no existing dataset for this task, so we need to collect one.

## Data Collection

We collect prompts from `#dreambot-1`--`#dreambot-25` channels on the [Stable Diffusion Discord server](https://discord.gg/stablediffusion). Users generate images based on their own prompts in these chats while allowing other users to view them. We export messages from these chats using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) and store them in `sd_dreambots_dotnet_raw_output.zip`.

## Data Cleaning

After collecting the chats, we clean the data and prepare it for labeling using `data_cleaning.ipynb`. The resulting cleaned data is stored in `cleaned_discord_prompts.tsv`.

## Labeling

We label the collected prompts using GPT-3 via OpenAI API with the `gpt3_prompts_labeling.py` script. This script generates image descriptions based on the prompts provided in the `cleaned_discord_prompts.tsv` file. The generated descriptions are saved to a CSV file.

The script logs all API usage information to the ```api_requests.log``` file, including the labeled Discord prompts. If the script encounters any errors during execution, you can restart it and it will skip any previously labeled strings to avoid additional extra costs.

OpenAI API token is required. It can be obtained from https://beta.openai.com/signup/. The following arguments can be passed to the script:

- `-t`, `--token`: The OpenAI API token.
- `-org`, `--organization`: The ID of your organization in OpenAI. This argument is optional and its default value is an empty string.
- `-o`, `--output`: The path to the output CSV file. This argument is optional and its default value is `result.csv`.
- `-i`, `--input`: The path to the input TSV file generated using `data_cleaning.ipynb`. This argument is optional and its default value is `cleaned_discord_prompts.tsv`.
- `-r`, `--remove_prompts_without_meaning`: This argument is optional and its default value is `True`. If set to `True`, prompts that do not describe a specific object will be removed from the output file. If set to `False`, the image description will have the value "_no_object" for such prompts.

To run the script, use the following command:

```bash
python gpt3_prompts_labeling.py --token <YOUR_OPENAI_API_TOKEN>
```

This will generate image descriptions using the OpenAI API and save the results to the file ```result.csv``` (by default). The input file ```cleaned_discord_prompts.tsv``` will be used by default. The script also generates two files, ```train_strings.txt``` and ```test_strings.txt```, which are used for model training.
