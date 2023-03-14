# Prompts collection and labeling
For Self-Supervised Pretraining (the first step in RLHF), we need to train the model on a custom dataset from the 
desired domain. Our model will solve the following problem: ''image description -> **prompt**'''. For example, 
''a cat -> **a photorealistic image of a cat, trending on artstation, cinematic lightning**''. Bold highlights what the 
model needs to write. The problem is that there is no such dataset.

Our objective is to collect such dataset. Here is how do we make it.

We collect prompts from ```#dreambot-1``` - ```#dreambot-25``` channels at
[Stable Diffusion Discord server](https://discord.gg/stablediffusion).
Users can generate images based on their own prompts in these chats while allowing other users to view them.
To export messages from these chats we apply [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).
We already downloaded these files and stored them in ```sd_dreambots_dotnet_raw_output.zip```

After all chats are collected, we clean data and prepare it for labeling with ```data_cleaning.ipynb```, result are 
stored in ```cleaned_prompts.tsv``` file

Finally, we apply ```gpt3_prompts_labeling.py``` script to collected prompts and get image descriptions with GPT-3 via
OpenAI API. It logs all information about API usage at ```api_requests.log``` as well as already labelled discord 
prompts, so in case of any error during script running you can just restart it, and it will skip all the labelled 
strings for prevent extra expenses.

This script generates image descriptions using OpenAI API based on prompts provided in a TSV file. The generated 
descriptions are saved to a CSV file.

## Usage

The script requires an OpenAI API token, which can be obtained from https://beta.openai.com/signup/. 
The following arguments can be passed to the script:

- `-t`, `--token`: The OpenAI API token.
- `-org`, `--organization`: The ID of the organization in OpenAI. This argument is optional and its default value is an empty string.
- `-o`, `--output`: The path to the output CSV file. This argument is optional and its default value is `result.csv`.
- `-i`, `--input`: The path to the input TSV file generated using `data_cleaning.ipynb`. This argument is optional and  its default value is `cleaned_discord_prompts.tsv`.
- `-r`, `--remove_prompts_without_meaning`: This argument is optional and its default value is `True`. If set to `True`, prompts that do not describe a specific object will be removed from the output file. If set to `False`, the image description will have the value "_no_object".

## Example

```angular2html
python gpt3_prompts_labeling.py --token <YOUR_OPENAI_API_TOKEN> --output descriptions.csv
```

This command will generate image descriptions using the OpenAI API and save the results to the file `descriptions.csv`. The input file `cleaned_discord_prompts.tsv` will be used by default.

