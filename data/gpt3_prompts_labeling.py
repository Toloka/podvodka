import openai
import pandas as pd
from openai.error import RateLimitError, ServiceUnavailableError, APIError
from tqdm import tqdm
import time
import logging
import argparse


def prepare_gpt3_prompt(few_shot_string, prompt_string):
    return few_shot_string.format(prompt_string)


PRICE_FOR_1000_TOKENS = 0.02

parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('-t', '--token', type=str, required=True, help='OpenAI API token')
parser.add_argument('-org', '--organization', type=str, required=False, help='Id of organization in OpenAI', default='')
parser.add_argument('-o', '--output', type=str, required=False, help='Path to output csv file', default='result.csv')
parser.add_argument('-i', '--input', type=str, required=False,
                    help='Path to input tsv file generated in data_cleaning.ipynb',
                    default='cleaned_discord_prompts.tsv')
parser.add_argument('-r', '--remove_prompts_without_meaning', type=bool, required=False,
                    help='''Should Discord prompts that don't describe a specific object be removed from output file?
                    If not, then the image_description will be "_no_object".''',
                    default=True)
args = parser.parse_args()
output_path = args.output

openai.api_key = args.token
if args.organization:
    openai.organization = args.organization

with open('few_shot_prompt.txt') as f:
    few_shot_prompt = f.read()

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('api_requests.log')
formatter = logging.Formatter(
    '%(asctime)s: %(message)s'
)
file_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

dataset = pd.read_csv(args.input, sep='\t')
dataset = dataset[~dataset.content.isna()]
dataset = dataset[~dataset.content.duplicated()].reset_index(drop=True)

total_tokens = 0
prompts_to_description = dict()
discord_prompts_iterator = iter(tqdm(dataset.content.values))

with open('api_requests.log') as f:
    file_lines = f.readlines()
    if len(file_lines) == 0:
        logger.info("{} | {} | {} | {}".format('sent_text', 'respond_text', 'tokens_spent', 'total_tokens_spent'))
    else:
        for line in file_lines[1:]:
            prompt_log, description_log, tokens_spent_log, total_tokens_spent_log = \
                ': '.join(line.split(': ')[1:]).split(' | ')
            prompts_to_description[prompt_log] = description_log
            total_tokens += int(tokens_spent_log)

exception_flag = True
while True:
    try:
        if exception_flag:
            discord_prompt = next(discord_prompts_iterator)
        if discord_prompt in prompts_to_description:
            continue
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prepare_gpt3_prompt(few_shot_prompt, discord_prompt),
            temperature=0.5,
            max_tokens=2300
        )
        image_description = response['choices'][0]['text']
        tokens = response['usage']['total_tokens']
        total_tokens += tokens
        prompts_to_description[discord_prompt] = image_description
        logger.info("{} | {} | {} | {}".format(discord_prompt, image_description, str(tokens), str(total_tokens)))
        tqdm.write(
            "discord prompt: {}\ngenerated image description: {}\ntokens spent: {}, total money spent: ${}\n".format(
                discord_prompt, image_description, tokens, round(total_tokens/1000*PRICE_FOR_1000_TOKENS, 2)),
            end='\n')
        exception_flag = True
    except (RateLimitError, ServiceUnavailableError, APIError):
        exception_flag = False
        time.sleep(1.0)
        continue
    except StopIteration:
        break

prompts = []
image_descriptions = []
for prompt, image_description in prompts_to_description.items():
    prompts.append(prompt)
    image_descriptions.append(image_description)

res_df = pd.DataFrame({'prompt': prompts, 'image_description': image_descriptions})
if args.remove_prompts_without_meaning:
    res_df = res_df[res_df.image_description != '_no_object']
res_df.to_csv(output_path, index=False)
