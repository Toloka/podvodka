from transformers import GPT2Config, GPT2Model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer
from transformers import PreTrainedModel
import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


SEP_TOKEN = "<SEP>"
MODEL_PATH = "gpt2-finetuned/checkpoint-300"
VAL_STRINGS_PATH = "val_strings.txt"

with open(VAL_STRINGS_PATH) as f:
    val_strings = f.read().split('\n')

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to('cuda:0')
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

val_generated = dict()
for val_string in tqdm(val_strings):
    val_string = val_string.split(SEP_TOKEN)[0] + SEP_TOKEN
    generated_text = text_generator(val_string, max_length=100, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    generated_string = generated_text.split('\n')[0]
    try:
        image_description, image_prompt = generated_string.split(SEP_TOKEN)
    except ValueError:
        continue
    val_generated[image_description] = image_prompt
    tqdm.write(generated_string, end='\n')
    
pd.DataFrame(val_generated.items(), columns=['image_description', 'generated_prompt']).to_csv('generated.tsv', sep='\t', index=None)
