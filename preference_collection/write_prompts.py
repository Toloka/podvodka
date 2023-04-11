import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import pandas as pd
from tqdm.auto import tqdm


MODEL_PATH = 'toloka/gpt2-large-supervised-prompt-writing'
SEP_TOKEN = "</s>"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to('cuda:0')
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

prompt_checker = pipeline('text-classification', model='../gpt-2-self-supervised-prompt-writing/prompt_correctness_classifier', device='cuda:0')


def check_prompt(img_description, prompt):
    return prompt_checker(f'{img_description}</s>{prompt}')[0]['label'] == 'CORRECT'


def generate(img_description):
    prompt = img_description + SEP_TOKEN
    
    for _ in range(10):
        generated_text = text_generator(prompt, max_length=100, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        generated_string = generated_text.replace('<|endoftext|>', '')
        try:
            image_description, image_prompt = generated_string.split(SEP_TOKEN)
            if check_prompt(img_description, image_prompt): 
                return image_prompt
        except ValueError:
            print('Error')
    return image_prompt


def generate_prompts(img_description, n_prompts=4):
    prompt_set = set()
    i = 0
    while len(prompt_set) < n_prompts:
        prompt_set.add(generate(img_description))
        i += 1
        if i == 10:
            break

    return prompt_set


def main(args):
    df_train = pd.read_csv(args.input_file)

    img_descriptions = [l.split('</s>')[0] for l in df_train['text'].values]
    prompts = [{l.split('</s>')[1].replace('<|endoftext|>', '')} for l in df_train['text'].values]

    for i, descr in tqdm(enumerate(img_descriptions), total=len(img_descriptions)):
        prompts[i] = prompts[i] | generate_prompts(descr)
        print(f'=={descr}==')
        for prompt in prompts[i]:
            print(prompt)

    df_lines = []
    for i, descr in enumerate(img_descriptions):
        for prompt in prompts[i]:
            df_lines.append([descr, prompt])
    
    df = pd.DataFrame(df_lines, columns=['image_description', 'prompt'])
    df.to_csv(args.prompts_file, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, default='gpt2_prompts.csv')
    parser.add_argument('--input_file', type=str, default='../data/train_strings.csv')
    args = parser.parse_args()
    main(args)
