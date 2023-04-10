import argparse
import pandas as pd
from tqdm.auto import tqdm
import numpy as np


rows = []


def add_comp(df, left, right, descr, base_url='https://sdcomparisons.blob.core.windows.net/prompts-comparison'):
    left_imgs = df[df['prompt'] == left]
    right_imgs = df[df['prompt'] == right]
    
    row = {}
    
    if len(left_imgs) < 4 or len(right_imgs) < 4:
        return 
    
    for i, image_name in enumerate(left_imgs['image_name'].iloc[:4]):
        row[f'INPUT:left_{i}'] = f'{base_url}/{image_name}'
    
    for i, image_name in enumerate(right_imgs['image_name'].iloc[:4]):
        row[f'INPUT:right_{i}'] = f'{base_url}/{image_name}'
    
    row['INPUT:prompt'] = descr
    rows.append(row)


def sample_prompt_pairs(df, descr, base_url='https://sdcomparisons.blob.core.windows.net/prompts-comparison'):
    prompts = pd.unique(df['prompt'])
    total_comp = int(3 * len(prompts) * np.log2(len(prompts)))
    cur_comp = 0
    
    for i in range(len(prompts) - 1):
        if np.random.randint(2) == 0:
            add_comp(df, prompts[i], prompts[i + 1], descr)
        else:
            add_comp(df, prompts[i + 1], prompts[i], descr)
        cur_comp += 1
    
    while cur_comp < total_comp:
        i = np.random.randint(len(prompts))
        j = np.random.randint(len(prompts))
        if i == j:
            continue
        add_comp(df, prompts[i], prompts[j], descr, base_url)
        cur_comp += 1


def main(args):
    res_map = pd.read_csv(args.img_name_map_file)
    for image_description, desc_df in tqdm(res_map.groupby('image_description')):
        sample_prompt_pairs(desc_df, image_description, args.base_url)

    tasks_df = pd.DataFrame(rows)
    tasks_df.to_csv(args.output_file, sep='\t', index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name_map_file', type=str, default='img_name_map.csv')
    parser.add_argument('--output_file', type=str, default='tasks.csv')
    parser.add_argument('--base_url', type=str, default='https://sdcomparisons.blob.core.windows.net/prompts-comparison')
    args = parser.parse_args()
    main(args)
