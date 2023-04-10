import argparse
import pandas as pd
from tqdm.auto import tqdm
import torch
from diffusers import StableDiffusionPipeline
from os.path import join


def main(args):
    if args.enable_safety_checker:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(args.device)

    df = pd.read_csv(args.prompts_file)

    cur_img = args.initial_index

    for prompt in tqdm(df['prompt']):
        images = pipe(prompt, num_images_per_prompt=args.num_images_per_prompt)
        for img in images[0]:
            img.save(join(args.output_dir, f'{cur_img}.png'))
            cur_img += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, default='gpt_prompts.csv')
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--enable_safety_checker', action='store_true')
    parser.add_argument('--model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_images_per_prompt', type=int, default=4)
    parser.add_argument('--initial_index', type=int, default=0)
    args = parser.parse_args()
    main(args)
