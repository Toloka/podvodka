import argparse
import os
from typing import List

import torch
from transformers import pipeline
import pandas as pd

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import torch
from trlx.data.default_configs import TRLConfig


def get_config(args):
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path=args.base_model_path, num_layers_unfrozen=args.num_layers_unfrozen),
        tokenizer=TokenizerConfig(tokenizer_path=args.base_model_path, truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=args.lr, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=args.lr)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=args.num_rollouts,
            chunk_size=args.chunk_size,
            ppo_epochs=4,
            init_kl_coef=args.init_kl_coef,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=args.vf_coef,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=80,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--init_kl_coef", type=float, default=0.05)
    parser.add_argument("--vf_coef", type=float, default=1)
    parser.add_argument("--num_layers_unfrozen", type=int, default=2)
    parser.add_argument("--train_path", type=str, default="/mnt/data/train_strings.csv")
    parser.add_argument("--val_path", type=str, default="/mnt/data/val_strings.csv")
    parser.add_argument("--output_path", type=str, default="/mnt/models/gpt2-large-rl-prompt-writing")
    parser.add_argument("--reward_model_path", type=str, default="toloka/prompts_reward_model")
    parser.add_argument("--base_model_path", type=str, default="toloka/gpt2-large-supervised-prompt-writing")
    args = parser.parse_args()

    config = TRLConfig.update(get_config(args).to_dict(), {})

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    reward_model = pipeline('text-classification', model=args.reward_model_path, device=device)

    @torch.no_grad()
    def score(text):
        return reward_model(text, function_to_apply='none')[0]['score']

    def reward_fn(prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        sentiments = [score(x + '</s>' + y) for x, y in zip(prompts, outputs)]
        return sentiments

    train_df = pd.read_csv(args.train_path)
    prompts = [l.split('</s>')[0] + '</s>' for l in train_df['text']]

    val_df = pd.read_csv(args.val_path)
    eval_prompts = [l.split('</s>')[0] + '</s>' for l in val_df['text']][:100]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts * 2,
        config=config,
    )

    trainer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
