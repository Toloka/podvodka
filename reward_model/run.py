import argparse
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from crowdkit.aggregation import BradleyTerry
from sklearn.model_selection import train_test_split
import wandb
from transformers import get_linear_schedule_with_warmup
import lightning as L
import torchmetrics
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from transformers import DataCollatorWithPadding
torch.set_float32_matmul_precision('medium')

sep_token = '</s>'


def sample_pairs(df):
    rows = []
    
    for desc, desc_df in df.groupby('desc'):
        n = len(desc_df)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                prompt_left = desc_df.iloc[i]['prompt']
                prompt_right = desc_df.iloc[j]['prompt']
                if desc_df.iloc[i]['score'] + desc_df.iloc[j]['score'] < 1e-9:
                    y = 0
                else:
                    y = desc_df.iloc[i]['score'] / (desc_df.iloc[i]['score'] + desc_df.iloc[j]['score'])
                
                left_text = f'{desc}{sep_token}{prompt_left}'
                right_text = f'{desc}{sep_token}{prompt_right}'
                rows.append([left_text, right_text, y])
    return pd.DataFrame(rows, columns=['left', 'right', 'y'])


def sample_aux_pairs(df):
    def sample_two_indices(n):
        i = np.random.randint(n)
        j = i
        while j == i:
            j = np.random.randint(n)
        return (i, j)

    def sample_neq(n, i):
        j = i
        k = 0
        while i == j and k < 100:
            j = np.random.randint(n)
            k += 1
        return j
    
    desc_to_prompts = []

    res = []

    words = set()
    for p in df['prompt']:
        for w in p.split():
            words.add(w)
    words = list(words)

    for desc, prompts in df.groupby('image_description'):
        desc_to_prompts.append((desc, prompts['prompt']))

    for i in range(len(desc_to_prompts)):
        prompt_left = desc_to_prompts[i][1].iloc[np.random.randint(len(desc_to_prompts[i][1]))]
        j = sample_neq(len(desc_to_prompts), i)
        prompt_right = desc_to_prompts[j][1].iloc[np.random.randint(len(desc_to_prompts[j][1]))]
        res.append([f'{desc_to_prompts[i][0]}{sep_token}{prompt_left}', f'{desc_to_prompts[i][0]}{sep_token}{prompt_right}', 1.0])

    for i in range(len(desc_to_prompts)):
        q = np.random.randint(len(desc_to_prompts[i][1]))
        prompt_left = desc_to_prompts[i][1].iloc[q]
        w = sample_neq(len(desc_to_prompts[i][1]), q)
        prompt_left2 = desc_to_prompts[i][1].iloc[w]
        res.append([f'{desc_to_prompts[i][0]}{sep_token}{prompt_left}', f'{desc_to_prompts[i][0]}{sep_token}{prompt_left2} {prompt_left}', 1.0])

    for i in range(len(desc_to_prompts)):
        prompt_left = desc_to_prompts[i][1].iloc[np.random.randint(len(desc_to_prompts[i][1]))]
        j = sample_neq(len(desc_to_prompts), i)

        n_words = np.random.randint(10)
        prompt_words = []
        for i in range(n_words):
            prompt_words.append(words[np.random.randint(len(words))])
        prompt_right = ' '.join(prompt_words)
        res.append([f'{desc_to_prompts[i][0]}{sep_token}{prompt_left}', f'{desc_to_prompts[i][0]}{sep_token}{prompt_right}', 1.0])
    return pd.DataFrame(res, columns=['left', 'right', 'y'])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):

        self.labels = [label for label in df['y']]
        self.left_texts = [tokenizer(text, truncation=True, return_tensors="pt") for text in df['left']]
        self.right_texts = [tokenizer(text, truncation=True, return_tensors="pt") for text in df['right']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def __getitem__(self, idx):

        batch_left_texts = self.left_texts[idx]
        batch_right_texts = self.right_texts[idx]
        batch_y = self.get_batch_labels(idx)

        return batch_left_texts['input_ids'][0], batch_left_texts['attention_mask'], batch_right_texts['input_ids'][0], batch_right_texts['attention_mask'], torch.tensor(batch_y, dtype=torch.float32)


class RobertaPairwiseRanker(nn.Module):

    def __init__(self):

        super(RobertaPairwiseRanker, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=1)

    def forward(self, left_input_id, left_mask, right_input_id, right_mask):
        left_linear_output = self.bert(input_ids=left_input_id, attention_mask=left_mask).logits
        right_linear_output = self.bert(input_ids=right_input_id, attention_mask=right_mask).logits

        left_exp = torch.exp(left_linear_output)

        return left_exp / (left_exp + torch.exp(right_linear_output))
    
    def score(self, input_id, mask):
        linear_output = self.bert(input_ids=input_id, attention_mask=mask).logits
        
        return linear_output


class PairwiseRanker(L.LightningModule):
    def __init__(self, model, aux_loader=None, aux_coef=0.1, lr=1e-3, weight_decay=0.01, total_training_steps=100):
        super().__init__()
        L.seed_everything(0)
        self.model = model
        self.criterion = nn.BCELoss()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_training_steps = total_training_steps
        self.aux_loader = aux_loader
        self.aux_coef = aux_coef

    def forward(self, left_input_id, left_mask, right_input_id, right_mask):
        return self.model(left_input_id, left_mask, right_input_id, right_mask)

    def aux_to_device(self, device, batch):
        left_input_id, left_mask, right_input_id, right_mask, y = batch
        return left_input_id.to(device), left_mask.to(device), right_input_id.to(device), right_mask.to(device), y.to(device)

    def training_step(self, batch, batch_idx):
        left_input_id, left_mask, right_input_id, right_mask, y = batch

        output = self.forward(left_input_id, left_mask, right_input_id, right_mask)
        output = output.reshape(-1)
        pred_labels = (output >= 0.5).long()
        y_labels = (y >= 0.5).long()
        self.train_acc(pred_labels, y_labels)

        orig_loss = self.criterion(output, y)
        self.log('train/orig_loss', orig_loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.aux_loader is not None:
            iterator = iter(self.aux_loader)
            aux_batch = next(iterator)
            left_input_id, left_mask, right_input_id, right_mask, y = self.aux_to_device(left_input_id.device, aux_batch)
            output = self.forward(left_input_id, left_mask, right_input_id, right_mask)
            output = output.reshape(-1)
            aux_loss = self.criterion(output, y)
            self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True)
            loss = orig_loss + self.aux_coef * aux_loss
            self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log('train/accuracy', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return orig_loss if self.aux_loader is None else loss

    def validation_step(self, batch, batch_idx):
        left_input_id, left_mask, right_input_id, right_mask, y = batch
        output = self.forward(left_input_id, left_mask, right_input_id, right_mask)
        output = output.reshape(-1)
        loss = self.criterion(output, y)
        pred_labels = (output >= 0.5).long()
        y_labels = (y >= 0.5).long()
        self.accuracy(pred_labels, y_labels)
        self.log('eval/accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('eval/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, 500, self.total_training_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--aux_coef', type=float, default=0.01)
    parser.add_argument('--img_name_map', type=str, default='/mnt/data/img_name_map.csv')
    parser.add_argument('--base_url', type=str, default='https://sdcomparisons.blob.core.windows.net/prompts-comparison/')
    parser.add_argument('--train_data', type=str, default='/mnt/data/train_data.csv')
    parser.add_argument('--output_model', type=str, default='/mnt/data/model.pt')
    parser.add_argument('--wandb_entity', type=str, default='toloka-research')
    parser.add_argument('--wandb_project', type=str, default='prompts_reward_model')
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
    )

    res_map = pd.read_csv(args.img_name_map)
    img2prompt = {args.base_url + img_name: prompt for _, img_name, prompt in tqdm(res_map[['image_name', 'prompt']].itertuples(), total=len(res_map))}
    df = pd.read_csv(args.train_data, sep='\t')
    df = df[df['GOLDEN:result'].isna()]
    df['left'] = df['INPUT:left_0'].apply(lambda x: img2prompt[x])
    df['right'] = df['INPUT:right_0'].apply(lambda x: img2prompt[x])
    ann_df = df[['INPUT:prompt', 'left', 'right', 'OUTPUT:result', 'ASSIGNMENT:worker_id']]
    ann_df.columns = ['desc', 'left', 'right', 'label', 'worker']
    ann_df['label'] = ann_df.apply(lambda row: row['left'] if row['label'] == 'left' else row['right'], axis=1)

    scores = {}

    for descr, d in ann_df.groupby('desc'):
        scores[descr] = BradleyTerry(100).fit_predict(d)


    train_desc, val_desc = train_test_split(list(scores.keys()), test_size=0.2, random_state=42)

    ds_train = []
    ds_val = []

    for desc, prompt_scores in scores.items():
        for prompt, score in prompt_scores.items():
            line = [desc, prompt, score]
            if desc in train_desc:
                ds_train.append(line)
            else:
                ds_val.append(line)
    ds_train = pd.DataFrame(ds_train, columns=['desc', 'prompt', 'score'])
    ds_val = pd.DataFrame(ds_val, columns=['desc', 'prompt', 'score'])

    ds_train = sample_pairs(ds_train)
    ds_val = sample_pairs(ds_val)

    res_map = res_map[~res_map['image_description'].isin(set(val_desc))]
    ds_aux = sample_aux_pairs(res_map)
    

    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    train, val, aux = Dataset(ds_train, tokenizer), Dataset(ds_val, tokenizer), Dataset(ds_aux, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def two_text_data_collator(inputs):
        left_dict = [{'input_ids': x[0]} for x in inputs]
        right_dict = [{'input_ids': x[2], 'y': x[4]} for x in inputs]
        
        col_left = data_collator(left_dict)
        col_right = data_collator(right_dict)
        
        return col_left['input_ids'], col_left['attention_mask'], col_right['input_ids'], col_right['attention_mask'], col_right['y']

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=two_text_data_collator)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, collate_fn=two_text_data_collator)
    aux_rand_sampler = torch.utils.data.RandomSampler(aux, num_samples=8, replacement=False)
    aux_dataloader = torch.utils.data.DataLoader(aux, batch_size=8, sampler=aux_rand_sampler, collate_fn=two_text_data_collator)

    ranker_model = RobertaPairwiseRanker()

    model = PairwiseRanker(ranker_model, aux_loader=aux_dataloader, aux_coef=args.aux_coef, lr=args.lr, weight_decay=args.weight_decay, total_training_steps=len(train_dataloader) * args.epochs)
    wandb_logger = WandbLogger()
    trainer = L.Trainer(max_epochs=args.epochs, callbacks=[RichProgressBar()], logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    torch.save(model.model.state_dict(), args.output_model)


if __name__ == '__main__':
    main()
