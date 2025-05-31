import argparse
import pathlib
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from models.embed_2_prompt import Embed2Prompt, Embed2PromptModel


def build_dataset(emb_path: str | pathlib.Path, prompt_path: str | pathlib.Path, tokenizer, max_len: int = 128) -> Dataset:
    emb: np.ndarray = np.load(emb_path)
    prompts: List[str] = [l.strip() for l in open(prompt_path, encoding='utf‑8')]
    assert len(prompts) == len(emb), '#prompts must equal #embeddings'

    ds = Dataset.from_dict({'emb': emb, 'text': prompts})

    def encode(example: Dict):
        item = tokenizer(
            example['text'],
            max_length=max_len,
            padding='max_length',
            truncation=True,
        )
        example['labels'] = item['input_ids']
        return example

    return ds.map(encode, remove_columns=['text'])


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    emb_vecs = torch.tensor([b['emb'] for b in batch], dtype=torch.float)
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
    return {'emb_vecs': emb_vecs, 'labels': labels}


def main():
    parser = argparse.ArgumentParser(description='Fine‑tune bridge from embeddings to prompts (GPT‑2 frozen)')
    parser.add_argument('--emb-path', required=True)
    parser.add_argument('--prompt-path', required=True)
    parser.add_argument('--output-dir', default='checkpoints/embed2prompt')
    parser.add_argument('--base-model', default='gpt2')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)  # slightly higher; only bridge learns
    parser.add_argument('--prefix-len', type=int, default=5)
    parser.add_argument('--max-len', type=int, default=250)
    args = parser.parse_args()

    # Tokenizer & LM (LM stays frozen).
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_lm = GPT2LMHeadModel.from_pretrained(args.base_model)
    hidden_size = base_lm.config.n_embd

    # Dataset
    ds = build_dataset(args.emb_path, args.prompt_path, tokenizer, max_len=args.max_len)

    # Bridge & wrapper model
    first_emb = ds[0]['emb']
    emb_dim = len(first_emb) if isinstance(first_emb, list) else first_emb.shape[0]
    bridge = Embed2Prompt(dim_in=emb_dim, dim_out=hidden_size, prefix_len=args.prefix_len)
    model = Embed2PromptModel(base_lm, bridge, tokenizer)

    # TrainingArguments – we train *only* the bridge, so small lr & few epochs suffice.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy='epoch',
        save_safetensors=False,           # avoid shared-memory error in safetensors
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate,
    )

    trainer.train()

    # Save bridge separately for lightweight deployment.
    bridge_path = pathlib.Path(args.output_dir) / 'bridge.pt'
    torch.save({'state_dict': bridge.state_dict(), 'config': {
        'dim_in': emb_dim,
        'dim_out': hidden_size,
        'prefix_len': args.prefix_len,
    }}, bridge_path)
    print(f'\nTraining complete – bridge weights saved to {bridge_path}.')


if __name__ == '__main__':
    main()
