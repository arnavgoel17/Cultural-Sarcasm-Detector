# src/preprocess.py

import torch

from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizerFast

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def get_tokenizer():
    """
    Loads and returns the DistilBERT tokenizer.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    return tokenizer


class SarcasmDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.texts      = dataframe["text"].tolist()
        self.labels     = dataframe["label"].tolist()
        self.domains    = dataframe["domain_idx"].tolist()

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text   = str(self.texts[idx])

        label  = self.labels[idx]
        domain = self.domains[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "domain": torch.tensor(domain, dtype=torch.long),
        }


def create_dataloaders(train_df, val_df, test_df, tokenizer):

    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset   = SarcasmDataset(val_df,   tokenizer)
    test_dataset  = SarcasmDataset(test_df,  tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from data_loader import load_all_data

    print("Loading data...")
    train_df, val_df, test_df, domain_to_idx = load_all_data()

    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    print("\nCreating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    batch = next(iter(train_loader))

    print(f"\nFirst batch shapes:")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  label:          {batch['label'].shape}")
    print(f"  domain:         {batch['domain'].shape}")

    print(f"\nSample token IDs (first example, first 20 tokens):")
    print(batch["input_ids"][0][:20])

    print(f"\nSample attention mask (first example, first 20 positions):")
    print(batch["attention_mask"][0][:20])

    print(f"\nLabels in first batch: {batch['label']}")
    print(f"Domains in first batch: {batch['domain']}")

    print("\nPreprocessing check passed!")