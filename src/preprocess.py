# src/preprocess.py

import torch
# torch is PyTorch — the deep learning framework
# "import torch" gives us access to tensors, which are the fundamental
# data structure of neural networks (like numpy arrays but GPU-compatible)

from torch.utils.data import Dataset, DataLoader
# Dataset  → a PyTorch class we inherit from to wrap our data
#            It tells PyTorch "here's how to get one example at index i"
# DataLoader → wraps a Dataset and handles batching, shuffling, parallel loading
#              It's what we actually loop over during training

from transformers import DistilBertTokenizerFast
# DistilBertTokenizerFast is the tokenizer for DistilBERT
# "Fast" = written in Rust under the hood — much faster than the plain Python version
# It converts raw text → token IDs → attention masks automatically

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
# Importing all our settings: PRETRAINED_MODEL, MAX_SEQ_LENGTH, etc.


# ─────────────────────────────────────────────────────────────────────
# TOKENIZER — load once, reuse everywhere
# ─────────────────────────────────────────────────────────────────────

def get_tokenizer():
    """
    Loads and returns the DistilBERT tokenizer.
    
    from_pretrained() downloads the tokenizer files from Hugging Face
    the first time you call it, then caches them locally.
    Next time you run, it loads from cache — no internet needed.
    
    The tokenizer contains:
    - A vocabulary of 30,522 word-pieces
    - Rules for how to split any English text into those pieces
    - Special token IDs ([CLS]=101, [SEP]=102, [PAD]=0, [UNK]=100)
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    # PRETRAINED_MODEL = "distilbert-base-uncased" from config.py
    # "uncased" means the tokenizer lowercases everything before tokenizing
    # So "GREAT" and "great" and "Great" all become the same token ID
    return tokenizer


# ─────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────

class SarcasmDataset(Dataset):
    """
    A custom PyTorch Dataset for our sarcasm data.

    WHY do we need a custom Dataset class?
    PyTorch's training loop needs to know:
      1. How many examples are there? (→ __len__)
      2. How do I get example number i? (→ __getitem__)
    
    By inheriting from Dataset and implementing these two methods,
    our class plugs seamlessly into PyTorch's DataLoader which handles
    batching, shuffling, and multi-worker loading automatically.

    "class" means we're defining a blueprint (like a template).
    "SarcasmDataset(Dataset)" means our class EXTENDS PyTorch's Dataset class
    — we get all of Dataset's built-in behaviour and add our own on top.
    """

    def __init__(self, dataframe, tokenizer, max_length=MAX_SEQ_LENGTH):
        """
        __init__ is the constructor — it runs when you create an instance:
            dataset = SarcasmDataset(train_df, tokenizer)
        
        "self" refers to this specific instance of the class.
        self.something = ... stores data ON the instance so other methods
        can access it later.

        Parameters:
            dataframe  → pandas DataFrame with columns: text, label, domain_idx
            tokenizer  → the DistilBERT tokenizer from get_tokenizer()
            max_length → maximum token length (128 from config)
        """
        self.texts      = dataframe["text"].tolist()
        self.labels     = dataframe["label"].tolist()
        self.domains    = dataframe["domain_idx"].tolist()
        # .tolist() converts a pandas Series to a plain Python list
        # Plain lists are slightly faster to index than pandas Series
        # We store them as instance variables (self.xxx) so __getitem__ can access them

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        PyTorch calls this internally to know when to stop iterating.
        
        Example: len(train_dataset) → 150661
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns one single example at position idx.
        PyTorch's DataLoader calls this thousands of times to build batches.

        Parameters:
            idx → integer index (0, 1, 2, ... len-1)

        Returns:
            A dictionary with tensors — the exact format DistilBERT expects
        """
        text   = str(self.texts[idx])
        # str() converts to string just in case — defensive programming
        # Occasionally pandas loads a number where you expect text

        label  = self.labels[idx]
        domain = self.domains[idx]

        # ── Tokenize the text ──
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,      # cut at 128 tokens
            padding="max_length",            # pad short sentences to exactly 128
            truncation=True,                 # cut long sentences at 128
            return_tensors="pt"              # return PyTorch tensors, not Python lists
            # "pt" = PyTorch. Could also be "tf" for TensorFlow or "np" for numpy
        )
        # encoding is a dict with keys:
        #   "input_ids"      → tensor of shape [1, 128] — the token ID integers
        #   "attention_mask" → tensor of shape [1, 128] — 1s for real tokens, 0s for padding

        # ── Squeeze out the batch dimension ──
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # The tokenizer returns shape [1, 128] (1 = batch of 1)
        # But our DataLoader will handle batching, so we need shape [128]
        # .squeeze(0) removes the first dimension: [1, 128] → [128]
        # Think of it as removing an unnecessary outer wrapper

        return {
            "input_ids":      input_ids,
            # LongTensor of shape [128] — token IDs fed into DistilBERT
            
            "attention_mask": attention_mask,
            # LongTensor of shape [128] — tells model which positions are real

            "label": torch.tensor(label, dtype=torch.long),
            # torch.tensor() converts a Python int to a PyTorch tensor
            # dtype=torch.long = 64-bit integer — required for classification labels

            "domain": torch.tensor(domain, dtype=torch.long),
            # Same — domain index as integer tensor
        }


# ─────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY — creates train/val/test loaders
# ─────────────────────────────────────────────────────────────────────

def create_dataloaders(train_df, val_df, test_df, tokenizer):
    """
    Wraps our DataFrames in SarcasmDataset instances,
    then wraps those in DataLoader instances.

    A DataLoader is what the training loop actually iterates over.
    It:
      - Calls __getitem__ for each index in a batch
      - Stacks individual examples into batched tensors
      - Optionally shuffles the order each epoch
      - Can load data in parallel using multiple CPU workers

    Returns: train_loader, val_loader, test_loader
    """

    # ── Create Dataset objects ──
    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset   = SarcasmDataset(val_df,   tokenizer)
    test_dataset  = SarcasmDataset(test_df,  tokenizer)

    # ── Create DataLoader objects ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,   # 32 examples per batch (from config)
        shuffle=True,
        # shuffle=True randomizes the order each epoch
        # CRITICAL for training — if the model always sees examples in the same order
        # it memorizes the sequence instead of learning the task
        num_workers=0,
        # num_workers = how many parallel CPU processes load data in the background
        # On Windows, num_workers > 0 requires special handling (spawn vs fork)
        # We use 0 (single process) to avoid Windows multiprocessing errors
        # This is slightly slower but completely reliable on Windows
        pin_memory=True
        # pin_memory=True stores batches in "pinned" (page-locked) CPU memory
        # This makes the transfer from CPU RAM → GPU VRAM significantly faster
        # Only use when training on GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        # During validation we don't compute gradients (no backprop)
        # So we can use a larger batch size (64) — uses same GPU memory
        # but processes examples twice as fast
        shuffle=False,
        # shuffle=False for validation/test — order doesn't matter for evaluation
        # and keeping order makes debugging easier
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


# ─────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────

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

    # ── Inspect one batch ──
    batch = next(iter(train_loader))
    # iter() makes the DataLoader iterable
    # next() pulls the very first batch out of it

    print(f"\nFirst batch shapes:")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    # → torch.Size([32, 128]) — 32 examples, each with 128 token IDs
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    # → torch.Size([32, 128])
    print(f"  label:          {batch['label'].shape}")
    # → torch.Size([32]) — one label per example
    print(f"  domain:         {batch['domain'].shape}")
    # → torch.Size([32])

    print(f"\nSample token IDs (first example, first 20 tokens):")
    print(batch["input_ids"][0][:20])
    # Should start with 101 ([CLS]) and end eventually with 102 ([SEP]) then 0s

    print(f"\nSample attention mask (first example, first 20 positions):")
    print(batch["attention_mask"][0][:20])
    # Should be 1s for real tokens then 0s for padding

    print(f"\nLabels in first batch: {batch['label']}")
    print(f"Domains in first batch: {batch['domain']}")

    print("\nPreprocessing check passed!")