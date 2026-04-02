# src/train.py

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from transformers import get_linear_schedule_with_warmup

from torch.optim import AdamW

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from tqdm import tqdm
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

from preprocess import get_tokenizer, create_dataloaders
from data_loader import load_all_data
from model import CulturalSarcasmDetector, CSDLoss, get_model, save_model

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()


def compute_metrics(all_labels, all_preds):
    """
    Computes accuracy and F1 score given true labels and predictions.
    """
    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):

    model.train()

    total_loss     = 0.0
    total_sarc_loss= 0.0
    total_dom_loss = 0.0
    all_labels     = []
    all_preds      = []

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
        leave=True
    )

    for batch in progress_bar:

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        domains        = batch["domain"].to(device)

        optimizer.zero_grad()

        with autocast():

            sarcasm_logits, domain_logits = model(input_ids, attention_mask)

            loss, sarc_loss, dom_loss = criterion(
                sarcasm_logits, domain_logits, labels, domains
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss      += loss.item()
        total_sarc_loss += sarc_loss.item()
        total_dom_loss  += dom_loss.item()

        preds = torch.argmax(sarcasm_logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr":   f"{current_lr:.2e}"
        })

    n_batches = len(loader)
    metrics = compute_metrics(all_labels, all_preds)

    return {
        "loss":      total_loss      / n_batches,
        "sarc_loss": total_sarc_loss / n_batches,
        "dom_loss":  total_dom_loss  / n_batches,
        "accuracy":  metrics["accuracy"],
        "f1":        metrics["f1"],
    }


def evaluate(model, loader, criterion, device, split_name="Val"):

    model.eval()

    total_loss      = 0.0
    total_sarc_loss = 0.0
    total_dom_loss  = 0.0
    all_labels      = []
    all_preds       = []
    all_sarc_probs  = []

    with torch.no_grad():

        progress_bar = tqdm(loader, desc=f"[{split_name}]", leave=False)

        for batch in progress_bar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            domains        = batch["domain"].to(device)

            with autocast():
                sarcasm_logits, domain_logits = model(input_ids, attention_mask)
                loss, sarc_loss, dom_loss = criterion(
                    sarcasm_logits, domain_logits, labels, domains
                )

            total_loss      += loss.item()
            total_sarc_loss += sarc_loss.item()
            total_dom_loss  += dom_loss.item()

            sarc_probs = torch.softmax(sarcasm_logits, dim=1)

            preds = torch.argmax(sarcasm_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_sarc_probs.extend(sarc_probs[:, 1].cpu().numpy())

    n_batches = len(loader)
    metrics   = compute_metrics(all_labels, all_preds)

    return {
        "loss":       total_loss      / n_batches,
        "sarc_loss":  total_sarc_loss / n_batches,
        "dom_loss":   total_dom_loss  / n_batches,
        "accuracy":   metrics["accuracy"],
        "f1":         metrics["f1"],
        "labels":     all_labels,
        "preds":      all_preds,
        "sarc_probs": all_sarc_probs,
    }


def train():

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: GPU not found, training on CPU (will be very slow)")

    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("\nLoading data...")
    train_df, val_df, test_df, domain_to_idx = load_all_data()

    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    print("\nBuilding model...")
    model     = get_model(device)
    criterion = CSDLoss().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    scaler = GradScaler(enabled=USE_FP16)

    best_val_f1   = 0.0
    best_epoch    = 0
    history       = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Total steps: {total_steps:,}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler, device, epoch
        )

        val_metrics = evaluate(model, val_loader, criterion, device, "Val")

        epoch_time = time.time() - epoch_start

        epoch_log = {
            "epoch":         epoch + 1,
            "train_loss":    round(train_metrics["loss"],     4),
            "train_sarc_loss":round(train_metrics["sarc_loss"],4),
            "train_acc":     round(train_metrics["accuracy"], 4),
            "train_f1":      round(train_metrics["f1"],       4),
            "val_loss":      round(val_metrics["loss"],       4),
            "val_sarc_loss": round(val_metrics["sarc_loss"],  4),
            "val_acc":       round(val_metrics["accuracy"],   4),
            "val_f1":        round(val_metrics["f1"],         4),
            "epoch_time_sec":round(epoch_time,                1),
        }
        history.append(epoch_log)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.0f}s)")
        print(f"  Train → loss: {epoch_log['train_loss']:.4f}  "
              f"acc: {epoch_log['train_acc']:.4f}  "
              f"f1: {epoch_log['train_f1']:.4f}")
        print(f"  Val   → loss: {epoch_log['val_loss']:.4f}  "
              f"acc: {epoch_log['val_acc']:.4f}  "
              f"f1: {epoch_log['val_f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch  = epoch + 1
            save_model(model, tokenizer)
            print(f"  ✓ New best model saved (val F1: {best_val_f1:.4f})")

        print("-" * 60)

    history_path = os.path.join(LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    print(f"\nBest model was from epoch {best_epoch} (val F1: {best_val_f1:.4f})")
    print("\nRunning final evaluation on test set...")

    from model import load_model
    best_model = load_model(device)
    test_metrics = evaluate(best_model, test_loader, criterion, device, "Test")

    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Loss:     {test_metrics['loss']:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=["Not sarcastic", "Sarcastic"]
    ))

    return history, test_metrics


if __name__ == "__main__":
    history, test_metrics = train()
    print("\nTraining complete.")