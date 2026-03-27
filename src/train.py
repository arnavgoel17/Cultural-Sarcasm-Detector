# src/train.py

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
# GradScaler and autocast are PyTorch's mixed precision tools
# autocast  → automatically runs operations in FP16 where safe, FP32 where needed
# GradScaler → prevents a problem called "gradient underflow" in FP16
#              (very small gradients become 0 in FP16 — scaler multiplies them
#               up before backward pass, then divides back before weight update)

from transformers import get_linear_schedule_with_warmup
# A learning rate scheduler from Hugging Face
# It controls HOW the learning rate changes during training:
#   - Warmup phase: LR slowly increases from 0 to 2e-5 (first 500 steps)
#   - Decay phase: LR linearly decreases from 2e-5 back to 0 (rest of training)
# This is the exact schedule used in the original BERT paper

from torch.optim import AdamW
# AdamW = Adam optimizer with Weight Decay fix
# Adam is the most popular optimizer for deep learning
# It adapts the learning rate individually for each parameter
# based on the history of its gradients
# "W" = weight decay is applied correctly (the original Adam had a bug with it)

import numpy as np
from sklearn.metrics import (
    accuracy_score,       # correct / total
    f1_score,             # harmonic mean of precision and recall
    classification_report # full breakdown per class
)
from tqdm import tqdm     # progress bars
import os
import sys
import json               # for saving training history to a file
import time               # for measuring epoch duration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Import everything we built
from preprocess import get_tokenizer, create_dataloaders
from data_loader import load_all_data
from model import CulturalSarcasmDetector, CSDLoss, get_model, save_model

# Windows fix — prevents multiprocessing errors when DataLoader spawns workers
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()


# ─────────────────────────────────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds):
    """
    Computes accuracy and F1 score given true labels and predictions.

    WHY F1 and not just accuracy?
    If 90% of examples are "not sarcastic", a dumb model that always
    predicts "not sarcastic" gets 90% accuracy — but it's useless.
    F1 score accounts for both precision and recall:
      Precision = of all things I called sarcastic, how many actually were?
      Recall    = of all actually sarcastic things, how many did I catch?
      F1        = harmonic mean of both → punishes lopsided models

    Parameters:
        all_labels → list of true integer labels
        all_preds  → list of predicted integer labels
    Returns:
        dict with accuracy and f1
    """
    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")
    # average="weighted" accounts for class imbalance
    # it weights each class's F1 by how many examples it has
    return {"accuracy": accuracy, "f1": f1}


# ─────────────────────────────────────────────────────────────────────
# ONE EPOCH OF TRAINING
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    """
    Runs one complete pass through the training data.

    "One epoch" = the model sees every training example exactly once.
    With 149,365 examples and batch_size=32, one epoch = 4,668 steps.

    Parameters:
        model     → our CulturalSarcasmDetector
        loader    → training DataLoader (yields batches)
        criterion → CSDLoss (our combined loss function)
        optimizer → AdamW (updates weights)
        scheduler → learning rate scheduler
        scaler    → GradScaler for FP16
        device    → cuda or cpu
        epoch     → current epoch number (for display)

    Returns:
        dict with average losses and metrics for this epoch
    """

    model.train()
    # .train() puts model in training mode
    # This ENABLES Dropout (neurons randomly zeroed)
    # and enables BatchNorm running stats updates (we don't use BN but good habit)
    # Always call .train() before training loop, .eval() before validation

    total_loss     = 0.0
    total_sarc_loss= 0.0
    total_dom_loss = 0.0
    all_labels     = []
    all_preds      = []
    # We accumulate losses and predictions across all batches
    # then average at the end to get epoch-level metrics

    # tqdm wraps our loader with a live progress bar
    # desc= sets the label on the left of the bar
    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
        leave=True
    )

    for batch in progress_bar:
        # Each "batch" is a dict with keys:
        # input_ids, attention_mask, label, domain
        # Each value is a tensor of shape [32, ...] (32 = batch size)

        # ── Move batch to GPU ──
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        domains        = batch["domain"].to(device)
        # .to(device) copies tensors from CPU RAM to GPU VRAM
        # The model is already on GPU — data must be on the same device
        # Otherwise PyTorch throws a "device mismatch" error

        # ── Zero gradients ──
        optimizer.zero_grad()
        # CRITICAL: PyTorch ACCUMULATES gradients by default
        # If you don't zero them, gradients from batch 1 add to batch 2's gradients
        # This would be like baking a new loaf of bread but adding the burnt
        # crust from yesterday's loaf — completely wrong
        # Must be called at the START of every training step

        # ── Forward pass with FP16 autocast ──
        with autocast():
            # autocast automatically decides which operations to run in FP16 vs FP32
            # Matrix multiplications (the expensive parts) → FP16 (fast, memory-efficient)
            # Softmax, loss computation → FP32 (needs precision)
            # You don't have to think about which is which — autocast handles it

            sarcasm_logits, domain_logits = model(input_ids, attention_mask)
            # Forward pass: data flows through DistilBERT → CLS → dropout → both heads
            # sarcasm_logits: [32, 2]
            # domain_logits:  [32, 7]

            loss, sarc_loss, dom_loss = criterion(
                sarcasm_logits, domain_logits, labels, domains
            )
            # loss = 0.7 × cross_entropy(sarcasm_logits, labels)
            #      + 0.3 × cross_entropy(domain_logits, domains)

        # ── Backward pass ──
        scaler.scale(loss).backward()
        # scaler.scale(loss) multiplies the loss by a large scale factor (e.g. 65536)
        # This prevents gradients from becoming 0 in FP16 (underflow)
        # .backward() computes gradients for all 67M parameters
        # PyTorch does this automatically using the chain rule of calculus
        # Each weight gets: gradient = how much did I cause this loss?

        # ── Gradient clipping ──
        scaler.unscale_(optimizer)
        # Unscale gradients back to normal magnitude before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Clips gradients so no single gradient is larger than 1.0
        # WHY? Occasionally a batch produces extreme gradients ("exploding gradients")
        # that would cause a catastrophically large weight update and break training
        # Clipping caps the norm of the gradient vector to 1.0 → stable training

        # ── Optimizer step ──
        scaler.step(optimizer)
        # scaler checks: were there any inf/nan gradients? (can happen in FP16)
        # If yes → skip this update (don't corrupt the model)
        # If no  → calls optimizer.step() which updates all 67M weights:
        #          weight = weight - learning_rate × gradient

        scaler.update()
        # Updates the scale factor for next iteration
        # If no inf/nan → scale stays same or increases (we can be bolder)
        # If inf/nan occurred → scale is halved (be more conservative)

        scheduler.step()
        # Advance the learning rate schedule by one step
        # During warmup: LR increases
        # After warmup: LR gradually decreases toward 0

        # ── Accumulate metrics ──
        total_loss      += loss.item()
        total_sarc_loss += sarc_loss.item()
        total_dom_loss  += dom_loss.item()
        # .item() converts a GPU tensor scalar to a Python float
        # We must do this to move the value off the GPU and free that memory

        # Get predictions for accuracy tracking
        preds = torch.argmax(sarcasm_logits, dim=1)
        # torch.argmax finds the index of the highest logit
        # For sarcasm_logits = [[-0.3, 1.8], [2.1, -0.5], ...]
        # argmax along dim=1 → [1, 0, ...] (1=sarcastic, 0=not)
        # dim=1 means "find max along the class dimension" (not the batch dimension)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        # .cpu() moves tensor back to CPU (required before .numpy())
        # .numpy() converts tensor to numpy array
        # .extend() adds all elements to our running list

        # ── Update progress bar ──
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr":   f"{current_lr:.2e}"
        })
        # set_postfix() shows live stats on the right of the progress bar
        # You'll see: "loss: 0.4231  lr: 1.85e-05" updating each step

    # ── Epoch summary ──
    n_batches = len(loader)
    metrics = compute_metrics(all_labels, all_preds)

    return {
        "loss":      total_loss      / n_batches,
        "sarc_loss": total_sarc_loss / n_batches,
        "dom_loss":  total_dom_loss  / n_batches,
        "accuracy":  metrics["accuracy"],
        "f1":        metrics["f1"],
    }
    # We divide by n_batches to get average loss per batch
    # (not total loss which grows with dataset size)


# ─────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device, split_name="Val"):
    """
    Runs inference on validation or test set — NO weight updates.

    Validation happens after every epoch to check:
    "Is the model actually getting better, or just memorizing training data?"

    If training loss goes down but validation loss goes UP → overfitting
    (model memorized training data, fails on new examples)

    Parameters:
        model      → CulturalSarcasmDetector
        loader     → val or test DataLoader
        criterion  → CSDLoss
        device     → cuda or cpu
        split_name → "Val" or "Test" (just for display)

    Returns:
        dict with losses and metrics
    """

    model.eval()
    # .eval() disables Dropout — all neurons active
    # We want deterministic, reproducible predictions during evaluation

    total_loss      = 0.0
    total_sarc_loss = 0.0
    total_dom_loss  = 0.0
    all_labels      = []
    all_preds       = []
    all_sarc_probs  = []
    # all_sarc_probs stores the actual probability scores (0.0 to 1.0)
    # This is what we show as the "sarcasm score" at inference time

    with torch.no_grad():
        # torch.no_grad() disables gradient tracking entirely
        # During validation we NEVER call .backward() so gradients are wasteful
        # This reduces memory usage by ~50% and speeds up inference 2x
        # Think of it as "read-only mode" for the model

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

            # Convert logits to probabilities using softmax
            sarc_probs = torch.softmax(sarcasm_logits, dim=1)
            # softmax converts raw logits to probabilities that sum to 1
            # [[-0.3, 1.8]] → [[0.12, 0.88]]
            # The second column (index 1) = probability of being sarcastic

            preds = torch.argmax(sarcasm_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_sarc_probs.extend(sarc_probs[:, 1].cpu().numpy())
            # [:, 1] selects the probability of class 1 (sarcastic) for every example

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


# ─────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────

def train():
    """
    Orchestrates the full training pipeline:
    1. Load data
    2. Build model, optimizer, scheduler
    3. Train for NUM_EPOCHS epochs
    4. Save best model (by validation F1)
    5. Log all metrics to a JSON file
    """

    # ── Setup ──
    torch.manual_seed(SEED)
    # Sets PyTorch's random seed for reproducibility
    # Ensures weight initialization and dropout masks are identical each run

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        # Also seed the GPU random number generator
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

    # ── Load Data ──
    print("\nLoading data...")
    train_df, val_df, test_df, domain_to_idx = load_all_data()

    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    # ── Build Model ──
    print("\nBuilding model...")
    model     = get_model(device)
    criterion = CSDLoss().to(device)
    # Move loss function to GPU too — it contains no learnable parameters
    # but it's cleaner to have everything on the same device

    # ── Optimizer ──
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,      # 2e-5 from config
        weight_decay=WEIGHT_DECAY  # 0.01 from config
    )
    # AdamW tracks two "moment" values per parameter:
    # m1 = exponential moving average of gradients (direction)
    # m2 = exponential moving average of squared gradients (magnitude)
    # The update rule adapts LR per parameter: params with noisy gradients
    # get smaller updates; params with consistent gradients get larger updates

    # ── Learning Rate Scheduler ──
    total_steps = len(train_loader) * NUM_EPOCHS
    # len(train_loader) = number of batches per epoch = 4,668
    # total_steps = 4,668 × 4 = 18,672

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,   # 500 steps of increasing LR
        num_training_steps=total_steps    # then linearly decay to 0
    )

    # ── FP16 Scaler ──
    scaler = GradScaler(enabled=USE_FP16)
    # enabled=USE_FP16 → if USE_FP16=False, scaler becomes a no-op
    # (does nothing but doesn't crash — lets us toggle FP16 from config easily)

    # ── Training State ──
    best_val_f1   = 0.0
    best_epoch    = 0
    history       = []
    # history stores metrics from every epoch so we can plot them later

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Total steps: {total_steps:,}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print("=" * 60)

    # ── Epoch Loop ──
    for epoch in range(NUM_EPOCHS):
        # range(4) → 0, 1, 2, 3
        epoch_start = time.time()

        # ── Train ──
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler, device, epoch
        )

        # ── Validate ──
        val_metrics = evaluate(model, val_loader, criterion, device, "Val")

        epoch_time = time.time() - epoch_start

        # ── Log epoch results ──
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

        # ── Save best model ──
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch  = epoch + 1
            save_model(model, tokenizer)
            print(f"  ✓ New best model saved (val F1: {best_val_f1:.4f})")
        # We save the model ONLY when validation F1 improves
        # This is called "early stopping by best checkpoint"
        # Even if the final epoch is slightly worse, we keep the best one

        print("-" * 60)

    # ── Save training history ──
    history_path = os.path.join(LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # ── Final test evaluation ──
    print(f"\nBest model was from epoch {best_epoch} (val F1: {best_val_f1:.4f})")
    print("\nRunning final evaluation on test set...")

    # Reload best saved model for test evaluation
    from model import load_model
    best_model = load_model(device)
    test_metrics = evaluate(best_model, test_loader, criterion, device, "Test")

    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Loss:     {test_metrics['loss']:.4f}")

    # Full classification report
    print("\nDetailed classification report:")
    print(classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=["Not sarcastic", "Sarcastic"]
    ))
    # Shows precision, recall, F1 separately for each class
    # This is exactly what your professor wants to see

    return history, test_metrics


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    history, test_metrics = train()
    print("\nTraining complete.")