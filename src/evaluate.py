# src/evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

from sklearn.metrics import (
    confusion_matrix,        # table of true vs predicted labels
    classification_report,   # precision, recall, F1 per class
    roc_curve,               # ROC curve data points
    auc                      # Area Under Curve — single number summarizing ROC
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from preprocess import get_tokenizer, create_dataloaders
from data_loader import load_all_data
from model import load_model, CSDLoss
from train import evaluate as run_evaluate
# We're importing the evaluate() function we wrote in train.py
# "as run_evaluate" avoids naming conflict with this file's own functions


# ─────────────────────────────────────────────────────────────────────
# PLOT 1 — Training curves
# ─────────────────────────────────────────────────────────────────────

def plot_training_curves(history_path=None):
    """
    Reads the training_history.json file saved during training
    and plots loss + F1 curves across epochs.

    These curves are the FIRST thing your professor will look at.
    They show:
    - Is the model actually learning? (loss going down)
    - Is it overfitting? (train loss drops but val loss rises)
    - When did it peak? (best epoch)

    Parameters:
        history_path → path to training_history.json
                       defaults to logs/training_history.json
    """
    if history_path is None:
        history_path = os.path.join(LOG_DIR, "training_history.json")

    with open(history_path, "r") as f:
        history = json.load(f)
    # json.load() reads the file and converts JSON → Python list of dicts

    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]
    train_f1    = [h["train_f1"]   for h in history]
    val_f1      = [h["val_f1"]     for h in history]
    train_acc   = [h["train_acc"]  for h in history]
    val_acc     = [h["val_acc"]    for h in history]
    # List comprehensions — compact way to extract one field from every dict in a list
    # [expression for item in iterable] builds a list by running expression for each item

    # ── Create figure with 3 subplots side by side ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # plt.subplots(rows, cols) creates a grid of plots
    # figsize=(18, 5) sets the figure size in inches
    # axes is a list of 3 Axes objects — one per subplot

    fig.suptitle("CSD Model — Training Curves", fontsize=16, fontweight="bold", y=1.02)
    # suptitle = "super title" — title above all subplots

    # ── Subplot 1: Loss ──
    ax = axes[0]
    ax.plot(epochs, train_loss, "o-", label="Train loss", color="#2196F3", linewidth=2)
    ax.plot(epochs, val_loss,   "s-", label="Val loss",   color="#FF5722", linewidth=2)
    # "o-" means plot with circle markers connected by lines
    # "s-" means square markers connected by lines
    ax.set_title("Loss per epoch",    fontsize=13)
    ax.set_xlabel("Epoch",            fontsize=11)
    ax.set_ylabel("Loss",             fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # alpha=0.3 makes the grid lines 30% opaque — subtle, not distracting

    # ── Subplot 2: F1 Score ──
    ax = axes[1]
    ax.plot(epochs, train_f1, "o-", label="Train F1", color="#2196F3", linewidth=2)
    ax.plot(epochs, val_f1,   "s-", label="Val F1",   color="#FF5722", linewidth=2)
    best_epoch = val_f1.index(max(val_f1))
    ax.axvline(
        x=epochs[best_epoch],
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best epoch ({epochs[best_epoch]})"
    )
    # axvline draws a vertical line at a specific x position
    # Marks exactly which epoch produced the best model — clean visual proof
    ax.set_title("F1 Score per epoch", fontsize=13)
    ax.set_xlabel("Epoch",             fontsize=11)
    ax.set_ylabel("F1 Score",          fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    # set_ylim fixes y-axis from 0 to 1 — F1 is always in this range

    # ── Subplot 3: Accuracy ──
    ax = axes[2]
    ax.plot(epochs, train_acc, "o-", label="Train acc", color="#2196F3", linewidth=2)
    ax.plot(epochs, val_acc,   "s-", label="Val acc",   color="#FF5722", linewidth=2)
    ax.set_title("Accuracy per epoch", fontsize=13)
    ax.set_xlabel("Epoch",             fontsize=11)
    ax.set_ylabel("Accuracy",          fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    # tight_layout automatically adjusts spacing so nothing overlaps

    save_path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # dpi=150 → 150 dots per inch — crisp enough for a report
    # bbox_inches="tight" prevents labels from being cut off at edges
    print(f"Training curves saved to: {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# PLOT 2 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(true_labels, pred_labels):
    """
    Plots a confusion matrix for the sarcasm classification task.

    A confusion matrix is a 2×2 table:

                  Predicted NOT    Predicted SARCASTIC
    Actually NOT  [True Negative]  [False Positive]
    Actually SARC [False Negative] [True Positive]

    It answers: "Where exactly is my model making mistakes?"
    - High False Positives → model too trigger-happy (sees sarcasm everywhere)
    - High False Negatives → model too conservative (misses real sarcasm)

    Parameters:
        true_labels → list of actual labels (0/1)
        pred_labels → list of predicted labels (0/1)
    """

    cm = confusion_matrix(true_labels, pred_labels)
    # cm is a 2×2 numpy array:
    # [[TN, FP],
    #  [FN, TP]]

    # Normalize to percentages so absolute dataset size doesn't distort the view
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    # axis=1 means "sum across each row" (each true class)
    # keepdims=True keeps the shape [2,1] so division broadcasts correctly
    # Multiplying by 100 converts 0.87 → 87.0%

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix — Sarcasm Detection", fontsize=14, fontweight="bold")

    labels = ["Not sarcastic", "Sarcastic"]

    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_normalized],
        ["Raw counts", "Normalized (%)"],
        ["d", ".1f"]
        # "d" = integer format, ".1f" = 1 decimal float format
    ):
        sns.heatmap(
            matrix,
            annot=True,          # write numbers inside each cell
            fmt=fmt,             # number format
            cmap="Blues",        # color scheme — darker = higher value
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,      # thin lines between cells
            cbar=True            # show color scale bar on the right
        )
        # sns.heatmap is the cleanest way to visualize a matrix
        # It handles all the color scaling automatically

        ax.set_title(title,           fontsize=12)
        ax.set_xlabel("Predicted",    fontsize=11)
        ax.set_ylabel("True label",   fontsize=11)
        ax.tick_params(axis="x", rotation=15)
        # rotation=15 tilts x-axis labels slightly so they don't overlap

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to: {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# PLOT 3 — ROC Curve
# ─────────────────────────────────────────────────────────────────────

def plot_roc_curve(true_labels, sarc_probs):
    """
    Plots the ROC (Receiver Operating Characteristic) curve.

    The ROC curve shows how well the model separates two classes
    across ALL possible decision thresholds (not just 0.5).

    X-axis: False Positive Rate (how often we falsely call something sarcastic)
    Y-axis: True Positive Rate  (how often we correctly catch sarcasm)

    A perfect model → top-left corner (0% false, 100% true)
    A random model  → diagonal line

    AUC (Area Under Curve):
    - 1.0 = perfect separation
    - 0.5 = random guessing
    - We typically aim for > 0.85 for a good classifier

    Parameters:
        true_labels → list of actual labels (0/1)
        sarc_probs  → list of predicted sarcasm probabilities (0.0 to 1.0)
    """

    fpr, tpr, thresholds = roc_curve(true_labels, sarc_probs)
    # fpr = false positive rates at each threshold
    # tpr = true positive rates at each threshold
    # thresholds = the decision threshold values tested

    roc_auc = auc(fpr, tpr)
    # auc computes the area under the curve using the trapezoidal rule
    # This single number summarizes the ROC curve

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr, tpr,
        color="#2196F3",
        linewidth=2,
        label=f"CSD Model (AUC = {roc_auc:.4f})"
    )
    ax.plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random classifier (AUC = 0.50)"
    )
    # The diagonal line = random guessing baseline
    # Our model should be well above this

    # Shade the area under the curve
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    # fill_between fills the region between the curve and y=0
    # alpha=0.1 makes it very transparent — decorative, not distracting

    ax.set_title("ROC Curve — Sarcasm Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate",           fontsize=11)
    ax.set_ylabel("True Positive Rate",            fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"ROC curve saved to: {save_path}")
    plt.show()
    plt.close()

    return roc_auc


# ─────────────────────────────────────────────────────────────────────
# PLOT 4 — Domain Accuracy Breakdown
# ─────────────────────────────────────────────────────────────────────

def plot_domain_accuracy(true_labels, pred_labels, true_domains):
    """
    Shows sarcasm detection accuracy broken down by domain.

    Answers: "Does the model perform equally across domains,
    or does it struggle with specific ones like technology?"

    This is a unique insight your professor probably won't expect —
    it shows you thought deeply about the cultural aspect of sarcasm.

    Parameters:
        true_labels  → list of actual sarcasm labels (0/1)
        pred_labels  → list of predicted sarcasm labels (0/1)
        true_domains → list of domain indices (0-6)
    """

    domain_results = {domain: {"correct": 0, "total": 0}
                      for domain in DOMAIN_LABELS}
    # Dictionary comprehension — builds {domain_name: {correct:0, total:0}}
    # for every domain in our DOMAIN_LABELS list

    for true, pred, dom_idx in zip(true_labels, pred_labels, true_domains):
        # zip() combines three lists element-by-element
        # Each iteration: true=actual label, pred=predicted label, dom_idx=domain int
        domain_name = DOMAIN_LABELS[dom_idx]
        domain_results[domain_name]["total"] += 1
        if true == pred:
            domain_results[domain_name]["correct"] += 1

    # Compute accuracy per domain
    accuracies = []
    counts     = []
    for domain in DOMAIN_LABELS:
        total = domain_results[domain]["total"]
        if total > 0:
            acc = domain_results[domain]["correct"] / total
        else:
            acc = 0.0
        accuracies.append(acc)
        counts.append(total)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color bars by accuracy — red for low, green for high
    colors = ["#4CAF50" if acc >= 0.80 else
              "#FF9800" if acc >= 0.70 else
              "#F44336" for acc in accuracies]
    # Ternary chain — equivalent to if/elif/else in one line
    # green ≥ 80%, orange ≥ 70%, red < 70%

    bars = ax.bar(DOMAIN_LABELS, accuracies, color=colors, edgecolor="white", linewidth=0.8)
    # ax.bar creates a bar chart
    # DOMAIN_LABELS = x-axis categories
    # accuracies = bar heights

    # Add count labels below bar tops
    for bar, count, acc in zip(bars, counts, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # center of bar
            bar.get_height() + 0.01,              # just above bar top
            f"{acc:.1%}\n(n={count})",            # e.g. "84.3%\n(n=1243)"
            ha="center", va="bottom", fontsize=9
        )
        # .1% formats 0.843 as "84.3%" automatically

    ax.set_title("Sarcasm detection accuracy by domain",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Domain",   fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.axhline(y=np.mean(accuracies), color="navy", linestyle="--",
               alpha=0.6, label=f"Mean: {np.mean(accuracies):.1%}")
    # axhline draws a horizontal reference line at the mean accuracy
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    # axis="y" draws grid lines only on y-axis (not x-axis — looks cleaner)

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "domain_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Domain accuracy plot saved to: {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# PLOT 5 — Sarcasm Score Distribution
# ─────────────────────────────────────────────────────────────────────

def plot_score_distribution(true_labels, sarc_probs):
    """
    Shows the distribution of predicted sarcasm scores (0.0–1.0)
    separately for truly sarcastic vs truly not-sarcastic examples.

    A well-trained model should show:
    - Not sarcastic examples → scores clustered near 0.0
    - Sarcastic examples     → scores clustered near 1.0
    - Good separation between the two distributions

    If they heavily overlap → model is uncertain and needs improvement.
    """

    probs_sarcastic     = [p for p, l in zip(sarc_probs, true_labels) if l == 1]
    probs_not_sarcastic = [p for p, l in zip(sarc_probs, true_labels) if l == 0]
    # List comprehensions with conditions
    # "if l == 1" filters to only keep probabilities where the TRUE label is sarcastic

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        probs_not_sarcastic,
        bins=50,               # 50 bars across the 0–1 range
        alpha=0.6,             # 60% opaque — lets you see overlap
        color="#2196F3",
        label="Not sarcastic",
        density=True           # normalize so area = 1 (makes comparison fair
                               # even if class sizes differ)
    )
    ax.hist(
        probs_sarcastic,
        bins=50,
        alpha=0.6,
        color="#FF5722",
        label="Sarcastic",
        density=True
    )

    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5,
               label="Decision threshold (0.5)")
    # Vertical line at 0.5 — the default boundary
    # Examples above 0.5 → predicted sarcastic
    # Examples below 0.5 → predicted not sarcastic

    ax.set_title("Predicted sarcasm score distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sarcasm score (0 = sincere, 1 = sarcastic)", fontsize=11)
    ax.set_ylabel("Density",                                     fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "score_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Score distribution saved to: {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────

def full_evaluation():
    """
    Runs the complete evaluation pipeline:
    1. Load the best saved model
    2. Run inference on test set
    3. Generate all 5 plots
    4. Print the full classification report
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")

    # ── Load data and model ──
    print("Loading data...")
    train_df, val_df, test_df, domain_to_idx = load_all_data()

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Creating dataloaders...")
    _, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )
    # We only need val and test loaders for evaluation
    # _ is a convention meaning "I don't need this value (train_loader)"

    print("Loading best model...")
    model     = load_model(device)
    criterion = CSDLoss().to(device)

    # ── Run evaluation ──
    print("\nRunning test set evaluation...")
    test_metrics = run_evaluate(model, test_loader, criterion, device, "Test")

    true_labels  = test_metrics["labels"]
    pred_labels  = test_metrics["preds"]
    sarc_probs   = test_metrics["sarc_probs"]

    # ── Print summary ──
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"Loss:      {test_metrics['loss']:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["Not sarcastic", "Sarcastic"]
    ))

    # ── Generate all plots ──
    print("\nGenerating plots...")

    # Plot 1: Training curves (reads from saved JSON)
    history_path = os.path.join(LOG_DIR, "training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(history_path)
    else:
        print("training_history.json not found — skipping training curves plot")

    # Plot 2: Confusion matrix
    plot_confusion_matrix(true_labels, pred_labels)

    # Plot 3: ROC curve
    roc_auc = plot_roc_curve(true_labels, sarc_probs)
    print(f"AUC Score: {roc_auc:.4f}")

    # Plot 4: Domain accuracy
    # We need to re-run test set to collect domain labels
    # (run_evaluate doesn't return domain labels, so we do a quick repass)
    model.eval()
    all_domains = []
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(test_loader, desc="Collecting domain labels", leave=False):
            all_domains.extend(batch["domain"].numpy())

    plot_domain_accuracy(true_labels, pred_labels, all_domains)

    # Plot 5: Score distribution
    plot_score_distribution(true_labels, sarc_probs)

    print("\nAll plots saved to:", LOG_DIR)
    print("Evaluation complete!")


if __name__ == "__main__":
    full_evaluation()