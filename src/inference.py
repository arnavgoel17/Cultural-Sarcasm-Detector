# src/inference.py

import torch
import torch.nn.functional as F
# F contains functional versions of neural network operations
# F.softmax() is identical to nn.Softmax() but applied as a function
# Preferred for one-off operations (not layers used repeatedly)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from preprocess import get_tokenizer
from model import load_model


# ─────────────────────────────────────────────────────────────────────
# CORE INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────

def predict(text, model, tokenizer, device):
    """
    Takes a raw sentence and returns sarcasm score + domain.

    This is the function that makes your model actually USEFUL —
    all the training was building up to this single function.

    Parameters:
        text      → any string (the sentence to analyze)
        model     → loaded CulturalSarcasmDetector
        tokenizer → DistilBERT tokenizer
        device    → cuda or cpu

    Returns:
        dict with:
            sarcasm_score → float 0.0 to 1.0
            is_sarcastic  → bool (True if score > 0.5)
            domain        → string e.g. "politics"
            domain_confidence → float 0.0 to 1.0
            all_domain_scores → dict of all 7 domain probabilities
    """

    model.eval()
    # Always set to eval mode before inference
    # Disables dropout → deterministic, reproducible predictions

    with torch.no_grad():
        # No gradients needed — we're just making predictions, not learning

        # ── Tokenize ──
        encoding = tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,   # 128
            padding="max_length",
            truncation=True,
            return_tensors="pt"          # pt = PyTorch tensors
        )
        # Returns dict with input_ids and attention_mask
        # Both shape [1, 128] — batch of 1

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        # Move to same device as the model

        # ── Forward pass ──
        sarcasm_logits, domain_logits = model(input_ids, attention_mask)
        # sarcasm_logits: [1, 2]
        # domain_logits:  [1, 7]

        # ── Sarcasm score ──
        sarcasm_probs = F.softmax(sarcasm_logits, dim=1)
        # F.softmax converts raw logits to probabilities summing to 1
        # dim=1 = apply softmax across the class dimension
        # e.g. [-0.3, 2.1] → [0.09, 0.91]

        sarcasm_score = sarcasm_probs[0, 1].item()
        # [0, 1] = first (only) batch item, class index 1 (sarcastic)
        # .item() converts single-element tensor to Python float

        is_sarcastic = sarcasm_score > 0.5
        # Simple threshold — above 0.5 = sarcastic prediction

        # ── Domain prediction ──
        domain_probs = F.softmax(domain_logits, dim=1)
        # [1, 7] → probabilities for each of our 7 domains

        domain_idx = torch.argmax(domain_probs, dim=1).item()
        # argmax finds index of highest probability
        # .item() converts tensor to Python int

        domain_name = DOMAIN_LABELS[domain_idx]
        # Convert index back to string: 0 → "politics", 1 → "entertainment" etc.

        domain_confidence = domain_probs[0, domain_idx].item()
        # Confidence of the predicted domain

        all_domain_scores = {
            DOMAIN_LABELS[i]: round(domain_probs[0, i].item(), 4)
            for i in range(len(DOMAIN_LABELS))
        }
        # Build a dict of ALL domain probabilities for transparency
        # e.g. {"politics": 0.72, "humor": 0.14, "general": 0.08, ...}

    return {
        "sarcasm_score":      round(sarcasm_score, 4),
        "is_sarcastic":       bool(is_sarcastic),
        "domain":             domain_name,
        "domain_confidence":  round(domain_confidence, 4),
        "all_domain_scores":  all_domain_scores,
    }


# ─────────────────────────────────────────────────────────────────────
# PRETTY PRINT HELPER
# ─────────────────────────────────────────────────────────────────────

def print_prediction(text, result):
    """
    Prints a formatted, human-readable version of the prediction.
    Makes your demo look clean and professional.
    """
    bar_length    = 30
    filled        = int(result["sarcasm_score"] * bar_length)
    score_bar     = "█" * filled + "░" * (bar_length - filled)
    # Visual score bar — like a loading bar but for sarcasm level
    # "█" * 24 + "░" * 6  →  ████████████████████████░░░░░░  (score ≈ 0.80)

    verdict = "SARCASTIC" if result["is_sarcastic"] else "SINCERE"
    # Ternary expression — short form of if/else for assignments

    print("\n" + "─" * 60)
    print(f'Input: "{text}"')
    print("─" * 60)
    print(f"Verdict:       {verdict}")
    print(f"Sarcasm score: {score_bar}  {result['sarcasm_score']:.4f}")
    print(f"Domain:        {result['domain'].upper()}  "
          f"(confidence: {result['domain_confidence']:.1%})")
    print("\nAll domain scores:")
    # Sort domains by score descending — highest confidence first
    sorted_domains = sorted(
        result["all_domain_scores"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    # sorted() returns a new sorted list
    # key=lambda x: x[1] means "sort by the second element of each tuple"
    # (each item from .items() is a (domain_name, score) tuple)
    # reverse=True = highest first

    for domain, score in sorted_domains:
        domain_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {domain:<15} {domain_bar}  {score:.4f}")
        # :<15 left-aligns the domain name in a field of width 15
        # So all bars start at the same column — clean alignment

    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────
# BATCH INFERENCE — analyze multiple sentences at once
# ─────────────────────────────────────────────────────────────────────

def predict_batch(texts, model, tokenizer, device):
    """
    Analyzes a list of sentences efficiently in one forward pass.

    More efficient than calling predict() in a loop because
    all sentences are tokenized and processed simultaneously on GPU.

    Parameters:
        texts    → list of strings
        model    → loaded CulturalSarcasmDetector
        tokenizer→ DistilBERT tokenizer
        device   → cuda or cpu

    Returns:
        list of result dicts (same format as predict())
    """

    model.eval()

    with torch.no_grad():
        # Tokenize all texts at once
        encoding = tokenizer(
            texts,                       # list of strings, not just one
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # When you pass a list, tokenizer returns [batch_size, 128] tensors
        # batch_size = len(texts)

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        sarcasm_logits, domain_logits = model(input_ids, attention_mask)

        sarcasm_probs = F.softmax(sarcasm_logits, dim=1)
        domain_probs  = F.softmax(domain_logits,  dim=1)

        # Extract results for each sentence
        results = []
        for i in range(len(texts)):
            sarc_score    = sarcasm_probs[i, 1].item()
            dom_idx       = torch.argmax(domain_probs[i]).item()
            dom_name      = DOMAIN_LABELS[dom_idx]
            dom_conf      = domain_probs[i, dom_idx].item()
            all_dom       = {
                DOMAIN_LABELS[j]: round(domain_probs[i, j].item(), 4)
                for j in range(len(DOMAIN_LABELS))
            }
            results.append({
                "text":               texts[i],
                "sarcasm_score":      round(sarc_score, 4),
                "is_sarcastic":       sarc_score > 0.5,
                "domain":             dom_name,
                "domain_confidence":  round(dom_conf, 4),
                "all_domain_scores":  all_dom,
            })

    return results


# ─────────────────────────────────────────────────────────────────────
# INTERACTIVE DEMO
# ─────────────────────────────────────────────────────────────────────

def interactive_demo():
    """
    Runs an interactive command-line demo.
    Type any sentence → get sarcasm score + domain instantly.
    Type 'quit' to exit.

    This is what you run during your professor demo.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model     = load_model(device)
    tokenizer = get_tokenizer()

    print("\n" + "=" * 60)
    print("  Cultural Sarcasm Detector — Interactive Demo")
    print("  Type a sentence to analyze. Type 'quit' to exit.")
    print("=" * 60)

    # ── Run some built-in examples first ──
    demo_sentences = [
        "Oh sure, because THAT'S a totally normal thing to do.",
        "Scientists discover that water is wet.",
        "The new budget plan will definitely help everyone equally.",
        "Yeah, because Monday mornings are my absolute favourite.",
        "The team played exceptionally well and won the championship.",
        "Oh great, another software update that fixes everything.",
        "I absolutely love sitting in traffic for three hours.",
        "The report shows steady growth in renewable energy adoption.",
    ]

    print("\nRunning built-in examples first...\n")
    batch_results = predict_batch(demo_sentences, model, tokenizer, device)
    for text, result in zip(demo_sentences, batch_results):
        print_prediction(text, result)

    # ── Interactive loop ──
    print("\n\nNow try your own sentences:")
    while True:
        try:
            user_input = input("\nEnter sentence (or 'quit'): ").strip()
            # input() waits for the user to type something and press Enter
            # .strip() removes accidental leading/trailing spaces
        except (EOFError, KeyboardInterrupt):
            # EOFError = user pressed Ctrl+Z (Windows end-of-file)
            # KeyboardInterrupt = user pressed Ctrl+C
            print("\nExiting.")
            break

        if user_input.lower() == "quit":
            # .lower() makes comparison case-insensitive
            # "QUIT", "Quit", "quit" all work
            print("Goodbye!")
            break

        if len(user_input) < 3:
            print("Please enter a longer sentence.")
            continue
            # continue skips the rest of this loop iteration
            # and goes back to the top (asks for input again)

        result = predict(user_input, model, tokenizer, device)
        print_prediction(user_input, result)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    interactive_demo()
