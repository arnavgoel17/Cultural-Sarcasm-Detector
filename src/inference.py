# src/inference.py

import torch
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from preprocess import get_tokenizer
from model import load_model

def predict(text, model, tokenizer, device):
    """
    Takes a raw sentence and returns sarcasm score + domain.

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

    with torch.no_grad():

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,   # 128
            padding="max_length",
            truncation=True,
            return_tensors="pt"          # pt = PyTorch tensors
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Forward pass
        sarcasm_logits, domain_logits = model(input_ids, attention_mask)

        # Sarcasm score
        sarcasm_probs = F.softmax(sarcasm_logits, dim=1)

        sarcasm_score = sarcasm_probs[0, 1].item()

        is_sarcastic = sarcasm_score > 0.5

        # Domain prediction
        domain_probs = F.softmax(domain_logits, dim=1)

        domain_idx = torch.argmax(domain_probs, dim=1).item()

        domain_name = DOMAIN_LABELS[domain_idx]

        domain_confidence = domain_probs[0, domain_idx].item()

        all_domain_scores = {
            DOMAIN_LABELS[i]: round(domain_probs[0, i].item(), 4)
            for i in range(len(DOMAIN_LABELS))
        }

    return {
        "sarcasm_score":      round(sarcasm_score, 4),
        "is_sarcastic":       bool(is_sarcastic),
        "domain":             domain_name,
        "domain_confidence":  round(domain_confidence, 4),
        "all_domain_scores":  all_domain_scores,
    }

def print_prediction(text, result):
    """
    Prints a formatted, human-readable version of the prediction.
    Parameters:
        text   → original input sentence
        result → dict returned by predict()
    """
    bar_length    = 30
    filled        = int(result["sarcasm_score"] * bar_length)
    score_bar     = "█" * filled + "░" * (bar_length - filled)
    

    verdict = "SARCASTIC" if result["is_sarcastic"] else "SINCERE"

    print("\n" + "─" * 60)
    print(f'Input: "{text}"')
    print("─" * 60)
    print(f"Verdict:       {verdict}")
    print(f"Sarcasm score: {score_bar}  {result['sarcasm_score']:.4f}")
    print(f"Domain:        {result['domain'].upper()}  "
          f"(confidence: {result['domain_confidence']:.1%})")
    print("\nAll domain scores:")
    sorted_domains = sorted(
        result["all_domain_scores"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for domain, score in sorted_domains:
        domain_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {domain:<15} {domain_bar}  {score:.4f}")

    print("─" * 60)

def predict_batch(texts, model, tokenizer, device):
    """
    Analyzes a list of sentences efficiently in one forward pass.

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

    print("\n\nNow try your own sentences:")
    while True:
        try:
            user_input = input("\nEnter sentence (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if len(user_input) < 3:
            print("Please enter a longer sentence.")
            continue

        result = predict(user_input, model, tokenizer, device)
        print_prediction(user_input, result)

if __name__ == "__main__":
    interactive_demo()
