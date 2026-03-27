# src/model.py

import torch
import torch.nn as nn
# torch.nn is PyTorch's neural network module
# "nn" contains everything you need to build a network:
# layers (Linear, Dropout), activation functions (ReLU), loss functions, etc.

from transformers import DistilBertModel
# DistilBertModel gives us the raw DistilBERT backbone
# "raw" means it outputs hidden states (vectors), not predictions
# We build our own prediction heads on top of it

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ─────────────────────────────────────────────────────────────────────
# CLASSIFICATION HEAD — reusable block used by both task heads
# ─────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    A small neural network that takes a 768-dim vector
    and outputs logits for N classes.

    Used twice in our model:
      - Once for sarcasm  (N=2: sarcastic / not sarcastic)
      - Once for domain   (N=7: politics, entertainment, ...)

    WHY inherit from nn.Module?
    nn.Module is PyTorch's base class for all neural network components.
    By inheriting from it:
      - PyTorch automatically tracks all learnable parameters (weights/biases)
      - .to(device) moves all parameters to GPU automatically
      - .parameters() returns all weights so the optimizer can update them
      - .train() and .eval() modes work correctly (affects Dropout)

    Architecture: Linear → ReLU → Dropout → Linear
    This is called a "two-layer MLP" (Multi-Layer Perceptron)
    It's the standard classification head used in BERT-family models
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.3):
        """
        Parameters:
            input_dim   → size of input vector (768 from DistilBERT)
            hidden_dim  → size of intermediate layer (256)
            output_dim  → number of classes to predict (2 or 7)
            dropout_prob→ fraction of neurons randomly zeroed during training
        """
        super().__init__()
        # super().__init__() calls nn.Module's constructor
        # ALWAYS required as the first line in __init__ when inheriting from nn.Module
        # Without it, PyTorch can't track your parameters

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # nn.Linear(in, out) is a fully connected layer
        # It learns a weight matrix of shape [hidden_dim, input_dim]
        # and a bias vector of shape [hidden_dim]
        # Operation: output = input @ weight.T + bias
        # (@ is matrix multiplication in Python)
        # So it transforms a 768-dim vector into a 256-dim vector

        self.relu = nn.ReLU()
        # ReLU = Rectified Linear Unit
        # The activation function: f(x) = max(0, x)
        # It sets all negative values to 0, keeps positive values unchanged
        # WHY? Without activation functions, stacking linear layers is mathematically
        # equivalent to ONE linear layer — they collapse into each other.
        # ReLU introduces non-linearity, letting the network learn complex patterns.

        self.dropout = nn.Dropout(p=dropout_prob)
        # Dropout randomly sets p% of neuron outputs to 0 during training
        # p=0.3 means 30% of neurons are zeroed out each forward pass
        # WHY? It forces the network to not rely on any single neuron
        # Each neuron must independently learn useful features
        # Result: better generalization, less overfitting
        # IMPORTANT: Dropout is automatically DISABLED during evaluation
        # (.eval() mode) — PyTorch handles this for us

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Second linear layer: 256 → 2 (sarcasm) or 256 → 7 (domain)
        # This produces the final "logits" — raw unnormalized scores per class
        # Higher logit = model is more confident about that class

    def forward(self, x):
        """
        Defines the forward pass — how data flows through this head.

        "forward" is a special method in nn.Module.
        When you call head(x), PyTorch automatically calls head.forward(x).
        You NEVER call forward() directly — always use head(x).

        Parameter:
            x → tensor of shape [batch_size, 768]
        Returns:
            logits → tensor of shape [batch_size, output_dim]
        """
        x = self.fc1(x)       # [batch, 768] → [batch, 256]
        x = self.relu(x)      # [batch, 256] → [batch, 256]  (negatives zeroed)
        x = self.dropout(x)   # [batch, 256] → [batch, 256]  (30% zeroed in training)
        x = self.fc2(x)       # [batch, 256] → [batch, 2 or 7]
        return x
        # We do NOT apply softmax here
        # PyTorch's CrossEntropyLoss expects raw logits and applies softmax internally
        # Applying softmax here AND using CrossEntropyLoss would apply it twice — wrong


# ─────────────────────────────────────────────────────────────────────
# MAIN MODEL — DistilBERT + two heads
# ─────────────────────────────────────────────────────────────────────

class CulturalSarcasmDetector(nn.Module):
    """
    The full model:
        1. DistilBERT backbone — encodes input text into rich vectors
        2. Shared dropout — regularizes the [CLS] representation
        3. Sarcasm head — predicts sarcastic (1) or not (0)
        4. Domain head  — predicts one of 7 domain categories

    The backbone is SHARED — both heads see the same DistilBERT output.
    This is the core idea of multi-task learning:
    - The domain task helps DistilBERT learn that "politics sarcasm"
      looks different from "sports sarcasm"
    - The sarcasm task remains the primary objective
    - Shared parameters = shared understanding, less overfitting
    """

    def __init__(self, num_domains=NUM_DOMAINS, dropout_prob=0.3):
        super().__init__()

        # ── DistilBERT Backbone ──
        self.distilbert = DistilBertModel.from_pretrained(PRETRAINED_MODEL)
        # DistilBertModel (not DistilBertForSequenceClassification)
        # We use the BASE model that outputs hidden states
        # "ForSequenceClassification" adds its own head — we want to add OUR heads
        # from_pretrained() loads all 66M pre-trained weights from Hugging Face cache

        # ── Shared Dropout ──
        self.dropout = nn.Dropout(p=dropout_prob)
        # Applied to the [CLS] vector before both heads see it
        # "Shared" means both heads receive the same dropped-out representation
        # This prevents the backbone from overfitting to either task specifically

        # ── Task-Specific Heads ──
        self.sarcasm_head = ClassificationHead(
            input_dim=768,    # DistilBERT always outputs 768-dim vectors
            hidden_dim=256,   # intermediate compression
            output_dim=2,     # 2 classes: not sarcastic (0), sarcastic (1)
            dropout_prob=dropout_prob
        )

        self.domain_head = ClassificationHead(
            input_dim=768,
            hidden_dim=256,
            output_dim=num_domains,  # 7 domain classes
            dropout_prob=dropout_prob
        )

    def forward(self, input_ids, attention_mask):
        """
        Full forward pass through the model.

        Parameters:
            input_ids      → [batch_size, 128] — token IDs
            attention_mask → [batch_size, 128] — 1s for real tokens, 0s for padding

        Returns:
            sarcasm_logits → [batch_size, 2]   — raw sarcasm scores
            domain_logits  → [batch_size, 7]   — raw domain scores
        """

        # ── Step 1: Run DistilBERT ──
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs is a special object with multiple fields
        # outputs.last_hidden_state → [batch_size, 128, 768]
        # This is a 768-dim vector for EVERY token position (128 positions)
        # So for a batch of 32 sentences: shape is [32, 128, 768]

        # ── Step 2: Extract [CLS] token representation ──
        cls_output = outputs.last_hidden_state[:, 0, :]
        # [:, 0, :] means: all batches, position 0, all 768 dims
        # Position 0 is ALWAYS the [CLS] token (token ID 101)
        # DistilBERT is trained so that [CLS] accumulates information
        # from the ENTIRE sentence through attention layers
        # This single 768-dim vector = the model's understanding of the whole sentence
        # Shape after slicing: [batch_size, 768]

        # ── Step 3: Apply shared dropout ──
        cls_output = self.dropout(cls_output)
        # [batch_size, 768] → [batch_size, 768]  (same shape, some values zeroed)

        # ── Step 4: Pass through both heads ──
        sarcasm_logits = self.sarcasm_head(cls_output)
        # [batch_size, 768] → [batch_size, 2]

        domain_logits = self.domain_head(cls_output)
        # [batch_size, 768] → [batch_size, 7]

        return sarcasm_logits, domain_logits
        # We return BOTH — the training loop will compute loss for both
        # and combine them using our weighted formula


# ─────────────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────

class CSDLoss(nn.Module):
    """
    Combined weighted loss for both tasks.

    Loss = (SARCASM_LOSS_WEIGHT × sarcasm_loss)
         + (DOMAIN_LOSS_WEIGHT  × domain_loss)
         = 0.7 × sarcasm_loss + 0.3 × domain_loss

    WHY combine losses?
    The optimizer updates ALL parameters (backbone + both heads) based
    on a single loss value. By combining both losses, we tell the model:
    "optimize for both tasks, but sarcasm matters 2.3x more than domain."

    WHY CrossEntropyLoss?
    CrossEntropyLoss is the standard loss for classification tasks.
    It measures how wrong the model's predicted probabilities are
    compared to the true label.
    Mathematically: -log(probability assigned to the correct class)
    If model says "90% sarcastic" and it IS sarcastic → loss ≈ 0.1 (good)
    If model says "10% sarcastic" and it IS sarcastic → loss ≈ 2.3 (bad)
    """

    def __init__(self):
        super().__init__()
        self.sarcasm_loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn  = nn.CrossEntropyLoss()
        # Two separate CrossEntropyLoss instances — one per task
        # They have the same formula but track gradients independently

    def forward(self, sarcasm_logits, domain_logits, sarcasm_labels, domain_labels):
        """
        Computes the combined weighted loss.

        Parameters:
            sarcasm_logits  → [batch, 2]   model's raw sarcasm predictions
            domain_logits   → [batch, 7]   model's raw domain predictions
            sarcasm_labels  → [batch]      true labels (0 or 1)
            domain_labels   → [batch]      true domain indices (0-6)

        Returns:
            total_loss     → scalar tensor (single number)
            sarcasm_loss   → scalar tensor (for logging)
            domain_loss    → scalar tensor (for logging)
        """
        sarcasm_loss = self.sarcasm_loss_fn(sarcasm_logits, sarcasm_labels)
        domain_loss  = self.domain_loss_fn(domain_logits,  domain_labels)
        # CrossEntropyLoss automatically:
        # 1. Applies softmax to logits (converts to probabilities)
        # 2. Picks the probability of the TRUE class
        # 3. Takes -log of that probability
        # 4. Averages across the batch

        total_loss = (SARCASM_LOSS_WEIGHT * sarcasm_loss +
                      DOMAIN_LOSS_WEIGHT  * domain_loss)
        # 0.7 * sarcasm_loss + 0.3 * domain_loss
        # Both SARCASM_LOSS_WEIGHT and DOMAIN_LOSS_WEIGHT come from config.py

        return total_loss, sarcasm_loss, domain_loss
        # We return all three so training.py can log sarcasm and domain loss
        # separately — very useful for understanding what the model is learning


# ─────────────────────────────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────────────────────────────

def get_model(device):
    """
    Instantiates the model and moves it to the specified device (GPU/CPU).

    Parameters:
        device → torch.device object ("cuda" or "cpu")
    Returns:
        model  → CulturalSarcasmDetector instance on the correct device
    """
    model = CulturalSarcasmDetector()
    model = model.to(device)
    # .to(device) moves ALL model parameters (weights, biases) to the GPU
    # After this, all computations happen on VRAM, not RAM
    # This is required — if model is on CPU but data is on GPU, PyTorch crashes

    # ── Print parameter count (good to show your professor) ──
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    # p.numel() = number of elements in parameter tensor
    # p.requires_grad = True means this parameter will be updated during training
    # All parameters have requires_grad=True by default

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # :, formats numbers with commas — 66,955,010 instead of 66955010

    return model


def save_model(model, tokenizer, path=MODEL_SAVE_DIR):
    """
    Saves the trained model weights and tokenizer to disk.

    We save both model AND tokenizer together because at inference time
    you need both — the tokenizer to process input text,
    the model to make predictions.
    """
    os.makedirs(path, exist_ok=True)
    # exist_ok=True means "don't raise an error if the folder already exists"

    # Save model weights
    model_path = os.path.join(path, "csd_model.pt")
    torch.save(model.state_dict(), model_path)
    # state_dict() returns a dictionary of all parameter tensors
    # {layer_name: weight_tensor, ...}
    # We save this dict, not the whole model object
    # WHY? Saving just weights is safer and more portable than pickling the whole object

    # Save tokenizer
    tokenizer_path = os.path.join(path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    # save_pretrained() saves all tokenizer files (vocab, config, merges)
    # into a folder — the standard Hugging Face format

    print(f"Model saved to:     {model_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")


def load_model(device, path=MODEL_SAVE_DIR):
    """
    Loads a saved model from disk for inference.

    Parameters:
        device → where to load the model (cuda or cpu)
        path   → folder where csd_model.pt was saved
    Returns:
        model  → loaded CulturalSarcasmDetector ready for inference
    """
    model = CulturalSarcasmDetector()
    model_path = os.path.join(path, "csd_model.pt")

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    # torch.load() reads the saved weight dictionary
    # map_location=device ensures weights load onto the correct device
    # (useful if you trained on GPU but run inference on CPU)

    # load_state_dict() copies those weights into our model's layers
    model = model.to(device)
    print(f"Model loaded from: {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nBuilding model...")
    model = get_model(device)

    # ── Dummy forward pass to verify shapes ──
    print("\nRunning dummy forward pass...")
    batch_size = 4
    dummy_input_ids      = torch.randint(0, 30522, (batch_size, MAX_SEQ_LENGTH)).to(device)
    dummy_attention_mask = torch.ones(batch_size, MAX_SEQ_LENGTH, dtype=torch.long).to(device)
    # torch.randint(low, high, shape) → random integers in [low, high)
    # 30522 = DistilBERT vocabulary size
    # torch.ones → all 1s (pretending all tokens are real, no padding)

    model.eval()
    # .eval() disables Dropout for this test
    # In eval mode all neurons are active and outputs are deterministic

    with torch.no_grad():
        # torch.no_grad() tells PyTorch not to track gradients
        # During inference/testing we don't need gradients
        # This saves memory and speeds up computation significantly
        sarcasm_logits, domain_logits = model(dummy_input_ids, dummy_attention_mask)

    print(f"Sarcasm logits shape: {sarcasm_logits.shape}")
    # → torch.Size([4, 2])
    print(f"Domain logits shape:  {domain_logits.shape}")
    # → torch.Size([4, 7])

    # ── Test loss computation ──
    print("\nTesting loss function...")
    criterion = CSDLoss()
    dummy_sarcasm_labels = torch.randint(0, 2, (batch_size,)).to(device)
    dummy_domain_labels  = torch.randint(0, NUM_DOMAINS, (batch_size,)).to(device)

    total_loss, s_loss, d_loss = criterion(
        sarcasm_logits, domain_logits,
        dummy_sarcasm_labels, dummy_domain_labels
    )
    print(f"Total loss:   {total_loss.item():.4f}")
    print(f"Sarcasm loss: {s_loss.item():.4f}")
    print(f"Domain loss:  {d_loss.item():.4f}")
    # .item() converts a single-element tensor to a plain Python float

    print("\nModel architecture check passed!")