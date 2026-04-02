# src/model.py

import torch
import torch.nn as nn

from transformers import DistilBertModel

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class ClassificationHead(nn.Module):
    """
    A small neural network that takes a 768-dim vector
    and outputs logits for N classes.

    Used twice in our model:
      - Once for sarcasm  (N=2: sarcastic / not sarcastic)
      - Once for domain   (N=7: politics, entertainment, ...)

    Architecture: Linear → ReLU → Dropout → Linear
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

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass i.e. how data flows through this head.

        Parameter:
            x → tensor of shape [batch_size, 768]
        Returns:
            logits → tensor of shape [batch_size, output_dim]
        """
        x = self.fc1(x)       # [batch, 768] → [batch, 256]
        x = self.relu(x)      # [batch, 256] → [batch, 256] 
        x = self.dropout(x)   # [batch, 256] → [batch, 256]
        x = self.fc2(x)       # [batch, 256] → [batch, 2 or 7]
        return x

class CulturalSarcasmDetector(nn.Module):
    """
    The full model:
        1. DistilBERT backbone — encodes input text into rich vectors
        2. Shared dropout — regularizes the [CLS] representation
        3. Sarcasm head — predicts sarcastic (1) or not (0)
        4. Domain head  — predicts one of 7 domain categories
    """

    def __init__(self, num_domains=NUM_DOMAINS, dropout_prob=0.3):
        super().__init__()

        # DistilBERT Backbone
        self.distilbert = DistilBertModel.from_pretrained(PRETRAINED_MODEL)

        # Shared Dropout
        self.dropout = nn.Dropout(p=dropout_prob)

        # Task-Specific Heads
        self.sarcasm_head = ClassificationHead(
            input_dim=768,    
            hidden_dim=256,   
            output_dim=2,    
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

        # Step 1: Run DistilBERT
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Step 2: Extract [CLS] token representation 
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Step 3: Apply shared dropout 
        cls_output = self.dropout(cls_output)

        sarcasm_logits = self.sarcasm_head(cls_output)

        domain_logits = self.domain_head(cls_output)

        return sarcasm_logits, domain_logits

class CSDLoss(nn.Module):
    """
    Combined weighted loss for both tasks.

    Loss = (SARCASM_LOSS_WEIGHT × sarcasm_loss)
         + (DOMAIN_LOSS_WEIGHT  × domain_loss)
         = 0.7 × sarcasm_loss + 0.3 × domain_loss
    """

    def __init__(self):
        super().__init__()
        self.sarcasm_loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn  = nn.CrossEntropyLoss()

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

        total_loss = (SARCASM_LOSS_WEIGHT * sarcasm_loss +
                      DOMAIN_LOSS_WEIGHT  * domain_loss)
       
        return total_loss, sarcasm_loss, domain_loss
        
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
    
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def save_model(model, tokenizer, path=MODEL_SAVE_DIR):
    """
    Saves the trained model weights and tokenizer to disk.

    We save both model AND tokenizer together because at inference time
    you need both — the tokenizer to process input text,
    the model to make predictions.
    """
    os.makedirs(path, exist_ok=True)

    # Save model weights
    model_path = os.path.join(path, "csd_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
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

    model = model.to(device)
    print(f"Model loaded from: {model_path}")
    return model

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nBuilding model...")
    model = get_model(device)

    print("\nRunning dummy forward pass...")
    batch_size = 4
    dummy_input_ids      = torch.randint(0, 30522, (batch_size, MAX_SEQ_LENGTH)).to(device)
    dummy_attention_mask = torch.ones(batch_size, MAX_SEQ_LENGTH, dtype=torch.long).to(device)

    model.eval()

    with torch.no_grad():
        sarcasm_logits, domain_logits = model(dummy_input_ids, dummy_attention_mask)

    print(f"Sarcasm logits shape: {sarcasm_logits.shape}")
    # → torch.Size([4, 2])
    print(f"Domain logits shape:  {domain_logits.shape}")
    # → torch.Size([4, 7])

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

    print("\nModel architecture check passed!")