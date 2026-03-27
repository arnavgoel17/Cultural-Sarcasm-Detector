# src/config.py

import os  # For building file paths that work on any OS

# ─────────────────────────────────────────────
# PATHS — where our files live
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath(__file__)  → the full path of THIS file (config.py)
# os.path.dirname(...)       → go one folder up (that's src/)
# So BASE_DIR = the root of your entire project

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_DIR     = os.path.join(BASE_DIR, "models")
LOG_DIR            = os.path.join(BASE_DIR, "logs")
# os.path.join builds paths correctly on both Windows (\) and Linux (/)

HEADLINES_PATH = os.path.join(DATA_RAW_DIR, "Sarcasm_Headlines_Dataset_v2.json")
REDDIT_PATH    = os.path.join(DATA_RAW_DIR, "train-balanced-sarcasm.csv")

# ─────────────────────────────────────────────
# MODEL — what pre-trained model we start from
# ─────────────────────────────────────────────
PRETRAINED_MODEL = "distilbert-base-uncased"
# "uncased" means the model doesn't distinguish "Hello" from "hello"
# This is better for sarcasm — sarcasm doesn't depend on capitalization patterns

MAX_SEQ_LENGTH = 128
# Every sentence gets padded or cut to exactly 128 tokens
# A "token" is roughly a word (sometimes a word-piece)
# 128 is enough for headlines and reddit comments, and saves GPU memory
# (Using 512 would use 4x more memory for little benefit here)

# ─────────────────────────────────────────────
# DOMAIN LABELS — what categories we classify into
# ─────────────────────────────────────────────
DOMAIN_LABELS = [
    "politics",      # 0
    "entertainment", # 1
    "technology",    # 2
    "sports",        # 3
    "lifestyle",     # 4
    "humor",         # 5
    "general",       # 6
]
# These are the 7 domains our model will output
# We'll map Reddit subreddits and news article sources to these
NUM_DOMAINS = len(DOMAIN_LABELS)  # = 7

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE = 32
# Batch size = how many examples we feed the model at once per training step
# Larger batch = faster training but more GPU memory
# 32 is safe for your 6GB GPU with DistilBERT

LEARNING_RATE = 2e-5
# 2e-5 = 0.00002 — how big each "learning step" is
# Too high → model overshoots and learns garbage
# Too low → model learns extremely slowly
# 2e-5 is the standard for fine-tuning BERT-family models (from the original paper)

NUM_EPOCHS = 4
# How many times we go through the entire training dataset
# 4 is standard for fine-tuning — enough to learn without memorizing (overfitting)

WARMUP_STEPS = 500
# For the first 500 training steps, we slowly increase the learning rate
# This prevents the model from making huge chaotic updates at the start
# Think of it as warming up before a sprint

WEIGHT_DECAY = 0.01
# A regularization technique — penalizes overly large model weights
# Helps the model generalize to new sentences it hasn't seen

# ─────────────────────────────────────────────
# LOSS WEIGHTS — how much we care about each task
# ─────────────────────────────────────────────
SARCASM_LOSS_WEIGHT = 0.7
DOMAIN_LOSS_WEIGHT  = 0.3
# Our total loss = 0.7 * sarcasm_loss + 0.3 * domain_loss
# We care more about sarcasm accuracy (that's our main task)
# Domain is secondary — it enriches the output but isn't the goal

# ─────────────────────────────────────────────
# GPU / MIXED PRECISION
# ─────────────────────────────────────────────
USE_FP16 = True
# FP16 = 16-bit floating point (half precision)
# Normally PyTorch uses FP32 (32-bit) — numbers stored with more decimal places
# FP16 cuts memory usage in half and is FASTER on modern GPUs like your RTX 3050
# The accuracy loss is negligible for our task

DEVICE = "cuda"
# "cuda" = use your NVIDIA GPU
# If GPU not found, we'll fall back to "cpu" in the training script

# ─────────────────────────────────────────────
# TRAINING SPLIT
# ─────────────────────────────────────────────
TRAIN_SPLIT = 0.85  # 85% of data for training
VAL_SPLIT   = 0.10  # 10% for validation (checking progress during training)
TEST_SPLIT  = 0.05  # 5% for final testing (professor's "unseen data" check)
# These must sum to 1.0

# ─────────────────────────────────────────────
# RANDOM SEED — for reproducibility
# ─────────────────────────────────────────────
SEED = 42
# Setting a fixed seed means: if you run training twice,
# you get the exact same random shuffles, same splits, same results
# Without this, every run gives slightly different results — embarrassing in demos