# src/config.py

import os 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # the root of your entire project

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_DIR     = os.path.join(BASE_DIR, "models")
LOG_DIR            = os.path.join(BASE_DIR, "logs")

HEADLINES_PATH = os.path.join(DATA_RAW_DIR, "Sarcasm_Headlines_Dataset_v2.json")
REDDIT_PATH    = os.path.join(DATA_RAW_DIR, "train-balanced-sarcasm.csv")

PRETRAINED_MODEL = "distilbert-base-uncased"

MAX_SEQ_LENGTH = 128    # Every sentence gets padded or cut to exactly 128 tokens

DOMAIN_LABELS = [
    "politics",      # 0
    "entertainment", # 1
    "technology",    # 2
    "sports",        # 3
    "lifestyle",     # 4
    "humor",         # 5
    "general",       # 6
]

NUM_DOMAINS = len(DOMAIN_LABELS) 

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SARCASM_LOSS_WEIGHT = 0.7
DOMAIN_LOSS_WEIGHT  = 0.3
USE_FP16 = True
DEVICE = "cuda"
TRAIN_SPLIT = 0.85  # 85% for training
VAL_SPLIT   = 0.10  # 10% for validation
TEST_SPLIT  = 0.05  # 5% for testing

# ─────────────────────────────────────────────
# RANDOM SEED — for reproducibility
# ─────────────────────────────────────────────
SEED = 42
# Setting a fixed seed means: if you run training twice,
# you get the exact same random shuffles, same splits, same results
# Without this, every run gives slightly different results — embarrassing in demos
