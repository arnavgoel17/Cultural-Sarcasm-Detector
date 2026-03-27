# src/data_loader.py

import pandas as pd  # for reading and manipulating tabular data
import json  # for reading the .json headlines file
import os  # for building file paths
import numpy as np  # for numerical operations like random sampling
from sklearn.model_selection import train_test_split  # splits data into train/val/test
from collections import Counter  # for counting label frequencies (we'll use this to check balance)

# We import our config so we never hardcode any path or setting here
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append adds a folder to Python's "search path"
# When you write "import config", Python looks in sys.path for a file named config.py
# We're adding the root csd/ folder so Python can find config.py
from config import *

# The * means "import every variable defined in config.py"
# So HEADLINES_PATH, DOMAIN_LABELS, etc. are all available here directly


# ─────────────────────────────────────────────────────────────────────
# SUBREDDIT → DOMAIN MAPPING
# This dictionary maps Reddit community names to our 7 domain categories
# We can't cover all 1000+ subreddits, so we cover the most common ones
# and fall back to "general" for anything we don't recognize
# ─────────────────────────────────────────────────────────────────────

SUBREDDIT_TO_DOMAIN = {
    # Politics
    "politics": "politics",
    "worldnews": "politics",
    "news": "politics",
    "worldpolitics": "politics",
    "Conservative": "politics",
    "Liberal": "politics",
    "government": "politics",
    "uspolitics": "politics",

    # Entertainment
    "movies": "entertainment",
    "television": "entertainment",
    "Music": "entertainment",
    "entertainment": "entertainment",
    "celebrity": "entertainment",
    "popculture": "entertainment",
    "books": "entertainment",
    "comicbooks": "entertainment",

    # Technology
    "technology": "technology",
    "tech": "technology",
    "programming": "technology",
    "science": "technology",
    "artificial": "technology",
    "gadgets": "technology",
    "apple": "technology",
    "Android": "technology",

    # Sports
    "sports": "sports",
    "nfl": "sports",
    "nba": "sports",
    "soccer": "sports",
    "baseball": "sports",
    "hockey": "sports",
    "MMA": "sports",
    "tennis": "sports",

    # Lifestyle
    "LifeProTips": "lifestyle",
    "relationship_advice": "lifestyle",
    "food": "lifestyle",
    "cooking": "lifestyle",
    "Fitness": "lifestyle",
    "travel": "lifestyle",
    "fashion": "lifestyle",
    "parenting": "lifestyle",

    # Humor
    "funny": "humor",
    "jokes": "humor",
    "humor": "humor",
    "memes": "humor",
    "AskReddit": "humor",  # AskReddit is very humor/sarcasm heavy
    "tifu": "humor",  # "Today I F***ed Up" — humor subreddit
    "hmmm": "humor",
    "WTF": "humor",
}

# ─────────────────────────────────────────────────────────────────────
# NEWS URL → DOMAIN MAPPING
# The headlines dataset doesn't have a domain column
# BUT the original dataset from Kaggle has a "article_link" column
# with URLs like "https://www.huffpost.com/entry/politics/..."
# We parse the URL to figure out the domain
# ─────────────────────────────────────────────────────────────────────

URL_KEYWORD_TO_DOMAIN = {
    "politics": "politics",
    "world": "politics",
    "congress": "politics",
    "election": "politics",
    "government": "politics",
    "entertainment": "entertainment",
    "celebrity": "entertainment",
    "movies": "entertainment",
    "music": "entertainment",
    "sports": "sports",
    "nfl": "sports",
    "nba": "sports",
    "tech": "technology",
    "science": "technology",
    "technology": "technology",
    "food": "lifestyle",
    "travel": "lifestyle",
    "style": "lifestyle",
    "weird-news": "humor",
    "comedy": "humor",
    "humor": "humor",
}


def url_to_domain(url):
    """
    Takes a URL string and returns a domain label.

    A "function" in Python is a reusable block of code.
    def = define a function
    url_to_domain = the name we give it
    (url) = the input it accepts (called a "parameter")

    The triple-quoted string right after def is a "docstring"
    — it documents what the function does. Good practice always.

    Example:
        url = "https://huffpost.com/entry/politics/trump-rally"
        → returns "politics"
    """
    if not isinstance(url, str):
        # isinstance checks if url is of type str (string)
        # If someone passes None or a number, we return "general" safely
        return "general"

    url_lower = url.lower()
    # .lower() converts "Politics" to "politics"
    # We do this so our keyword matching is case-insensitive

    for keyword, domain in URL_KEYWORD_TO_DOMAIN.items():
        # .items() returns each key-value pair as (keyword, domain)
        # "for keyword, domain in ..." is called "tuple unpacking"
        # We're checking if any keyword appears anywhere in the URL
        if keyword in url_lower:
            return domain
            # As soon as we find a match, we return immediately
            # "return" exits the function and gives back the value

    return "general"  # If no keyword matched, default to "general"


def subreddit_to_domain(subreddit):
    """
    Converts a subreddit name to one of our 7 domain categories.
    Uses our SUBREDDIT_TO_DOMAIN dictionary above.
    Falls back to 'general' if the subreddit isn't in our mapping.
    """
    if not isinstance(subreddit, str):
        return "general"

    # dict.get(key, default) → returns value if key exists, else returns default
    # This is safer than SUBREDDIT_TO_DOMAIN[subreddit] which would crash on unknown keys
    return SUBREDDIT_TO_DOMAIN.get(subreddit.strip(), "general")
    # .strip() removes any accidental leading/trailing spaces from the subreddit name


# ─────────────────────────────────────────────────────────────────────
# LOAD HEADLINES DATASET
# ─────────────────────────────────────────────────────────────────────

def load_headlines(path=HEADLINES_PATH):
    """
    Reads the Sarcasm_Headlines_Dataset_v2.json file.
    Returns a cleaned pandas DataFrame with columns: text, label, domain
    """

    print("Loading headlines dataset...")

    # The file format is "JSON Lines" — each line is a separate JSON object
    # Example line: {"is_sarcastic": 1, "headline": "...", "article_link": "..."}
    # We read line by line and collect into a list of dicts
    records = []
    with open(path, "r", encoding="utf-8") as f:
        # open() opens the file. "r" = read mode. encoding="utf-8" handles special characters.
        # "with" is a context manager — it automatically closes the file when done,
        # even if an error occurs. Always use "with" for file operations.
        for line in f:
            # f is like a book — iterating over it reads one line at a time
            line = line.strip()
            # .strip() removes newline characters (\n) at the end of each line
            if line:
                # Skip empty lines (if any) — empty string is "falsy" in Python
                records.append(json.loads(line))
                # json.loads() converts a JSON string into a Python dictionary
                # "loads" = "load from string" (as opposed to json.load() which reads a file)

    # Convert our list of dicts into a DataFrame
    # A DataFrame is like a table — rows are examples, columns are attributes
    df = pd.DataFrame(records)
    # df now has columns: "is_sarcastic", "headline", "article_link"

    print(f"  Raw headlines shape: {df.shape}")
    # df.shape returns (num_rows, num_columns) — e.g., (28619, 3)
    # f"..." is an f-string — it lets you embed variables directly in a string

    # ── Rename columns to our unified format ──
    df = df.rename(columns={
        "headline": "text",
        "is_sarcastic": "label"
    })
    # We want every dataset to have the same column names: "text", "label", "domain"
    # rename(columns={old_name: new_name}) does this in-place

    # ── Infer domain from article URL ──
    if "article_link" in df.columns:
        # Check if the column exists first — defensive programming
        df["domain"] = df["article_link"].apply(url_to_domain)
        # .apply(function) runs a function on every value in a column
        # So for each URL in article_link, we call url_to_domain(url) and store the result
    else:
        df["domain"] = "general"  # fallback if URL column is missing

    # ── Keep only the columns we need ──
    df = df[["text", "label", "domain"]]
    # Selecting specific columns from a DataFrame uses double brackets [["col1", "col2"]]
    # Single bracket df["text"] gives a Series (one column)
    # Double bracket df[["text"]] gives a DataFrame (still a table, just 1 column)

    # ── Clean the text ──
    df["text"] = df["text"].str.strip()
    # .str gives access to string methods on a whole column at once
    # .str.strip() removes leading/trailing whitespace from every text entry

    df = df.dropna(subset=["text", "label"])
    # dropna() removes rows with missing (NaN) values
    # subset= means "only drop rows where THESE specific columns have missing values"
    # We don't want to drop rows just because domain is missing (we have a fallback)

    df = df[df["text"].str.len() > 5]
    # Filter out rows where text is extremely short (likely garbage data)
    # .str.len() returns the character length of each string
    # df[condition] keeps only rows where condition is True

    # ── Ensure label is integer (0 or 1) ──
    df["label"] = df["label"].astype(int)
    # astype(int) converts the column to integer type
    # JSON numbers sometimes load as floats (1.0 instead of 1) — this fixes that

    print(f"  Clean headlines shape: {df.shape}")
    print(f"  Domain distribution:\n{df['domain'].value_counts()}\n")
    # value_counts() counts how many times each unique value appears — very useful for sanity checks

    return df


# ─────────────────────────────────────────────────────────────────────
# LOAD REDDIT DATASET
# ─────────────────────────────────────────────────────────────────────

def load_reddit(path=REDDIT_PATH, max_samples=150000):
    """
    Reads the train_balanced_sarcasm.csv file.

    max_samples=150000: We cap at 150k rows because the full dataset
    has ~1 million rows — way too much for laptop training.
    We sample 150k while keeping it balanced (equal sarcastic/not).

    Returns a cleaned DataFrame with columns: text, label, domain
    """

    print("Loading Reddit dataset...")

    # pd.read_csv() reads a CSV file directly into a DataFrame
    # usecols= tells pandas to only load specific columns — saves RAM
    # The Reddit CSV has many columns (parent comment, score, author, etc.)
    # We only need: "label" (0/1), "comment" (the text), "subreddit"
    df = pd.read_csv(
        path,
        usecols=["label", "comment", "subreddit"],
        encoding="utf-8",
        on_bad_lines="skip"
        # on_bad_lines="skip" ignores any malformed rows in the CSV
        # Reddit data can have weird characters that break the CSV format
    )

    print(f"  Raw Reddit shape: {df.shape}")

    # ── Balance the dataset ──
    # "Balanced" means equal number of sarcastic (label=1) and non-sarcastic (label=0) examples
    # If we have 700k non-sarcastic and 300k sarcastic, the model learns to just say "not sarcastic"
    # for everything and gets 70% accuracy by being lazy — we don't want that

    samples_per_class = max_samples // 2
    # // is integer division — 150000 // 2 = 75000 (no decimal)
    # So we want 75,000 sarcastic and 75,000 non-sarcastic

    df_sarcastic = df[df["label"] == 1].sample(
        n=min(samples_per_class, (df["label"] == 1).sum()),
        random_state=SEED
    )
    df_not_sarcastic = df[df["label"] == 0].sample(
        n=min(samples_per_class, (df["label"] == 0).sum()),
        random_state=SEED
    )
    # df[df["label"] == 1] filters to only sarcastic rows
    # .sample(n=...) randomly picks n rows
    # min(samples_per_class, total_available) prevents errors if there aren't enough rows
    # random_state=SEED makes sampling reproducible (same rows chosen every run)

    df = pd.concat([df_sarcastic, df_not_sarcastic], ignore_index=True)
    # pd.concat() stacks two DataFrames on top of each other (like gluing tables)
    # ignore_index=True resets row numbers from 0 instead of keeping original indices

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # frac=1 means "sample 100% of the rows" = shuffle the entire DataFrame
    # Without this, all sarcastic rows come first, then all non-sarcastic — bad for training
    # reset_index(drop=True) resets row numbers after shuffling
    # drop=True means "don't add the old index as a new column"

    # ── Rename columns ──
    df = df.rename(columns={"comment": "text"})
    # "label" is already named "label" so we only rename "comment" → "text"

    # ── Map subreddits to domains ──
    df["domain"] = df["subreddit"].apply(subreddit_to_domain)
    # Same .apply() trick as before — runs subreddit_to_domain on each subreddit name

    # ── Keep only unified columns ──
    df = df[["text", "label", "domain"]]

    # ── Clean text ──
    df["text"] = df["text"].str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 5]
    df["label"] = df["label"].astype(int)

    # ── Remove rows where text is "[deleted]" or "[removed]" ──
    # Reddit has placeholder text for deleted comments — useless for training
    df = df[~df["text"].isin(["[deleted]", "[removed]", ""])]
    # .isin([...]) checks if each value is in the given list → returns True/False per row
    # ~ is the "NOT" operator for boolean arrays — flips True↔False
    # So we're keeping rows where text is NOT one of those placeholders

    print(f"  Clean Reddit shape: {df.shape}")
    print(f"  Domain distribution:\n{df['domain'].value_counts()}\n")

    return df


# ─────────────────────────────────────────────────────────────────────
# MERGE BOTH DATASETS
# ─────────────────────────────────────────────────────────────────────

def load_all_data():
    """
    Loads both datasets, merges them, converts domain strings to integers,
    and splits into train / validation / test sets.

    Returns: train_df, val_df, test_df, domain_to_idx (a mapping dict)
    """

    df_headlines = load_headlines()
    df_reddit = load_reddit()

    # ── Merge ──
    df = pd.concat([df_headlines, df_reddit], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Shuffle again after merging so headlines and reddit rows are interleaved
    # We don't want the model to see all headlines first then all reddit — it would overfit

    print(f"Combined dataset shape: {df.shape}")

    # ── Convert domain labels (strings) to integers ──
    # Neural networks can't work with strings like "politics" — they need numbers
    # We build a mapping: {"politics": 0, "entertainment": 1, ...}
    domain_to_idx = {domain: idx for idx, domain in enumerate(DOMAIN_LABELS)}
    # This is a "dictionary comprehension" — a compact way to build a dict
    # enumerate(DOMAIN_LABELS) gives (0, "politics"), (1, "entertainment"), etc.
    # We flip it: domain name → index number

    df["domain_idx"] = df["domain"].map(domain_to_idx)
    # .map(dict) replaces each value in a column using the dictionary
    # "politics" → 0, "entertainment" → 1, etc.

    df["domain_idx"] = df["domain_idx"].fillna(DOMAIN_LABELS.index("general"))
    # .fillna() replaces NaN values with a default
    # If a domain didn't match any label (shouldn't happen but just in case),
    # we default to the index of "general"

    # ── Check class balance before splitting ──
    label_counts = Counter(df["label"])
    print(f"Label balance → sarcastic: {label_counts[1]}, not sarcastic: {label_counts[0]}")
    # Counter is like value_counts() but works on plain Python lists too

    # ── Train / Val / Test split ──
    # We split TWICE using sklearn's train_test_split
    # First split: separate test from the rest
    # Second split: separate val from train

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT,  # 5% goes to test
        random_state=SEED,
        stratify=df["label"]  # stratify ensures both splits have same label ratio
        # Without stratify, by bad luck you might get 60% sarcastic in train, 40% in test
        # stratify= guarantees the ratio is preserved
    )

    # From the remaining 95%, split into train and val
    # val_size relative to train_val: if total=100, test=5, remaining=95
    # we want val=10% of total = 10/95 ≈ 0.1053 of remaining
    relative_val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=SEED,
        stratify=train_val_df["label"]
    )

    # Reset indices — after splitting, row numbers are scrambled
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

    return train_df, val_df, test_df, domain_to_idx


# ─────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK — run this file directly to test it
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # "__main__" is a special Python variable
    # This block ONLY runs when you execute THIS file directly:
    #   python src/data_loader.py
    # It does NOT run when another file imports data_loader
    # Perfect for quick testing without affecting the rest of the project

    train_df, val_df, test_df, domain_to_idx = load_all_data()

    print("\nDomain mapping:", domain_to_idx)
    print("\nSample rows from training set:")
    print(train_df.sample(5, random_state=42).to_string())
    # .to_string() prints the full DataFrame without truncating columns