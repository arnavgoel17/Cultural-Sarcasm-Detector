# src/data_loader.py

import pandas as pd  
import json  
import os 
import numpy as np 
from sklearn.model_selection import train_test_split  
from collections import Counter  

# We import our config so we never hardcode any path or setting here
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

# SUBREDDIT → DOMAIN MAPPING
# This dictionary maps Reddit community names to 7 domain categories
# and fall back to "general" for anything we don't recognize

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
    "AskReddit": "humor",
    "tifu": "humor",
    "hmmm": "humor",
    "WTF": "humor",
}

# NEWS URL → DOMAIN MAPPING
# We parse the URL to figure out the domain

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
    Example:
        url = "https://huffpost.com/entry/politics/trump-rally"
        → returns "politics"
    """
    if not isinstance(url, str):
        # checks if url is not of type str return "general"
        return "general"

    url_lower = url.lower()

    for keyword, domain in URL_KEYWORD_TO_DOMAIN.items():
        # check if any keyword appears anywhere in the URL
        if keyword in url_lower:
            return domain

    return "general"  # default "general"


def subreddit_to_domain(subreddit):
    """
    Converts subreddit to domain
    """
    if not isinstance(subreddit, str):
        return "general" # default general

    return SUBREDDIT_TO_DOMAIN.get(subreddit.strip(), "general")

def load_headlines(path=HEADLINES_PATH):
    """
    Reads the Sarcasm_Headlines_Dataset_v2.json file.
    """

    print("Loading headlines dataset...")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        
        for line in f:
            
            line = line.strip()
            
            if line:
                
                records.append(json.loads(line))
                

    df = pd.DataFrame(records)
    # df: "is_sarcastic", "headline", "article_link"

    print(f"  Raw headlines shape: {df.shape}")

    df = df.rename(columns={
        "headline": "text",
        "is_sarcastic": "label"
    })

    if "article_link" in df.columns:
        
        df["domain"] = df["article_link"].apply(url_to_domain)
        
    else:
        df["domain"] = "general"  # default "general"

    df = df[["text", "label", "domain"]]

    df["text"] = df["text"].str.strip()

    df = df.dropna(subset=["text", "label"])

    df = df[df["text"].str.len() > 5]

    df["label"] = df["label"].astype(int)

    print(f"  Clean headlines shape: {df.shape}")
    print(f"  Domain distribution:\n{df['domain'].value_counts()}\n")

    return df


def load_reddit(path=REDDIT_PATH, max_samples=150000):
    """
    Reads the train_balanced_sarcasm.csv file.
    """

    print("Loading Reddit dataset...")

    df = pd.read_csv(
        path,
        usecols=["label", "comment", "subreddit"],    # we want only these columns
        encoding="utf-8",
        on_bad_lines="skip"
    )

    print(f"  Raw Reddit shape: {df.shape}")

    samples_per_class = max_samples // 2  # balance the dataset

    df_sarcastic = df[df["label"] == 1].sample(
        n=min(samples_per_class, (df["label"] == 1).sum()),
        random_state=SEED
    )
    df_not_sarcastic = df[df["label"] == 0].sample(
        n=min(samples_per_class, (df["label"] == 0).sum()),
        random_state=SEED
    )

    df = pd.concat([df_sarcastic, df_not_sarcastic], ignore_index=True)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    df = df.rename(columns={"comment": "text"})

    df["domain"] = df["subreddit"].apply(subreddit_to_domain)

    df = df[["text", "label", "domain"]]

    df["text"] = df["text"].str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 5]
    df["label"] = df["label"].astype(int)

    df = df[~df["text"].isin(["[deleted]", "[removed]", ""])]

    print(f"  Clean Reddit shape: {df.shape}")
    print(f"  Domain distribution:\n{df['domain'].value_counts()}\n")

    return df

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

    print(f"Combined dataset shape: {df.shape}")

    domain_to_idx = {domain: idx for idx, domain in enumerate(DOMAIN_LABELS)}

    df["domain_idx"] = df["domain"].map(domain_to_idx)

    df["domain_idx"] = df["domain_idx"].fillna(DOMAIN_LABELS.index("general"))

    label_counts = Counter(df["label"])
    print(f"Label balance → sarcastic: {label_counts[1]}, not sarcastic: {label_counts[0]}")

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT, 
        random_state=SEED,
        stratify=df["label"] 
    )
    relative_val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=SEED,
        stratify=train_val_df["label"]
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

    return train_df, val_df, test_df, domain_to_idx

if __name__ == "__main__":

    train_df, val_df, test_df, domain_to_idx = load_all_data()

    print("\nDomain mapping:", domain_to_idx)
    print("\nSample rows from training set:")
    print(train_df.sample(5, random_state=42).to_string())
