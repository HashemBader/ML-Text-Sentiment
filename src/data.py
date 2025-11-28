"""
data.py
- Loads raw IMDb CSV (from Kaggle)
- Cleans minimal issues
- Creates classification target if missing
- Writes processed train/val/test CSVs to out_dir
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import set_seed, ensure_dir

set_seed(42)

DEFAULT_CLASS_COL = "sentiment"     # expected 'positive' / 'negative'
DEFAULT_REG_COL = "rating"          # numeric rating column if present

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def prepare_targets(df: pd.DataFrame, class_col=DEFAULT_CLASS_COL, reg_col=DEFAULT_REG_COL):
    # If classification column is missing but numeric rating exists, derive sentiment.
    if class_col not in df.columns:
        if reg_col in df.columns:
            # threshold: rating >= 7 -> positive (this is a reasonable default)
            df[class_col] = df[reg_col].apply(lambda r: "positive" if float(r) >= 7.0 else "negative")
        else:
            raise ValueError(f"Neither {class_col} nor {reg_col} present in dataframe. Please provide labels.")
    # Ensure regression column exists; if not, attempt to create approximate using rating if possible.
    if reg_col not in df.columns:
        # If numeric rating cannot be found, create placeholder (NaNs) — user may fill later
        df[reg_col] = pd.NA
    return df

import re
import string

def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def minimal_cleaning(df: pd.DataFrame):
    # Basic cleaning — drop rows with empty reviews
    text_col = "review" if "review" in df.columns else df.columns[0]
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    
    # Apply notebook cleaning logic
    print("Applying text cleaning...")
    df[text_col] = df[text_col].apply(clean_text)
    
    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"Dropped {before_dedup - len(df)} duplicates.")
    return df

def split_and_save(df: pd.DataFrame, out_dir: str, test_size=0.2, val_size=0.1, random_state=42):
    ensure_dir(out_dir)
    text_col = "review" if "review" in df.columns else df.columns[0]
    class_col = DEFAULT_CLASS_COL
    # First split off test
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[class_col], random_state=random_state)
    # Then split train/val
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_relative, stratify=train_val[class_col], random_state=random_state)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)
    return train, val, test

def main(args):
    df = load_data(args.input_csv)
    df = prepare_targets(df)
    df = minimal_cleaning(df)
    train, val, test = split_and_save(df, args.out_dir, test_size=args.test_size, val_size=args.val_size, random_state=args.seed)
    print("Saved processed files to", args.out_dir)
    print("Train shape:", train.shape, "Val shape:", val.shape, "Test shape:", test.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to raw IMDb csv")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Directory to save processed CSVs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
