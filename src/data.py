import pandas as pd
import re
import string
import os
import argparse
from sklearn.model_selection import train_test_split

# Constants
RAW_DATA_PATH = "data/imdb_dataset.csv"
CLEANED_DATA_PATH = "data/imdb_dataset_cleaned.csv"
PROCESSED_DIR = "data/processed"
RANDOM_STATE = 42

def clean_text(text):
    """
    Cleans the input text by performing lowercasing, HTML removal,
    punctuation removal, and whitespace normalization.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=RAW_DATA_PATH, help="Path to raw IMDb csv")
    parser.add_argument("--out_dir", type=str, default=PROCESSED_DIR, help="Directory to save processed CSVs")
    args = parser.parse_args()

    print(f"Loading data from {args.input_csv}...")
    if not os.path.exists(args.input_csv):
        print(f"Error: File not found at {args.input_csv}")
        return

    df = pd.read_csv(args.input_csv)
    print(f"Original shape: {df.shape}")

    print("Dropping duplicates...")
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")

    print("Cleaning text...")
    if 'review' in df.columns:
        df['review'] = df['review'].apply(clean_text)
    else:
        print("Error: 'review' column not found.")
        return

    print(f"Saving cleaned dataset to {CLEANED_DATA_PATH}...")
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Splitting and saving to {args.out_dir}...")
    ensure_dir(args.out_dir)

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
