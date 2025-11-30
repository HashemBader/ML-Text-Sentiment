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
    Cleans the text as per the notebook logic:
    1. Lowercase
    2. Remove HTML tags
    3. Remove punctuation
    4. Remove extra whitespace
    """
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
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

    # 1. Drop duplicates (as per notebook)
    print("Dropping duplicates...")
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")

    # 2. Clean text (as per notebook)
    print("Cleaning text...")
    if 'review' in df.columns:
        df['review'] = df['review'].apply(clean_text)
    else:
        print("Error: 'review' column not found.")
        return

    # Save cleaned full dataset (as per notebook)
    print(f"Saving cleaned dataset to {CLEANED_DATA_PATH}...")
    df.to_csv(CLEANED_DATA_PATH, index=False)

    # 3. Split into Train/Val/Test (for evaluate.py support)
    # Notebook uses 70/30 split usually, but we need train/val/test for robust pipeline
    # We will use a standard split or match what other scripts expect.
    # train_nn.py expects to split data itself or load from processed.
    # evaluate.py expects test.csv in data/processed.
    
    print(f"Splitting and saving to {args.out_dir}...")
    ensure_dir(args.out_dir)

    # Split into Train (70%), Test (30%) first
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    
    # Further split Train into Train (85% of 70% ≈ 60% total) and Val (15% of 70% ≈ 10% total)
    # Or just keep simple Train/Test if validation is done via CV in scripts.
    # However, standard practice is to have a validation set saved.
    # Let's do a simple Train/Test split save for now as scripts do their own CV/splitting mostly,
    # but evaluate.py specifically needs test.csv.
    
    # Saving
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
