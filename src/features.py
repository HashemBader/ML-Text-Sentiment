"""
features.py
- Implements TF-IDF vectorization
- Saves/Loads vectorizer artifact
- Transforms text data to sparse matrices
"""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import ensure_dir

def fit_vectorizer(train_csv_path, save_path, text_col="review", max_features=5000):
    """
    Fits a TF-IDF vectorizer on the training data and saves it.
    """
    ensure_dir(os.path.dirname(save_path))
    df = pd.read_csv(train_csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    vectorizer.fit(texts)
    
    joblib.dump(vectorizer, save_path)
    print(f"Vectorizer fitted on {len(texts)} documents and saved to {save_path}")
    return vectorizer

def load_vectorizer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}. Please fit it first.")
    return joblib.load(path)

def transform_csv(csv_path, vectorizer_path, text_col="review"):
    """
    Loads a CSV, transforms the text column using the saved vectorizer.
    Returns: (X_sparse_matrix, original_dataframe)
    """
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()
    
    vectorizer = load_vectorizer(vectorizer_path)
    X = vectorizer.transform(texts)
    return X, df
