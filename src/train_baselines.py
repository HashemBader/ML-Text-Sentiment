import pandas as pd
import numpy as np
import re
import string
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# Constants
DATA_PATH = os.path.join("data", "imdb_dataset_cleaned.csv")
RANDOM_STATE = 42

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Please run 'python3 src/data.py' first to generate cleaned data.")
    return pd.read_csv(path)

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("IMDB_Baselines")

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Ensure no NaN values
    df = df.dropna(subset=['review', 'sentiment'])
    
    X = df['review']
    y = df['sentiment']

    # Train-test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    with mlflow.start_run(run_name="LogisticRegression_Pipeline"):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])

        print("Running Cross-Validation...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_accuracy = np.mean(cv_scores)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {mean_cv_accuracy:.4f}")

        print("Training model on full training set...")
        pipeline.fit(X_train, y_train)

        print("Evaluating on test set...")
        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Classification Report:\n", report)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        mlflow.sklearn.log_model(pipeline, "model")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/classification_logreg.joblib")
        print("Model saved to models/classification_logreg.joblib")
        
        joblib.dump(pipeline.named_steps['tfidf'], "models/tfidf_vectorizer.joblib")
        print("Vectorizer saved to models/tfidf_vectorizer.joblib")

if __name__ == "__main__":
    main()
