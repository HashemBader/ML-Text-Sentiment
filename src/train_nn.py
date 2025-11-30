import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
import string

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
    mlflow.set_experiment("IMDB_Sentiment_Analysis_NN")

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Ensure no NaN values
    df = df.dropna(subset=['review', 'sentiment'])
    
    X = df['review']
    y = df['sentiment']

    # Train-test split (70% train, 30% test to match notebook)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # 1. Basic Model Run
    print("\n--- 1. Running Basic Model ---")
    with mlflow.start_run(run_name="MLP_Basic"):
        pipeline_basic = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64,),
                learning_rate_init=0.001,
                alpha=1e-4,
                max_iter=1000,
                random_state=RANDOM_STATE,
                early_stopping=True,
                verbose=True
            ))
        ])
        
        pipeline_basic.fit(X_train, y_train)
        
        y_pred = pipeline_basic.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Basic Model Test Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)
        
        mlflow.log_param("model_type", "MLP_Basic")
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.sklearn.log_model(pipeline_basic, "model_basic")

    # 2. Train Best Model (MLP)
    print("\n--- 2. Training Best Model (MLP) ---")
    with mlflow.start_run(run_name="MLP_Best"):
        # Best parameters found from notebook/experiments
        # hidden_layer_sizes=(64,), alpha=0.0001
        
        # Custom training loop to capture validation loss
        print("Training model with custom loop for 40 iterations...")
        
        # Split X_train into train and validation for loss monitoring
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train)
        
        # Fit vectorizer on training part
        tfidf = TfidfVectorizer(max_features=5000)
        X_t_vec = tfidf.fit_transform(X_t)
        X_v_vec = tfidf.transform(X_v)
        
        # Initialize MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),
            learning_rate_init=0.001,
            alpha=0.0001,
            random_state=RANDOM_STATE,
            verbose=False # We will print our own progress
        )
        
        classes = np.unique(y_train)
        train_losses = []
        val_losses = []
        
        from sklearn.metrics import log_loss
        
        for i in range(40):
            mlp.partial_fit(X_t_vec, y_t, classes=classes)
            
            # Calculate losses
            y_t_prob = mlp.predict_proba(X_t_vec)
            y_v_prob = mlp.predict_proba(X_v_vec)
            
            tl = log_loss(y_t, y_t_prob)
            vl = log_loss(y_v, y_v_prob)
            
            train_losses.append(tl)
            val_losses.append(vl)
            
            print(f"Iteration {i+1}/40 - Train Loss: {tl:.4f}, Val Loss: {vl:.4f}")
            
        # Attach custom loss history to the model
        mlp.custom_train_loss_ = train_losses
        mlp.custom_val_loss_ = val_losses
        
        # Reconstruct pipeline for compatibility
        pipeline_best = Pipeline([
            ('tfidf', tfidf),
            ('clf', mlp)
        ])
        
        # Evaluate on original test set
        y_pred_best = pipeline_best.predict(X_test)
        accuracy_best = accuracy_score(y_test, y_pred_best)
        report_best = classification_report(y_test, y_pred_best)
        
        print(f"Best Model Test Accuracy: {accuracy_best:.4f}")
        print("Classification Report:\n", report_best)
        
        mlflow.log_param("model_type", "MLP_Best")
        mlflow.log_param("hidden_layer_sizes", "(64,)")
        mlflow.log_param("alpha", 0.0001)
        mlflow.log_metric("test_accuracy", accuracy_best)
        
        mlflow.sklearn.log_model(pipeline_best, "model_best")
        
        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline_best, "models/mlp_best.joblib")
        print("Best model saved to models/mlp_best.joblib")

if __name__ == "__main__":
    main()