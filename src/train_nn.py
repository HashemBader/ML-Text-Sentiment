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

def load_data(path):
    if not os.path.exists(path):
        # Fallback to raw data if cleaned doesn't exist, and clean it
        raw_path = os.path.join("data", "imdb_dataset.csv")
        if os.path.exists(raw_path):
            print(f"Cleaned data not found. Loading and cleaning from {raw_path}...")
            df = pd.read_csv(raw_path)
            df = df.drop_duplicates()
            df['review'] = df['review'].apply(clean_text)
            return df
        else:
            raise FileNotFoundError(f"Data file not found at {path} or {raw_path}.")
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
            ('tfidf', TfidfVectorizer()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.05,
                max_iter=2000,
                random_state=RANDOM_STATE,
                early_stopping=True,
                shuffle=True,
                n_iter_no_change=20
            ))
        ])
        
        pipeline_basic.fit(X_train, y_train)
        y_pred = pipeline_basic.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Basic Model Test Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)
        
        mlflow.log_params(pipeline_basic.get_params())
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log classification report metrics
        mlflow.log_metric("precision_weighted", report_dict['weighted avg']['precision'])
        mlflow.log_metric("recall_weighted", report_dict['weighted avg']['recall'])
        mlflow.log_metric("f1_weighted", report_dict['weighted avg']['f1-score'])
        
        # Log full report as text artifact
        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(pipeline_basic, "model_basic")

    # 2. Random Search
    print("\n--- 2. Running Random Search ---")
    with mlflow.start_run(run_name="MLP_RandomSearch"):
        pipeline_search = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MLPClassifier(
                max_iter=1000, 
                random_state=RANDOM_STATE, 
                activation='relu',
                solver='adam',
                early_stopping=True
            ))
        ])
        
        param_dist = {
            'clf__hidden_layer_sizes': [(64,), (128,), (64, 32)],
            'clf__alpha': [0.0001, 0.001],
        }
        
        random_search = RandomizedSearchCV(
            pipeline_search, 
            param_distributions=param_dist, 
            n_iter=6, 
            cv=3, 
            scoring='f1_macro', 
            n_jobs=-1, 
            verbose=1,
            random_state=RANDOM_STATE
        )
        
        print("Starting Hyperparameter Tuning...")
        random_search.fit(X_train, y_train)
        
        print("\nBest Parameters Found:")
        print(random_search.best_params_)
        print(f"Best Cross-Validation F1 Score: {random_search.best_score_:.4f}")
        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_f1_score", random_search.best_score_)
        
        best_model = random_search.best_estimator_

    # 3. Best Model Evaluation
    print("\n--- 3. Evaluating Best Model ---")
    with mlflow.start_run(run_name="MLP_Best"):
        y_pred_best = best_model.predict(X_test)
        accuracy_best = accuracy_score(y_test, y_pred_best)
        report_best = classification_report(y_test, y_pred_best)
        report_best_dict = classification_report(y_test, y_pred_best, output_dict=True)
        
        print(f"Best Model Test Accuracy: {accuracy_best:.4f}")
        print("Classification Report:\n", report_best)
        
        mlflow.log_params(best_model.get_params())
        mlflow.log_metric("test_accuracy", accuracy_best)
        
        # Log classification report metrics
        mlflow.log_metric("precision_weighted", report_best_dict['weighted avg']['precision'])
        mlflow.log_metric("recall_weighted", report_best_dict['weighted avg']['recall'])
        mlflow.log_metric("f1_weighted", report_best_dict['weighted avg']['f1-score'])
        
        # Log full report as text artifact
        mlflow.log_text(report_best, "classification_report.txt")
        
        mlflow.sklearn.log_model(best_model, "model_best")
        
        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/mlp_best.pkl")
        print("Best model saved to models/mlp_best.pkl")

if __name__ == "__main__":
    main()