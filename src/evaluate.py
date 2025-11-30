import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Constants
MODELS_DIR = "models"
DATA_DIR = "data/processed"
REPORTS_DIR = "reports"
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH)
    # Drop NaNs if any
    df = df.dropna(subset=['review', 'sentiment'])
    return df['review'], df['sentiment']

def plot_learning_curve(mlp_pipeline, out_path):
    """Plots the loss curve from the MLP classifier."""
    mlp_model = mlp_pipeline.named_steps['clf']
    
    plt.figure(figsize=(8, 6))
    
    if hasattr(mlp_model, 'custom_train_loss_'):
        plt.plot(mlp_model.custom_train_loss_, label='Training Loss', color='tab:blue')
        if hasattr(mlp_model, 'custom_val_loss_'):
            plt.plot(mlp_model.custom_val_loss_, label='Validation Loss', color='tab:orange')
        plt.ylabel("Log Loss")
        plt.xlabel("Iterations")
        plt.title("Neural Network Learning Curve")
        plt.legend()
        plt.grid(True)
        
    elif hasattr(mlp_model, 'loss_curve_'):
        fig, ax1 = plt.subplots(figsize=(8, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(mlp_model.loss_curve_, color=color, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_ is not None:
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Validation Accuracy', color=color)
            ax2.plot(mlp_model.validation_scores_, color=color, label='Validation Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax1.legend(loc='upper right')
        plt.title("Neural Network Learning Curve")
        
    else:
        print("Warning: MLP model does not have loss history. Skipping learning curve plot.")
        return

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved learning curve to {out_path}")

def plot_confusion_matrix_nn(model, X_test, y_test, out_path):
    """Plots confusion matrix for the Neural Network."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Neural Network Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

def plot_feature_importance(logreg_pipeline, out_path, top_n=20):
    """Plots top N positive and negative features from Logistic Regression."""
    vectorizer = logreg_pipeline.named_steps['tfidf']
    clf = logreg_pipeline.named_steps['clf']
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    
    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    feature_importance = feature_importance.sort_values(by='coefficient', ascending=False)
    
    top_pos = feature_importance.head(top_n)
    top_neg = feature_importance.tail(top_n)
    
    top_features = pd.concat([top_pos, top_neg])
    
    plt.figure(figsize=(10, 8))
    colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
    sns.barplot(x='coefficient', y='feature', data=top_features, palette=colors)
    plt.title(f"Top {top_n} Positive and Negative Features (Logistic Regression)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")

def evaluate_model(model, X_test, y_test, model_name):
    """Calculates metrics for a model."""
    y_pred = model.predict(X_test)
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

def main():
    ensure_dir(REPORTS_DIR)
    
    print("Loading data...")
    X_test, y_test = load_test_data()
    
    print("Loading models...")
    try:
        logreg_pipeline = load_model("classification_logreg.joblib")
        mlp_pipeline = load_model("mlp_best.joblib")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_baselines.py and train_nn.py first.")
        return

    print("Generating NN Learning Curve...")
    plot_learning_curve(mlp_pipeline, os.path.join(REPORTS_DIR, "nn_learning_curve.png"))
    
    print("Generating NN Confusion Matrix...")
    plot_confusion_matrix_nn(mlp_pipeline, X_test, y_test, os.path.join(REPORTS_DIR, "nn_confusion_matrix.png"))
    
    print("Generating Feature Importance Plot...")
    plot_feature_importance(logreg_pipeline, os.path.join(REPORTS_DIR, "feature_importance.png"))
    
    print("Generating Model Comparison Table...")
    metrics = []
    metrics.append(evaluate_model(logreg_pipeline, X_test, y_test, "Logistic Regression"))
    metrics.append(evaluate_model(mlp_pipeline, X_test, y_test, "Neural Network"))
    
    df_metrics = pd.DataFrame(metrics)
    print("\nModel Comparison:")
    print(df_metrics)
    
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")

if __name__ == "__main__":
    main()
