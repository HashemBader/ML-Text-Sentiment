# ML Text Sentiment Analysis Project

This project implements both classical machine learning and neural network approaches for text sentiment analysis using the IMDB dataset.

## Project Structure

```
project/
├── README.md                # This file - setup and run instructions
├── requirements.txt         # Python dependencies with pinned versions
├── src/
│   ├── data.py             # Data loading, cleaning, and splitting
│   ├── features.py         # Text preprocessing and feature engineering
│   ├── train_baselines.py  # Classical ML training (Logistic Regression, SVM, etc.)
│   ├── train_nn.py         # Neural network training (LSTM, Transformer, etc.)
│   ├── evaluate.py         # Model evaluation, metrics, and visualizations
│   └── utils.py            # Shared utility functions
├── notebooks/              # Optional EDA notebooks (not for training)
├── models/                 # Saved model artifacts
├── mlruns/                 # MLflow experiment tracking
└── data/
    └── README.md          # Data download and description
```

## Setup Instructions

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

Follow instructions in `data/README.md` to download and prepare the IMDB dataset.

## Usage

### Classical Machine Learning Baselines

Train classical ML models (Logistic Regression, SVM, Random Forest):

```bash
python src/train_baselines.py
```

### Neural Network Models

Train neural network models (LSTM, GRU, Transformer):

```bash
python src/train_nn.py
```

### Model Evaluation

Evaluate trained models and generate reports:

```bash
python src/evaluate.py
```

### MLflow Tracking

Start the MLflow UI to view experiment results:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Experiment Tracking

This project uses MLflow for experiment tracking. All training runs log:

- Model parameters and hyperparameters
- Training and validation metrics
- Model artifacts
- Feature importance (for applicable models)
- Confusion matrices and evaluation plots

## Models Implemented

### Classical ML Baselines
- Logistic Regression with TF-IDF features
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes

### Neural Networks
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM
- Simple Transformer

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Results

Results and model comparisons can be viewed in the MLflow UI after running experiments.

## Reproducibility

All experiments are tracked with MLflow to ensure reproducibility. Random seeds are set consistently across all training scripts.