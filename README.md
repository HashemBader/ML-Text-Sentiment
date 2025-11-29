# IMDb Text Sentiment Analysis

This project performs sentiment analysis on the IMDb Movie Reviews dataset using both baseline machine learning models (Logistic Regression) and neural networks (MLP).

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/HashemBader/ML-Text-Sentiment.git
    cd ML-Text-Sentiment
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    # Create virtual environment
    python3 -m venv venv

    # Activate virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Install the required packages using:
    ```bash
    pip install -r project/requirements.txt
    ```

## Data Preparation

The dataset is expected to be in `data/imdb_dataset.csv`. If you don't have the processed data yet, run the data processing script:

```bash
python3 src/data.py --input_csv data/imdb_dataset.csv --out_dir data/processed
```

This will clean the text and split the data into `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.

## Training

### Baseline Model (Logistic Regression)

To train the baseline Logistic Regression model and save it for evaluation:

```bash
python3 src/train_baselines.py
```

This script:
- Loads the cleaned data.
- Trains a Logistic Regression model with TF-IDF vectorizer.
- Logs metrics to MLflow.
- Saves the trained model to `models/classification_logreg.joblib`.

### Neural Network (MLP)

To train the MLP Classifier:

```bash
python3 src/train_nn.py
```

This script:
- Trains the best MLP model using optimized hyperparameters.
- Logs results to MLflow.
- Saves the best model to `models/mlp_best.joblib`.

## Evaluation

To generate evaluation reports and plots (Confusion Matrix, Target Distribution, etc.):

```bash
python3 src/evaluate.py --data_dir data/processed --models_dir models --output_dir reports
```

The generated plots will be saved in the `reports/` directory:
- `target_distribution.png`
- `corr_heatmap_or_box.png`
- `confusion_matrix.png` (requires trained baseline model)

## Notebooks

For exploratory data analysis and initial experiments, refer to:
- `notebooks/Text-Sentiment.ipynb`
