"""
Classical machine learning baseline training module.
Implements various classical ML algorithms for sentiment classification with MLflow tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import joblib
from pathlib import Path
import time

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Custom modules
from data import IMDBDataLoader
from features import TextPreprocessor, FeatureExtractor, SentimentFeatures
from utils import set_random_seeds, save_model, load_model
from evaluate import evaluate_classifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Trainer class for classical machine learning baselines.
    """
    
    def __init__(self, experiment_name: str = "sentiment_baselines"):
        """
        Initialize baseline trainer.
        
        Args:
            experiment_name (str): MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
    def prepare_features(self, 
                        train_texts: List[str], 
                        val_texts: List[str], 
                        test_texts: List[str],
                        feature_type: str = "tfidf") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for training.
        
        Args:
            train_texts (List[str]): Training texts
            val_texts (List[str]): Validation texts
            test_texts (List[str]): Test texts
            feature_type (str): Type of features to extract ("tfidf", "count", or "combined")
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, val, test features
        """
        logger.info(f"Preparing {feature_type} features...")
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        train_processed = self.preprocessor.preprocess_texts(train_texts)
        val_processed = self.preprocessor.preprocess_texts(val_texts)
        test_processed = self.preprocessor.preprocess_texts(test_texts)
        
        # Extract features based on type
        if feature_type == "tfidf":
            # Fit on training data
            train_features, vectorizer = self.feature_extractor.extract_tfidf_features(train_processed)
            
            # Transform validation and test data
            val_features = vectorizer.transform(val_processed).toarray()
            test_features = vectorizer.transform(test_processed).toarray()
            
        elif feature_type == "count":
            # Fit on training data
            train_features, vectorizer = self.feature_extractor.extract_count_features(train_processed)
            
            # Transform validation and test data
            val_features = vectorizer.transform(val_processed).toarray()
            test_features = vectorizer.transform(test_processed).toarray()
            
        elif feature_type == "combined":
            # Extract both TF-IDF and sentiment features
            train_tfidf, tfidf_vectorizer = self.feature_extractor.extract_tfidf_features(train_processed)
            train_sentiment = SentimentFeatures.extract_sentiment_features(train_texts).values
            train_features = np.hstack([train_tfidf, train_sentiment])
            
            val_tfidf = tfidf_vectorizer.transform(val_processed).toarray()
            val_sentiment = SentimentFeatures.extract_sentiment_features(val_texts).values
            val_features = np.hstack([val_tfidf, val_sentiment])
            
            test_tfidf = tfidf_vectorizer.transform(test_processed).toarray()
            test_sentiment = SentimentFeatures.extract_sentiment_features(test_texts).values
            test_features = np.hstack([test_tfidf, test_sentiment])
        
        logger.info(f"Feature shapes - Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")
        
        return train_features, val_features, test_features
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get configuration for different baseline models.
        
        Returns:
            Dict[str, Dict]: Model configurations
        """
        return {
            "logistic_regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            },
            "svm_linear": {
                "model": SVC(kernel="linear", random_state=42, probability=True),
                "params": {
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"]
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            },
            "naive_bayes": {
                "model": MultinomialNB(),
                "params": {
                    "alpha": [0.1, 0.5, 1.0, 2.0]
                }
            }
        }
    
    def train_single_model(self,
                          model_name: str,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          feature_type: str = "tfidf",
                          use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            feature_type (str): Type of features used
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Training {model_name}...")
        
        # Set random seeds
        set_random_seeds(42)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{feature_type}"):
            start_time = time.time()
            
            # Get model configuration
            model_configs = self.get_model_configs()
            model_config = model_configs[model_name]
            
            if use_grid_search and model_config["params"]:
                # Hyperparameter tuning with grid search
                logger.info(f"Performing grid search for {model_name}...")
                
                grid_search = GridSearchCV(
                    model_config["model"],
                    model_config["params"],
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                logger.info(f"Best parameters: {best_params}")
                
            else:
                # Train with default parameters
                best_model = model_config["model"]
                best_model.fit(X_train, y_train)
                best_params = {}
            
            training_time = time.time() - start_time
            
            # Make predictions
            train_pred = best_model.predict(X_train)
            val_pred = best_model.predict(X_val)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("feature_type", feature_type)
            mlflow.log_param("feature_dim", X_train.shape[1])
            mlflow.log_param("train_samples", X_train.shape[0])
            
            if best_params:
                for param, value in best_params.items():
                    mlflow.log_param(param, value)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("cv_mean", cv_mean)
            mlflow.log_metric("cv_std", cv_std)
            mlflow.log_metric("training_time", training_time)
            
            # Log model
            signature = infer_signature(X_train, train_pred)
            mlflow.sklearn.log_model(
                best_model, 
                f"{model_name}_model",
                signature=signature
            )
            
            # Save model locally
            model_path = Path("models") / f"{model_name}_{feature_type}.joblib"
            save_model(best_model, str(model_path))
            
            # Prepare results
            results = {
                "model": best_model,
                "model_name": model_name,
                "feature_type": feature_type,
                "best_params": best_params,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "training_time": training_time,
                "train_predictions": train_pred,
                "val_predictions": val_pred
            }
            
            logger.info(f"{model_name} - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            
            return results
    
    def train_all_baselines(self,
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           feature_types: List[str] = ["tfidf", "combined"]) -> Dict[str, Dict]:
        """
        Train all baseline models.
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            feature_types (List[str]): Types of features to use
            
        Returns:
            Dict[str, Dict]: All training results
        """
        logger.info("Training all baseline models...")
        
        all_results = {}
        model_names = list(self.get_model_configs().keys())
        
        for feature_type in feature_types:
            logger.info(f"\n=== Training with {feature_type} features ===")
            
            # Prepare features
            X_train, X_val, X_test = self.prepare_features(
                train_df['review'].tolist(),
                val_df['review'].tolist(),
                test_df['review'].tolist(),
                feature_type=feature_type
            )
            
            y_train = train_df['label'].values
            y_val = val_df['label'].values
            y_test = test_df['label'].values
            
            feature_results = {}
            
            for model_name in model_names:
                try:
                    # Train model
                    result = self.train_single_model(
                        model_name, X_train, y_train, X_val, y_val, feature_type
                    )
                    
                    # Evaluate on test set
                    test_pred = result["model"].predict(X_test)
                    test_accuracy = accuracy_score(y_test, test_pred)
                    result["test_accuracy"] = test_accuracy
                    result["test_predictions"] = test_pred
                    
                    feature_results[model_name] = result
                    
                    logger.info(f"{model_name} ({feature_type}) - Test Accuracy: {test_accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} with {feature_type}: {str(e)}")
                    continue
            
            all_results[feature_type] = feature_results
        
        self.results = all_results
        return all_results
    
    def print_results_summary(self):
        """Print a summary of all training results."""
        logger.info("\n=== BASELINE RESULTS SUMMARY ===")
        
        for feature_type, feature_results in self.results.items():
            logger.info(f"\n{feature_type.upper()} Features:")
            logger.info("-" * 50)
            
            results_list = []
            for model_name, result in feature_results.items():
                results_list.append({
                    "Model": model_name,
                    "Train Acc": f"{result['train_accuracy']:.4f}",
                    "Val Acc": f"{result['val_accuracy']:.4f}",
                    "Test Acc": f"{result['test_accuracy']:.4f}",
                    "CV Mean": f"{result['cv_mean']:.4f}",
                    "Time (s)": f"{result['training_time']:.2f}"
                })
            
            # Sort by test accuracy
            results_list.sort(key=lambda x: float(x["Test Acc"]), reverse=True)
            
            # Print results
            for result in results_list:
                logger.info(f"{result['Model']:15} | "
                          f"Train: {result['Train Acc']} | "
                          f"Val: {result['Val Acc']} | "
                          f"Test: {result['Test Acc']} | "
                          f"CV: {result['CV Mean']} | "
                          f"Time: {result['Time (s']}")


def main():
    """
    Main function to train all baseline models.
    """
    logger.info("Starting baseline model training...")
    
    # Load data
    loader = IMDBDataLoader()
    
    try:
        # Try to load existing splits
        train_df, val_df, test_df = loader.load_splits()
    except FileNotFoundError:
        # Create new splits if they don't exist
        logger.info("Creating new data splits...")
        raw_data = loader.load_raw_data()
        cleaned_data = loader.clean_data(raw_data)
        train_df, val_df, test_df = loader.create_splits(cleaned_data)
        loader.save_splits(train_df, val_df, test_df)
    
    # Initialize trainer
    trainer = BaselineTrainer()
    
    # Train all models
    results = trainer.train_all_baselines(train_df, val_df, test_df)
    
    # Print summary
    trainer.print_results_summary()
    
    logger.info("Baseline training completed!")
    logger.info("View results in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()