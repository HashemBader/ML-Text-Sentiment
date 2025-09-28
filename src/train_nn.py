"""
Neural network training module for sentiment classification.
Implements LSTM, GRU, and Transformer models with MLflow tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import time
from pathlib import Path

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, Dense, Dropout, 
    Bidirectional, GlobalMaxPool1D, Conv1D,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# MLflow for experiment tracking
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature

# Custom modules
from data import IMDBDataLoader
from features import TextPreprocessor
from utils import set_random_seeds, create_dirs
from evaluate import evaluate_classifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow logging level
tf.get_logger().setLevel('ERROR')


class NeuralNetworkTrainer:
    """
    Trainer class for neural network models.
    """
    
    def __init__(self, experiment_name: str = "sentiment_neural_nets"):
        """
        Initialize neural network trainer.
        
        Args:
            experiment_name (str): MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.tokenizer = None
        self.max_features = 20000
        self.max_length = 200
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Create directories
        create_dirs(['models', 'plots'])
    
    def prepare_sequences(self, 
                         train_texts: List[str], 
                         val_texts: List[str], 
                         test_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert texts to sequences for neural networks.
        
        Args:
            train_texts (List[str]): Training texts
            val_texts (List[str]): Validation texts  
            test_texts (List[str]): Test texts
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, val, test sequences
        """
        logger.info("Converting texts to sequences...")
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(
            num_words=self.max_features,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        self.tokenizer.fit_on_texts(train_texts)
        
        # Convert texts to sequences
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        val_sequences = self.tokenizer.texts_to_sequences(val_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        
        # Pad sequences
        train_padded = pad_sequences(train_sequences, maxlen=self.max_length, padding='post', truncating='post')
        val_padded = pad_sequences(val_sequences, maxlen=self.max_length, padding='post', truncating='post')
        test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Log statistics
        vocab_size = len(self.tokenizer.word_index) + 1
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"Max sequence length: {self.max_length}")
        logger.info(f"Sequence shapes - Train: {train_padded.shape}, Val: {val_padded.shape}, Test: {test_padded.shape}")
        
        return train_padded, val_padded, test_padded
    
    def build_lstm_model(self, 
                        embedding_dim: int = 128,
                        lstm_units: int = 64,
                        dropout_rate: float = 0.5,
                        bidirectional: bool = True) -> Model:
        """
        Build LSTM model.
        
        Args:
            embedding_dim (int): Embedding dimension
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate
            bidirectional (bool): Whether to use bidirectional LSTM
            
        Returns:
            Model: Compiled LSTM model
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            Dropout(dropout_rate),
            Bidirectional(LSTM(lstm_units, return_sequences=False)) if bidirectional else LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru_model(self,
                       embedding_dim: int = 128,
                       gru_units: int = 64,
                       dropout_rate: float = 0.5,
                       bidirectional: bool = True) -> Model:
        """
        Build GRU model.
        
        Args:
            embedding_dim (int): Embedding dimension
            gru_units (int): Number of GRU units
            dropout_rate (float): Dropout rate
            bidirectional (bool): Whether to use bidirectional GRU
            
        Returns:
            Model: Compiled GRU model
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            Dropout(dropout_rate),
            Bidirectional(GRU(gru_units, return_sequences=False)) if bidirectional else GRU(gru_units),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self,
                       embedding_dim: int = 128,
                       filters: int = 128,
                       kernel_size: int = 5,
                       dropout_rate: float = 0.5) -> Model:
        """
        Build CNN model.
        
        Args:
            embedding_dim (int): Embedding dimension
            filters (int): Number of filters
            kernel_size (int): Kernel size
            dropout_rate (float): Dropout rate
            
        Returns:
            Model: Compiled CNN model
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            Dropout(dropout_rate),
            Conv1D(filters, kernel_size, activation='relu'),
            GlobalMaxPool1D(),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_transformer_model(self,
                               embedding_dim: int = 128,
                               num_heads: int = 4,
                               ff_dim: int = 128,
                               dropout_rate: float = 0.1) -> Model:
        """
        Build simple Transformer model.
        
        Args:
            embedding_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            dropout_rate (float): Dropout rate
            
        Returns:
            Model: Compiled Transformer model
        """
        inputs = tf.keras.Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = Embedding(self.max_features, embedding_dim)(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate
        )(embedding, embedding)
        
        # Add & Norm
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + embedding)
        
        # Feed Forward
        ffn_output = Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = Dense(embedding_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        
        # Add & Norm
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D()(ffn_output)
        
        # Classification head
        outputs = Dense(64, activation='relu')(pooled)
        outputs = Dropout(dropout_rate)(outputs)
        outputs = Dense(1, activation='sigmoid')(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get configuration for different neural network models.
        
        Returns:
            Dict[str, Dict]: Model configurations
        """
        return {
            "lstm": {
                "builder": self.build_lstm_model,
                "params": {
                    "embedding_dim": 128,
                    "lstm_units": 64,
                    "dropout_rate": 0.5,
                    "bidirectional": True
                }
            },
            "gru": {
                "builder": self.build_gru_model,
                "params": {
                    "embedding_dim": 128,
                    "gru_units": 64,
                    "dropout_rate": 0.5,
                    "bidirectional": True
                }
            },
            "cnn": {
                "builder": self.build_cnn_model,
                "params": {
                    "embedding_dim": 128,
                    "filters": 128,
                    "kernel_size": 5,
                    "dropout_rate": 0.5
                }
            },
            "transformer": {
                "builder": self.build_transformer_model,
                "params": {
                    "embedding_dim": 128,
                    "num_heads": 4,
                    "ff_dim": 128,
                    "dropout_rate": 0.1
                }
            }
        }
    
    def train_single_model(self,
                          model_name: str,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          epochs: int = 10,
                          batch_size: int = 32,
                          patience: int = 3) -> Dict[str, Any]:
        """
        Train a single neural network model.
        
        Args:
            model_name (str): Name of the model
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation sequences
            y_val (np.ndarray): Validation labels
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            patience (int): Early stopping patience
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Training {model_name}...")
        
        # Set random seeds
        set_random_seeds(42)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_nn"):
            start_time = time.time()
            
            # Get model configuration
            model_configs = self.get_model_configs()
            model_config = model_configs[model_name]
            
            # Build model
            model = model_config["builder"](**model_config["params"])
            
            logger.info(f"Model summary for {model_name}:")
            model.summary(print_fn=logger.info)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'models/{model_name}_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Make predictions
            train_pred_prob = model.predict(X_train, verbose=0)
            val_pred_prob = model.predict(X_val, verbose=0)
            
            train_pred = (train_pred_prob > 0.5).astype(int).flatten()
            val_pred = (val_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            train_accuracy = np.mean(train_pred == y_train)
            val_accuracy = np.mean(val_pred == y_val)
            
            # Get best epoch metrics
            best_epoch = np.argmax(history.history['val_accuracy'])
            best_val_acc = max(history.history['val_accuracy'])
            best_val_loss = history.history['val_loss'][best_epoch]
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("max_features", self.max_features)
            mlflow.log_param("max_length", self.max_length)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("patience", patience)
            
            for param, value in model_config["params"].items():
                mlflow.log_param(param, value)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("total_params", model.count_params())
            
            # Log training history
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history['loss'],
                history.history['accuracy'],
                history.history['val_loss'],
                history.history['val_accuracy']
            )):
                mlflow.log_metric("epoch_loss", loss, step=epoch)
                mlflow.log_metric("epoch_accuracy", acc, step=epoch)
                mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)
                mlflow.log_metric("epoch_val_accuracy", val_acc, step=epoch)
            
            # Log model
            signature = infer_signature(X_train, train_pred_prob)
            mlflow.tensorflow.log_model(
                model,
                f"{model_name}_model",
                signature=signature
            )
            
            # Save model locally
            model.save(f'models/{model_name}_final.h5')
            
            # Prepare results
            results = {
                "model": model,
                "model_name": model_name,
                "history": history.history,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "training_time": training_time,
                "total_params": model.count_params(),
                "train_predictions": train_pred,
                "val_predictions": val_pred,
                "train_pred_prob": train_pred_prob.flatten(),
                "val_pred_prob": val_pred_prob.flatten()
            }
            
            logger.info(f"{model_name} - Train Acc: {train_accuracy:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}, Best Val Acc: {best_val_acc:.4f}")
            
            return results
    
    def train_all_models(self,
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        epochs: int = 10,
                        batch_size: int = 32) -> Dict[str, Dict]:
        """
        Train all neural network models.
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            
        Returns:
            Dict[str, Dict]: All training results
        """
        logger.info("Training all neural network models...")
        
        # Prepare sequences
        X_train, X_val, X_test = self.prepare_sequences(
            train_df['review'].tolist(),
            val_df['review'].tolist(), 
            test_df['review'].tolist()
        )
        
        y_train = train_df['label'].values.astype(np.float32)
        y_val = val_df['label'].values.astype(np.float32)
        y_test = test_df['label'].values.astype(np.float32)
        
        all_results = {}
        model_names = list(self.get_model_configs().keys())
        
        for model_name in model_names:
            try:
                # Train model
                result = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, epochs, batch_size
                )
                
                # Evaluate on test set
                test_pred_prob = result["model"].predict(X_test, verbose=0)
                test_pred = (test_pred_prob > 0.5).astype(int).flatten()
                test_accuracy = np.mean(test_pred == y_test)
                
                result["test_accuracy"] = test_accuracy
                result["test_predictions"] = test_pred
                result["test_pred_prob"] = test_pred_prob.flatten()
                
                all_results[model_name] = result
                
                logger.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.results = all_results
        return all_results
    
    def print_results_summary(self):
        """Print a summary of all training results."""
        logger.info("\n=== NEURAL NETWORK RESULTS SUMMARY ===")
        
        results_list = []
        for model_name, result in self.results.items():
            results_list.append({
                "Model": model_name.upper(),
                "Train Acc": f"{result['train_accuracy']:.4f}",
                "Val Acc": f"{result['val_accuracy']:.4f}",
                "Test Acc": f"{result['test_accuracy']:.4f}",
                "Best Val": f"{result['best_val_accuracy']:.4f}",
                "Params": f"{result['total_params']:,}",
                "Time (s)": f"{result['training_time']:.2f}"
            })
        
        # Sort by test accuracy
        results_list.sort(key=lambda x: float(x["Test Acc"]), reverse=True)
        
        # Print results
        logger.info("-" * 80)
        for result in results_list:
            logger.info(f"{result['Model']:12} | "
                       f"Train: {result['Train Acc']} | "
                       f"Val: {result['Val Acc']} | "
                       f"Test: {result['Test Acc']} | "
                       f"Best: {result['Best Val']} | "
                       f"Params: {result['Params']:>8} | "
                       f"Time: {result['Time (s)']}")


def main():
    """
    Main function to train all neural network models.
    """
    logger.info("Starting neural network training...")
    
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
    trainer = NeuralNetworkTrainer()
    
    # Train all models
    results = trainer.train_all_models(train_df, val_df, test_df, epochs=10, batch_size=32)
    
    # Print summary
    trainer.print_results_summary()
    
    logger.info("Neural network training completed!")
    logger.info("View results in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()