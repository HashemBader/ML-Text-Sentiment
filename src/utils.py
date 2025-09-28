"""
Utility functions and shared helpers for the ML project.
Contains common functions used across different modules.
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import warnings
import sys
import platform
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed}")


def create_dirs(directories: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories (List[str]): List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def save_model(model: Any, 
               filepath: str, 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model (Any): Trained model object
        filepath (str): Path to save the model
        metadata (Optional[Dict[str, Any]]): Additional metadata to save
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        
        # Save metadata if provided
        if metadata:
            metadata_path = str(filepath).replace('.joblib', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {str(e)}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Any: Loaded model object
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        raise


def load_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load model metadata.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        Optional[Dict[str, Any]]: Metadata dictionary or None
    """
    try:
        metadata_path = str(filepath).replace('.joblib', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            logger.warning(f"No metadata found for {filepath}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading metadata for {filepath}: {str(e)}")
        return None


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath (str): Path to file
        
    Returns:
        str: Human-readable file size
    """
    try:
        size_bytes = Path(filepath).stat().st_size
        
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
        
    except Exception as e:
        logger.error(f"Error getting file size for {filepath}: {str(e)}")
        return "Unknown"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Model object (sklearn, tensorflow, etc.)
        
    Returns:
        int: Number of parameters
    """
    try:
        # For TensorFlow/Keras models
        if hasattr(model, 'count_params'):
            return model.count_params()
        
        # For PyTorch models
        elif hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # For sklearn models, estimate based on features
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim == 1:
                return len(model.coef_) + 1  # +1 for bias
            else:
                return model.coef_.size + len(model.coef_)  # coefs + biases
        
        else:
            return 0
            
    except Exception as e:
        logger.warning(f"Could not count parameters: {str(e)}")
        return 0


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict[str, float]: Memory usage statistics
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }
        
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return {}
    except Exception as e:
        logger.warning(f"Error getting memory usage: {str(e)}")
        return {}


def log_system_info() -> None:
    """
    Log system information for debugging and reproducibility.
    """
    try:
        import platform
        import sys
        
        logger.info("=== SYSTEM INFORMATION ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Architecture: {platform.architecture()}")
        logger.info(f"Processor: {platform.processor()}")
        
        # Memory usage
        memory = get_memory_usage()
        if memory:
            logger.info(f"Memory usage: {memory['rss_mb']:.1f} MB ({memory['percent']:.1f}%)")
        
        # GPU information if available
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            logger.info(f"GPUs available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name}")
        except:
            pass
        
    except Exception as e:
        logger.warning(f"Error logging system info: {str(e)}")


def print_dataframe_info(df: pd.DataFrame, 
                        name: str = "DataFrame", 
                        show_sample: bool = True,
                        n_rows: int = 5) -> None:
    """
    Print comprehensive information about a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the DataFrame
        show_sample (bool): Whether to show sample rows
        n_rows (int): Number of sample rows to show
    """
    logger.info(f"\n=== {name.upper()} INFO ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Data types
    logger.info("\nData types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info("\nMissing values:")
        for col, count in missing.items():
            if count > 0:
                logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        logger.info("\nNo missing values")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"\nMemory usage: {memory_mb:.2f} MB")
    
    # Sample data
    if show_sample and len(df) > 0:
        logger.info(f"\nFirst {n_rows} rows:")
        print(df.head(n_rows).to_string())


def validate_data_splits(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        label_col: str = 'label') -> bool:
    """
    Validate data splits for consistency.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        label_col (str): Label column name
        
    Returns:
        bool: True if validation passes
    """
    try:
        logger.info("Validating data splits...")
        
        # Check if all splits have data
        assert len(train_df) > 0, "Training set is empty"
        assert len(val_df) > 0, "Validation set is empty"  
        assert len(test_df) > 0, "Test set is empty"
        
        # Check if all splits have the same columns
        train_cols = set(train_df.columns)
        val_cols = set(val_df.columns)
        test_cols = set(test_df.columns)
        
        assert train_cols == val_cols == test_cols, "Column mismatch between splits"
        
        # Check if label column exists
        assert label_col in train_df.columns, f"Label column '{label_col}' not found"
        
        # Check label distribution
        train_dist = train_df[label_col].value_counts(normalize=True).sort_index()
        val_dist = val_df[label_col].value_counts(normalize=True).sort_index()
        test_dist = test_df[label_col].value_counts(normalize=True).sort_index()
        
        logger.info("Label distributions:")
        logger.info(f"  Train: {train_dist.to_dict()}")
        logger.info(f"  Val:   {val_dist.to_dict()}")
        logger.info(f"  Test:  {test_dist.to_dict()}")
        
        # Check for class balance (warn if imbalanced)
        for name, dist in [("Train", train_dist), ("Val", val_dist), ("Test", test_dist)]:
            min_class_ratio = dist.min()
            if min_class_ratio < 0.1:
                logger.warning(f"{name} set is highly imbalanced (min class: {min_class_ratio:.3f})")
        
        logger.info("Data split validation passed!")
        return True
        
    except AssertionError as e:
        logger.error(f"Data split validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error during data split validation: {str(e)}")
        return False


def create_experiment_config(params: Dict[str, Any], 
                            timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Create experiment configuration with metadata.
    
    Args:
        params (Dict[str, Any]): Experiment parameters
        timestamp (Optional[str]): Experiment timestamp
        
    Returns:
        Dict[str, Any]: Complete experiment configuration
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = {
        'experiment_id': f"exp_{timestamp}",
        'timestamp': timestamp,
        'parameters': params,
        'system_info': {
            'python_version': sys.version,
            'platform': platform.platform()
        }
    }
    
    # Add memory info if available
    memory = get_memory_usage()
    if memory:
        config['system_info']['memory_mb'] = memory['rss_mb']
    
    return config


def compare_model_sizes(model_paths: List[str]) -> pd.DataFrame:
    """
    Compare sizes of saved models.
    
    Args:
        model_paths (List[str]): List of model file paths
        
    Returns:
        pd.DataFrame: Comparison of model sizes
    """
    comparisons = []
    
    for path in model_paths:
        if Path(path).exists():
            size = get_file_size(path)
            model_name = Path(path).stem
            
            comparisons.append({
                'model': model_name,
                'path': path,
                'size': size,
                'size_bytes': Path(path).stat().st_size
            })
        else:
            logger.warning(f"Model file not found: {path}")
    
    df = pd.DataFrame(comparisons)
    if len(df) > 0:
        df = df.sort_values('size_bytes', ascending=False).reset_index(drop=True)
    
    return df


def cleanup_files(patterns: List[str], directory: str = ".") -> None:
    """
    Clean up files matching given patterns.
    
    Args:
        patterns (List[str]): File patterns to delete
        directory (str): Directory to search in
    """
    deleted_count = 0
    
    for pattern in patterns:
        files_to_delete = Path(directory).glob(pattern)
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"Deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {str(e)}")
    
    logger.info(f"Cleaned up {deleted_count} files")


def main():
    """
    Demonstrate utility functions.
    """
    logger.info("Utility functions loaded successfully!")
    
    # Set seeds
    set_random_seeds(42)
    
    # Log system info
    log_system_info()
    
    # Create sample directories
    create_dirs(['temp', 'temp/models', 'temp/plots'])
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'text': ['This is a sample text'] * 100,
        'label': np.random.binomial(1, 0.6, 100)
    })
    
    print_dataframe_info(df, "Sample Data")
    
    logger.info("Utility demo completed!")


if __name__ == "__main__":
    main()