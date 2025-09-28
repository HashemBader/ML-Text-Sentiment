"""
Data loading, cleaning, and splitting module.
Handles IMDB dataset preprocessing and train/validation/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import os
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMDBDataLoader:
    """
    Class for loading and preprocessing IMDB movie review dataset.
    """
    
    def __init__(self, data_path: str = "data/IMDB Dataset.csv"):
        """
        Initialize data loader.
        
        Args:
            data_path (str): Path to the IMDB dataset CSV file
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw IMDB dataset from CSV file.
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.raw_data)} samples")
            logger.info(f"Columns: {list(self.raw_data.columns)}")
            
            # Basic data info
            logger.info(f"Data shape: {self.raw_data.shape}")
            logger.info(f"Sentiment distribution:\n{self.raw_data['sentiment'].value_counts()}")
            
            return self.raw_data
            
        except FileNotFoundError:
            logger.error(f"Dataset not found at {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataset.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info("Cleaning data...")
        
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Check for missing values
        missing_values = cleaned_df.isnull().sum()
        logger.info(f"Missing values:\n{missing_values}")
        
        # Remove duplicates
        initial_size = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['review'])
        final_size = len(cleaned_df)
        logger.info(f"Removed {initial_size - final_size} duplicate reviews")
        
        # Remove extremely short reviews (less than 10 characters)
        cleaned_df = cleaned_df[cleaned_df['review'].str.len() >= 10]
        logger.info(f"Removed reviews with less than 10 characters")
        
        # Convert sentiment to binary labels
        cleaned_df['label'] = cleaned_df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Basic text statistics
        cleaned_df['review_length'] = cleaned_df['review'].str.len()
        cleaned_df['word_count'] = cleaned_df['review'].str.split().str.len()
        
        logger.info(f"Final dataset size: {len(cleaned_df)}")
        logger.info(f"Average review length: {cleaned_df['review_length'].mean():.2f} characters")
        logger.info(f"Average word count: {cleaned_df['word_count'].mean():.2f} words")
        
        self.processed_data = cleaned_df
        return cleaned_df
    
    def create_splits(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        val_size: float = 0.2, 
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Processed dataset
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of remaining data for validation set
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify splits by label
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test dataframes
        """
        logger.info("Creating train/validation/test splits...")
        
        # Prepare features and labels
        X = df['review']
        y = df['label'] if 'label' in df.columns else df['sentiment']
        
        stratify_col = y if stratify else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_col
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        # Create DataFrames with additional information
        train_df = pd.DataFrame({
            'review': X_train.values,
            'label': y_train.values
        }).reset_index(drop=True)
        
        val_df = pd.DataFrame({
            'review': X_val.values,
            'label': y_val.values
        }).reset_index(drop=True)
        
        test_df = pd.DataFrame({
            'review': X_test.values,
            'label': y_test.values
        }).reset_index(drop=True)
        
        # Log split information
        logger.info(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # Check class distribution in each split
        for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            pos_ratio = (split_df['label'] == 1).mean()
            logger.info(f"{name} set - Positive ratio: {pos_ratio:.3f}")
        
        return train_df, val_df, test_df
    
    def save_splits(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        output_dir: str = "data"
    ) -> None:
        """
        Save train/validation/test splits to CSV files.
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            output_dir (str): Directory to save split files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save splits
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        logger.info(f"Saved data splits to {output_path}")
    
    def load_splits(self, data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved train/validation/test splits.
        
        Args:
            data_dir (str): Directory containing split files
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test dataframes
        """
        data_path = Path(data_dir)
        
        try:
            train_df = pd.read_csv(data_path / "train.csv")
            val_df = pd.read_csv(data_path / "val.csv")
            test_df = pd.read_csv(data_path / "test.csv")
            
            logger.info("Loaded existing data splits")
            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            return train_df, val_df, test_df
            
        except FileNotFoundError as e:
            logger.error(f"Split files not found: {e}")
            raise


def main():
    """
    Main function to demonstrate data loading and splitting.
    """
    # Initialize data loader
    loader = IMDBDataLoader()
    
    # Load and process data
    raw_data = loader.load_raw_data()
    cleaned_data = loader.clean_data(raw_data)
    
    # Create splits
    train_df, val_df, test_df = loader.create_splits(cleaned_data)
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
    
    logger.info("Data processing completed successfully!")


if __name__ == "__main__":
    main()