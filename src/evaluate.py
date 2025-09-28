"""
Model evaluation module.
Implements comprehensive evaluation metrics, plots, and analysis for both classical and neural models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ML evaluation libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Statistical analysis
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self, save_plots: bool = True, plots_dir: str = "plots"):
        """
        Initialize model evaluator.
        
        Args:
            save_plots (bool): Whether to save generated plots
            plots_dir (str): Directory to save plots
        """
        self.save_plots = save_plots
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_pred_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_prob (Optional[np.ndarray]): Predicted probabilities
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, average='binary')
        
        # Probabilistic metrics (if probabilities are provided)
        if y_pred_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_prob)
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Additional derived metrics
        metrics['positive_rate'] = (tp + fn) / len(y_true)
        metrics['negative_rate'] = (tn + fp) / len(y_true)
        metrics['prediction_positive_rate'] = (tp + fp) / len(y_true)
        
        return metrics
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            title: str = "Confusion Matrix",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Create confusion matrix plot.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            title (str): Plot title
            save_name (Optional[str]): Name to save plot
            
        Returns:
            plt.Figure: Confusion matrix figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            fig.savefig(self.plots_dir / f"{save_name}_confusion_matrix.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_pred_prob: np.ndarray,
                      title: str = "ROC Curve",
                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Create ROC curve plot.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_prob (np.ndarray): Predicted probabilities
            title (str): Plot title
            save_name (Optional[str]): Name to save plot
            
        Returns:
            plt.Figure: ROC curve figure
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc_score = roc_auc_score(y_true, y_pred_prob)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            fig.savefig(self.plots_dir / f"{save_name}_roc_curve.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self,
                                   y_true: np.ndarray,
                                   y_pred_prob: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   save_name: Optional[str] = None) -> plt.Figure:
        """
        Create precision-recall curve plot.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_prob (np.ndarray): Predicted probabilities
            title (str): Plot title
            save_name (Optional[str]): Name to save plot
            
        Returns:
            plt.Figure: Precision-recall curve figure
        """
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        ap_score = average_precision_score(y_true, y_pred_prob)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot PR curve
        ax.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap_score:.3f})')
        
        # Baseline (random classifier)
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Random (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            fig.savefig(self.plots_dir / f"{save_name}_pr_curve.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self,
                              y_true: np.ndarray,
                              y_pred_prob: np.ndarray,
                              title: str = "Calibration Curve",
                              save_name: Optional[str] = None,
                              n_bins: int = 10) -> plt.Figure:
        """
        Create calibration curve plot.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_prob (np.ndarray): Predicted probabilities
            title (str): Plot title
            save_name (Optional[str]): Name to save plot
            n_bins (int): Number of bins for calibration
            
        Returns:
            plt.Figure: Calibration curve figure
        """
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_prob, n_bins=n_bins
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, 
               label="Model")
        ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Perfectly calibrated")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            fig.savefig(self.plots_dir / f"{save_name}_calibration.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_distribution(self,
                                   y_true: np.ndarray,
                                   y_pred_prob: np.ndarray,
                                   title: str = "Prediction Distribution",
                                   save_name: Optional[str] = None) -> plt.Figure:
        """
        Create prediction distribution plot.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_prob (np.ndarray): Predicted probabilities
            title (str): Plot title
            save_name (Optional[str]): Name to save plot
            
        Returns:
            plt.Figure: Prediction distribution figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate probabilities by true class
        neg_probs = y_pred_prob[y_true == 0]
        pos_probs = y_pred_prob[y_true == 1]
        
        # Plot histograms
        ax.hist(neg_probs, bins=30, alpha=0.7, label='Negative', color='red', density=True)
        ax.hist(pos_probs, bins=30, alpha=0.7, label='Positive', color='blue', density=True)
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            fig.savefig(self.plots_dir / f"{save_name}_pred_distribution.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_evaluation_report(self,
                               model_results: Dict[str, Dict],
                               dataset_name: str = "Test Set") -> pd.DataFrame:
        """
        Create comprehensive evaluation report for multiple models.
        
        Args:
            model_results (Dict[str, Dict]): Results from multiple models
            dataset_name (str): Name of the evaluation dataset
            
        Returns:
            pd.DataFrame: Evaluation report
        """
        report_data = []
        
        for model_name, results in model_results.items():
            # Get predictions and true labels
            y_true = results.get('y_true', None)
            y_pred = results.get('y_pred', results.get('test_predictions', None))
            y_pred_prob = results.get('y_pred_prob', results.get('test_pred_prob', None))
            
            if y_true is None or y_pred is None:
                logger.warning(f"Missing predictions for {model_name}, skipping...")
                continue
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, y_pred_prob)
            
            # Add model info
            metrics['model'] = model_name
            metrics['dataset'] = dataset_name
            
            # Add training info if available
            if 'training_time' in results:
                metrics['training_time'] = results['training_time']
            if 'total_params' in results:
                metrics['total_params'] = results['total_params']
            
            report_data.append(metrics)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Reorder columns
        col_order = ['model', 'dataset', 'accuracy', 'precision', 'recall', 'f1', 
                    'specificity', 'roc_auc', 'average_precision', 'training_time']
        
        # Only include columns that exist
        col_order = [col for col in col_order if col in report_df.columns]
        other_cols = [col for col in report_df.columns if col not in col_order]
        report_df = report_df[col_order + other_cols]
        
        return report_df
    
    def compare_models_plot(self,
                          model_results: Dict[str, Dict],
                          metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
                          title: str = "Model Comparison",
                          save_name: Optional[str] = None) -> plt.Figure:
        """
        Create model comparison plot.
        
        Args:
            model_results (Dict[str, Dict]): Results from multiple models
            metrics (List[str]): Metrics to compare
            title (str): Plot title
            save_name (Optional[str]): Name to save plot
            
        Returns:
            plt.Figure: Model comparison figure
        """
        # Create evaluation report
        report_df = self.create_evaluation_report(model_results)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in report_df.columns:
                ax = axes[i]
                
                # Create bar plot
                bars = ax.bar(report_df['model'], report_df[metric])
                
                # Add value labels on bars
                for bar, value in zip(bars, report_df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_title(f'{metric.capitalize()}', fontweight='bold')
                ax.set_ylabel(metric.capitalize())
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.0)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots and save_name:
            fig.savefig(self.plots_dir / f"{save_name}_comparison.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def evaluate_model(self,
                      model_name: str,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_pred_prob: Optional[np.ndarray] = None,
                      create_plots: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model_name (str): Name of the model
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_prob (Optional[np.ndarray]): Predicted probabilities
            create_plots (bool): Whether to create evaluation plots
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_prob)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
        
        # Create plots if requested
        if create_plots:
            results['plots'] = {}
            
            # Confusion matrix
            results['plots']['confusion_matrix'] = self.plot_confusion_matrix(
                y_true, y_pred, f"{model_name} - Confusion Matrix", model_name
            )
            
            if y_pred_prob is not None:
                # ROC curve
                results['plots']['roc_curve'] = self.plot_roc_curve(
                    y_true, y_pred_prob, f"{model_name} - ROC Curve", model_name
                )
                
                # Precision-recall curve
                results['plots']['pr_curve'] = self.plot_precision_recall_curve(
                    y_true, y_pred_prob, f"{model_name} - PR Curve", model_name
                )
                
                # Calibration curve
                results['plots']['calibration'] = self.plot_calibration_curve(
                    y_true, y_pred_prob, f"{model_name} - Calibration", model_name
                )
                
                # Prediction distribution
                results['plots']['pred_dist'] = self.plot_prediction_distribution(
                    y_true, y_pred_prob, f"{model_name} - Predictions", model_name
                )
        
        # Print summary
        logger.info(f"{model_name} Evaluation Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        if y_pred_prob is not None:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return results


def evaluate_classifier(model, X_test, y_test, model_name="Model"):
    """
    Utility function for quick model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_prob = None
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_prob = model.decision_function(X_test)
    
    # Evaluate model
    results = evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_prob)
    
    return results


def main():
    """
    Demonstrate evaluation functionality.
    """
    logger.info("Model evaluation module loaded successfully!")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_pred_prob = np.random.beta(2, 2, n_samples)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate synthetic model
    results = evaluator.evaluate_model(
        "Demo Model", y_true, y_pred, y_pred_prob
    )
    
    logger.info("Demo evaluation completed!")


if __name__ == "__main__":
    main()