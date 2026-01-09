"""
Model Evaluation Module
Computes all required metrics: Accuracy, Precision, Recall, F1, ROC-AUC
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates classification model performance
    Follows Open/Closed Principle - can extend with new metrics
    """

    def __init__(self, average: str = "weighted"):
        """
        Initialize evaluator

        Args:
            average: Averaging strategy for multiclass metrics
                    ('micro', 'macro', 'weighted', 'binary')
        """
        self.average = average
        self.metrics: Dict[str, float] = {}
        self.confusion_mat: Optional[np.ndarray] = None

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model performance")

        # Accuracy
        self.metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision
        self.metrics["precision"] = precision_score(
            y_true, y_pred, average=self.average, zero_division=0
        )

        # Recall
        self.metrics["recall"] = recall_score(y_true, y_pred, average=self.average, zero_division=0)

        # F1-Score
        self.metrics["f1"] = f1_score(y_true, y_pred, average=self.average, zero_division=0)

        # ROC-AUC (requires probabilities)
        if y_pred_proba is not None:
            try:
                # For binary classification, use probabilities of positive class
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    self.metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    self.metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                self.metrics["roc_auc"] = 0.0
        else:
            logger.warning("Predicted probabilities not provided, skipping ROC-AUC")
            self.metrics["roc_auc"] = 0.0

        # Confusion matrix
        self.confusion_mat = confusion_matrix(y_true, y_pred)

        # Log metrics
        logger.info("=" * 60)
        logger.info("Model Evaluation Results:")
        logger.info("-" * 60)
        for metric_name, metric_value in self.metrics.items():
            logger.info(f"{metric_name.upper():>15}: {metric_value:.4f}")
        logger.info("=" * 60)

        return self.metrics

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred, digits=4)
        logger.info(f"\nClassification Report:\n{report}")
        return report

    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix"""
        return self.confusion_mat

    def plot_confusion_matrix(
        self, save_path: Optional[str] = None, class_names: Optional[list] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            save_path: Path to save plot
            class_names: Names of classes

        Returns:
            Matplotlib figure
        """
        if self.confusion_mat is None:
            raise ValueError("No confusion matrix available. Run evaluate() first.")

        if class_names is None:
            class_names = ["On Time", "Late"]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            self.confusion_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot ROC curve

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics"""
        return self.metrics

    def compare_with_threshold(self, primary_metric: str, threshold: float) -> bool:
        """
        Check if model meets minimum threshold

        Args:
            primary_metric: Metric to check (e.g., 'accuracy')
            threshold: Minimum required value

        Returns:
            True if metric >= threshold, False otherwise
        """
        if primary_metric not in self.metrics:
            logger.error(f"Metric {primary_metric} not found in evaluation results")
            return False

        metric_value = self.metrics[primary_metric]
        meets_threshold = metric_value >= threshold

        logger.info(
            f"Threshold check: {primary_metric} = {metric_value:.4f} "
            f"{'≥' if meets_threshold else '<'} {threshold:.4f} → "
            f"{'PASS ✓' if meets_threshold else 'FAIL ✗'}"
        )

        return meets_threshold

    def summarize_cv_results(
        self, cv_results: list[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Summarize cross-validation results

        Args:
            cv_results: List of metric dictionaries from each fold

        Returns:
            Dictionary with mean and std for each metric
        """
        logger.info(f"Summarizing {len(cv_results)} CV folds")

        summary = {}

        # Get all metric names
        metric_names = cv_results[0].keys()

        for metric_name in metric_names:
            values = [fold[metric_name] for fold in cv_results]

            summary[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        # Log summary
        logger.info("=" * 70)
        logger.info("Cross-Validation Summary:")
        logger.info("-" * 70)
        logger.info(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        logger.info("-" * 70)

        for metric_name, stats in summary.items():
            logger.info(
                f"{metric_name:<15} "
                f"{stats['mean']:<10.4f} "
                f"{stats['std']:<10.4f} "
                f"{stats['min']:<10.4f} "
                f"{stats['max']:<10.4f}"
            )

        logger.info("=" * 70)

        return summary
