"""
Model Training Module with Optuna Hyperparameter Tuning
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from src.config import settings
from src.features.preprocessing import DataPreprocessor
from src.models.evaluate import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains XGBoost classifier with Optuna hyperparameter tuning
    Follows Dependency Inversion - can swap different model types
    """

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        random_state: int = 42,
    ):
        """
        Initialize model trainer

        Args:
            preprocessor: Fitted DataPreprocessor
            random_state: Random seed
        """
        self.preprocessor = preprocessor
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: Optional[Dict] = None
        self.cv_results: list[Dict[str, float]] = []

    def calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Calculate scale_pos_weight for class imbalance

        Args:
            y: Target array

        Returns:
            scale_pos_weight value
        """
        # Count classes
        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)

        # Calculate ratio
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        logger.info(f"Class distribution: {n_negative} negative (0), {n_positive} positive (1)")
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        return scale_pos_weight

    def create_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_splits,
        scale_pos_weight: float,
    ):
        """
        Create Optuna objective function for hyperparameter tuning

        Args:
            X_train: Training features
            y_train: Training target
            cv_splits: Cross-validation splitter
            scale_pos_weight: Class imbalance weight

        Returns:
            Objective function
        """

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function"""

            # Suggest hyperparameters
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "scale_pos_weight": scale_pos_weight,
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            }

            # Create model
            model = xgb.XGBClassifier(**params)

            # Cross-validation score
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_splits,
                scoring=settings.optuna.metric,  # 'roc_auc' by default
                n_jobs=-1,
            )

            # Return mean score
            return scores.mean()

        return objective

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_splits,
        scale_pos_weight: float,
    ) -> Dict:
        """
        Tune hyperparameters using Optuna

        Args:
            X_train: Training features
            y_train: Training target
            cv_splits: Cross-validation splitter
            scale_pos_weight: Class imbalance weight

        Returns:
            Best hyperparameters
        """
        logger.info("Starting Optuna hyperparameter tuning")
        logger.info(f"Trials: {settings.optuna.n_trials}, Timeout: {settings.optuna.timeout}s")
        logger.info(f"Optimizing metric: {settings.optuna.metric}")

        # Create study
        study = optuna.create_study(
            direction=settings.optuna.direction,
            study_name="xgboost_delivery_lateness",
        )

        # Create objective
        objective = self.create_objective(X_train, y_train, cv_splits, scale_pos_weight)

        # Optimize
        study.optimize(
            objective,
            n_trials=settings.optuna.n_trials,
            timeout=settings.optuna.timeout,
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = study.best_params
        self.best_params["scale_pos_weight"] = scale_pos_weight
        self.best_params["random_state"] = self.random_state
        self.best_params["eval_metric"] = "logloss"
        self.best_params["use_label_encoder"] = False

        logger.info("=" * 70)
        logger.info("Optuna Optimization Complete!")
        logger.info(f"Best {settings.optuna.metric}: {study.best_value:.4f}")
        logger.info("Best hyperparameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info("=" * 70)

        return self.best_params

    def train_with_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_splits,
    ) -> Tuple[xgb.XGBClassifier, list]:
        """
        Train model with cross-validation

        Args:
            X_train: Training features
            y_train: Training target
            cv_splits: Cross-validation splitter

        Returns:
            Tuple of (trained model, cv_results)
        """
        logger.info(f"Training with {settings.model.cv_folds}-fold cross-validation")

        if self.best_params is None:
            raise ValueError("No hyperparameters available. Run tune_hyperparameters() first.")

        evaluator = ModelEvaluator()
        fold_results = []

        # Train and evaluate on each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits.split(X_train, y_train), 1):
            logger.info(f"Training fold {fold_idx}/{settings.model.cv_folds}")

            # Split data
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Train model
            model = xgb.XGBClassifier(**self.best_params)
            model.fit(X_fold_train, y_fold_train)

            # Predict
            y_pred = model.predict(X_fold_val)
            y_pred_proba = model.predict_proba(X_fold_val)

            # Evaluate
            metrics = evaluator.evaluate(y_fold_val, y_pred, y_pred_proba)
            fold_results.append(metrics)

            logger.info(
                f"Fold {fold_idx} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}"
            )

        # Summarize CV results
        cv_summary = evaluator.summarize_cv_results(fold_results)
        self.cv_results = fold_results

        # Train final model on full training data
        logger.info("Training final model on full training set")
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(X_train, y_train)

        logger.info("Training complete!")

        return self.model, cv_summary

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = True,
    ) -> xgb.XGBClassifier:
        """
        Full training pipeline

        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to run Optuna tuning

        Returns:
            Trained model
        """
        logger.info("=" * 70)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 70)

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = 1.0
        if settings.model.handle_imbalance:
            scale_pos_weight = self.calculate_scale_pos_weight(y_train)

        # Create CV splits
        cv_splits = self.preprocessor.create_cv_splits(
            X=None,  # Not needed for StratifiedKFold.split()
            y=y_train,
            n_splits=settings.model.cv_folds,
            random_state=self.random_state,
        )

        # Tune hyperparameters
        if tune_hyperparameters:
            self.tune_hyperparameters(X_train, y_train, cv_splits, scale_pos_weight)
        else:
            # Use default parameters
            logger.info("Using default hyperparameters (no tuning)")
            self.best_params = {
                "max_depth": settings.model.max_depth,
                "learning_rate": settings.model.learning_rate,
                "n_estimators": settings.model.n_estimators,
                "min_child_weight": settings.model.min_child_weight,
                "subsample": settings.model.subsample,
                "colsample_bytree": settings.model.colsample_bytree,
                "gamma": settings.model.gamma,
                "reg_alpha": settings.model.reg_alpha,
                "reg_lambda": settings.model.reg_lambda,
                "scale_pos_weight": scale_pos_weight,
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            }

        # Train with CV
        model, cv_summary = self.train_with_cv(X_train, y_train, cv_splits)

        logger.info("=" * 70)
        logger.info("Training Pipeline Complete!")
        logger.info("=" * 70)

        return model

    def save_model(self, model_path: Path) -> None:
        """
        Save trained model to disk

        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: Path) -> xgb.XGBClassifier:
        """
        Load model from disk

        Args:
            model_path: Path to load model from

        Returns:
            Loaded model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        return self.model

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """
        Get feature importance from trained model

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("No trained model available")

        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info("Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
            logger.info(f"  {i:2d}. {feature:<30s}: {importance:.4f}")

        return feature_importance
