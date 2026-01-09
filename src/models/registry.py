"""
Model Registry Module using MLflow
"""

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Manages model versioning and registration with MLflow
    Follows Interface Segregation - focused registry operations
    """

    def __init__(self):
        """Initialize MLflow registry"""
        # Set tracking URI
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

        # Set experiment
        mlflow.set_experiment(settings.mlflow.experiment_name)

        self.client = MlflowClient()

        logger.info(f"MLflow tracking URI: {settings.mlflow.tracking_uri}")
        logger.info(f"MLflow experiment: {settings.mlflow.experiment_name}")

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start MLflow run

        Args:
            run_name: Name for the run

        Returns:
            Active MLflow run
        """
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id}")

        return run

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow

        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics to MLflow")

    def log_artifact(self, artifact_path: Path, artifact_name: Optional[str] = None) -> None:
        """
        Log artifact to MLflow

        Args:
            artifact_path: Path to artifact file
            artifact_name: Optional name for artifact
        """
        mlflow.log_artifact(str(artifact_path), artifact_path=artifact_name)
        logger.info(f"Logged artifact: {artifact_path}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log model to MLflow

        Args:
            model: Trained model
            artifact_path: Path within run's artifact URI
            registered_model_name: Name for model registry
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

        logger.info(f"Logged model to MLflow: {artifact_path}")

        if registered_model_name:
            logger.info(f"Registered model: {registered_model_name}")

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, Path]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Register model with MLflow

        Args:
            model: Trained model
            model_name: Name for registered model
            metrics: Model metrics
            params: Model parameters
            artifacts: Optional artifacts to log
            tags: Optional tags

        Returns:
            Run ID
        """
        logger.info("=" * 70)
        logger.info(f"Registering model: {model_name}")
        logger.info("=" * 70)

        with mlflow.start_run() as run:
            # Log parameters
            self.log_params(params)

            # Log metrics
            self.log_metrics(metrics)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if artifact_path.exists():
                        self.log_artifact(artifact_path, artifact_name)

            # Log and register model
            self.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
            )

            run_id = run.info.run_id

        logger.info(f"Model registered successfully. Run ID: {run_id}")
        logger.info("=" * 70)

        return run_id

    def promote_to_champion(
        self,
        model_name: str,
        version: Optional[int] = None,
    ) -> None:
        """
        Promote model to 'champion' alias

        Args:
            model_name: Registered model name
            version: Model version (uses latest if None)
        """
        logger.info(f"Promoting model to champion: {model_name}")

        # Get latest version if not specified
        if version is None:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model: {model_name}")

            version = max(int(v.version) for v in versions)

        # Set alias
        self.client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version,
        )

        logger.info(f"Model version {version} promoted to champion")

    def promote_to_candidate(
        self,
        model_name: str,
        version: Optional[int] = None,
    ) -> None:
        """
        Promote model to 'candidate' alias

        Args:
            model_name: Registered model name
            version: Model version (uses latest if None)
        """
        logger.info(f"Promoting model to candidate: {model_name}")

        # Get latest version if not specified
        if version is None:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model: {model_name}")

            version = max(int(v.version) for v in versions)

        # Set alias
        self.client.set_registered_model_alias(
            name=model_name,
            alias="candidate",
            version=version,
        )

        logger.info(f"Model version {version} promoted to candidate")

    def load_model_by_alias(
        self,
        model_name: str,
        alias: str = "champion",
    ) -> Any:
        """
        Load model by alias

        Args:
            model_name: Registered model name
            alias: Model alias ('champion', 'candidate', etc.)

        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}@{alias}"

        logger.info(f"Loading model: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)

        logger.info("Model loaded successfully")

        return model

    def get_model_info(
        self,
        model_name: str,
        alias: str = "champion",
    ) -> Dict[str, Any]:
        """
        Get model information

        Args:
            model_name: Registered model name
            alias: Model alias

        Returns:
            Dictionary with model info
        """
        # Get model version by alias
        model_version = self.client.get_model_version_by_alias(model_name, alias)

        info = {
            "name": model_name,
            "version": model_version.version,
            "alias": alias,
            "run_id": model_version.run_id,
            "status": model_version.status,
            "creation_timestamp": model_version.creation_timestamp,
        }

        return info

    def compare_and_promote(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        threshold_metric: str = "accuracy",
        threshold_value: float = 0.75,
    ) -> bool:
        """
        Compare model with threshold and promote if meets criteria

        Args:
            model_name: Model name
            current_metrics: Current model metrics
            threshold_metric: Metric to check
            threshold_value: Minimum value required

        Returns:
            True if promoted to champion, False if candidate
        """
        logger.info("=" * 70)
        logger.info("Model Promotion Decision")
        logger.info("=" * 70)

        metric_value = current_metrics.get(threshold_metric, 0.0)

        logger.info(f"Threshold metric: {threshold_metric}")
        logger.info(f"Threshold value: {threshold_value:.4f}")
        logger.info(f"Current value: {metric_value:.4f}")

        if metric_value >= threshold_value:
            logger.info("✓ PASS - Promoting to CHAMPION")
            self.promote_to_champion(model_name)
            logger.info("=" * 70)
            return True
        else:
            logger.info("✗ FAIL - Promoting to CANDIDATE")
            self.promote_to_candidate(model_name)
            logger.info(
                f"Model needs {threshold_value - metric_value:.4f} improvement to become champion"
            )
            logger.info("=" * 70)
            return False
