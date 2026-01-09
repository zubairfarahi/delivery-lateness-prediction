"""
Configuration Management using Pydantic Settings
"""

from pathlib import Path
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    """Data-related configuration"""

    data_path: Path = Field(
        default=Path("data/delivery_trips.csv"),
        description="Path to the raw data file",
    )
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Directory for processed data"
    )
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Test set size ratio")
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class ModelConfig(BaseSettings):
    """Model training configuration"""

    model_name: str = Field(default="xgboost_late_delivery", description="Model identifier")
    cv_folds: int = Field(default=5, ge=2, description="Number of CV folds")
    cv_strategy: Literal["stratified", "kfold"] = Field(
        default="stratified", description="Cross-validation strategy"
    )

    # XGBoost hyperparameters (will be tuned by Optuna)
    max_depth: int = Field(default=6, ge=1, le=20)
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0)
    n_estimators: int = Field(default=100, ge=10, le=1000)
    min_child_weight: int = Field(default=1, ge=1)
    subsample: float = Field(default=0.8, ge=0.1, le=1.0)
    colsample_bytree: float = Field(default=0.8, ge=0.1, le=1.0)
    gamma: float = Field(default=0.0, ge=0.0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)

    # Class imbalance handling
    handle_imbalance: bool = Field(default=True, description="Use scale_pos_weight for imbalance")


class OptunaConfig(BaseSettings):
    """Hyperparameter tuning configuration"""

    n_trials: int = Field(default=50, ge=1, description="Number of Optuna trials")
    timeout: int = Field(default=3600, ge=60, description="Timeout in seconds for optimization")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimization direction"
    )
    metric: str = Field(default="roc_auc", description="Metric to optimize")


class EvaluationConfig(BaseSettings):
    """Model evaluation configuration"""

    accuracy_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy for champion promotion",
    )
    primary_metric: str = Field(default="roc_auc", description="Primary metric for model selection")
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1", "roc_auc"],
        description="Metrics to track",
    )


class MLflowConfig(BaseSettings):
    """MLflow tracking configuration"""

    tracking_uri: str = Field(default="file:./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(
        default="delivery_lateness_prediction", description="Experiment name"
    )
    model_registry_name: str = Field(
        default="delivery_late_classifier", description="Registered model name"
    )


class APIConfig(BaseSettings):
    """FastAPI configuration"""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    reload: bool = Field(default=False, description="Enable auto-reload")
    workers: int = Field(default=1, ge=1, description="Number of workers")
    model_alias: str = Field(default="champion", description="MLflow model alias to load")


class LoggingConfig(BaseSettings):
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="1 week", description="Log retention period")


class Settings(BaseSettings):
    """Main application settings - Aggregates all configs"""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


# Global settings instance (Singleton pattern)
settings = Settings()
