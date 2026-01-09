"""
Main Training Script
End-to-end training pipeline for delivery lateness prediction
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import settings
from src.data.loader import DataLoader
from src.data.validation import DataValidator
from src.features.engineering import FeatureEngineer
from src.features.preprocessing import DataPreprocessor
from src.models.evaluate import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.train import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main training pipeline"""

    logger.info("=" * 80)
    logger.info("DELIVERY LATENESS PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 80)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    logger.info("\n[STEP 1/9] Loading data...")

    data_loader = DataLoader()
    df = data_loader.load_raw_data(settings.data.data_path)

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # ========================================================================
    # 2. VALIDATE DATA
    # ========================================================================
    logger.info("\n[STEP 2/9] Validating data...")

    validator = DataValidator()
    validation_results = validator.validate(df)

    if not validation_results["is_valid"]:
        logger.error("Data validation failed!")
        logger.error(f"Errors: {validation_results['errors']}")
        sys.exit(1)

    logger.info("✓ Data validation passed")

    # ========================================================================
    # 3. TRAIN/TEST SPLIT
    # ========================================================================
    logger.info("\n[STEP 3/9] Splitting data...")

    train_df, test_df = DataPreprocessor.train_test_split_data(
        df,
        test_size=settings.data.test_size,
        random_state=settings.data.random_state,
        stratify_column="is_late_delivery",
    )

    # Save splits
    data_loader.save_train_test_data(train_df, test_df, settings.data.processed_dir)

    # ========================================================================
    # 4. FEATURE ENGINEERING
    # ========================================================================
    logger.info("\n[STEP 4/9] Engineering features...")

    feature_engineer = FeatureEngineer()

    # Fit and transform on train set
    train_df_engineered = feature_engineer.fit_transform(train_df)

    # Transform test set
    test_df_engineered = feature_engineer.transform(test_df)

    logger.info(f"Created {len(feature_engineer.get_feature_names())} new features")

    # ========================================================================
    # 5. PREPARE FEATURES
    # ========================================================================
    logger.info("\n[STEP 5/9] Preparing features...")

    # Split features and target
    X_train, y_train = DataPreprocessor.split_features_target(
        train_df_engineered, target_column="is_late_delivery"
    )
    X_test, y_test = DataPreprocessor.split_features_target(
        test_df_engineered, target_column="is_late_delivery"
    )

    # Define feature types
    numerical_features = [
        "distance_km",
        "estimated_duration_min",
        "actual_duration_min",
        "num_stops",
        "vehicle_age_years",
        "load_weight_kg",
        "departure_hour",
        # Engineered numerical features
        "duration_diff",
        "duration_ratio",
        "is_ahead_of_schedule",
        "duration_pct_diff",
        "min_per_km",
        "is_rush_hour",
        "is_weekend",
        "load_per_km",
        "stops_per_km",
        "avg_distance_per_stop",
        "bad_weather_rural",
        "heavy_load_old_vehicle",
        "many_stops_urban",
        "complex_route",
        "bad_weather_rush_hour",
        "vehicle_capacity",
        "load_efficiency",
    ]

    categorical_features = [
        "vehicle_type",
        "day_of_week",
        "weather_condition",
        "region",
        # Engineered categorical features
        "time_of_day",
        "day_type",
        "load_category",
        "distance_category",
        "stops_category",
        "vehicle_region",
        "weather_region",
    ]

    # Filter features that exist in the dataframe
    numerical_features = [f for f in numerical_features if f in X_train.columns]
    categorical_features = [f for f in categorical_features if f in X_train.columns]

    logger.info(f"Numerical features: {len(numerical_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")

    # ========================================================================
    # 6. PREPROCESSING
    # ========================================================================
    logger.info("\n[STEP 6/9] Preprocessing features...")

    preprocessor = DataPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Fit on training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)

    logger.info(f"Transformed shape: {X_train_transformed.shape}")

    # Save preprocessor
    preprocessor_path = Path("models/preprocessor.joblib")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")

    # ========================================================================
    # 7. MODEL TRAINING
    # ========================================================================
    logger.info("\n[STEP 7/9] Training model...")

    trainer = ModelTrainer(
        preprocessor=preprocessor,
        random_state=settings.data.random_state,
    )

    # Train with Optuna tuning and CV
    model = trainer.train(
        X_train_transformed,
        y_train.values,
        tune_hyperparameters=True,
    )

    # ========================================================================
    # 8. EVALUATION
    # ========================================================================
    logger.info("\n[STEP 8/9] Evaluating model...")

    evaluator = ModelEvaluator()

    # Predict on test set
    y_test_pred = model.predict(X_test_transformed)
    y_test_proba = model.predict_proba(X_test_transformed)

    # Evaluate
    test_metrics = evaluator.evaluate(y_test.values, y_test_pred, y_test_proba)

    # Classification report
    evaluator.get_classification_report(y_test.values, y_test_pred)

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(save_path=plots_dir / "confusion_matrix.png")

    # Plot ROC curve
    evaluator.plot_roc_curve(
        y_test.values,
        y_test_proba[:, 1],
        save_path=plots_dir / "roc_curve.png",
    )

    # Feature importance
    feature_importance = trainer.get_feature_importance(preprocessor.get_feature_names())

    # Plot feature importance
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = list(feature_importance.items())[:20]
    features, importance = zip(*top_features)

    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    logger.info(f"Feature importance plot saved to {plots_dir / 'feature_importance.png'}")

    # ========================================================================
    # 9. MODEL REGISTRY
    # ========================================================================
    logger.info("\n[STEP 9/9] Registering model...")

    registry = ModelRegistry()

    # Prepare artifacts
    artifacts = {
        "confusion_matrix": plots_dir / "confusion_matrix.png",
        "roc_curve": plots_dir / "roc_curve.png",
        "feature_importance": plots_dir / "feature_importance.png",
    }

    # Prepare tags
    tags = {
        "model_type": "XGBoost",
        "task": "binary_classification",
        "dataset": "delivery_trips",
        "features_count": str(X_train_transformed.shape[1]),
    }

    # Register model
    run_id = registry.register_model(
        model=model,
        model_name=settings.mlflow.model_registry_name,
        metrics=test_metrics,
        params=trainer.best_params,
        artifacts=artifacts,
        tags=tags,
    )

    # Compare with threshold and promote
    is_champion = registry.compare_and_promote(
        model_name=settings.mlflow.model_registry_name,
        current_metrics=test_metrics,
        threshold_metric="accuracy",
        threshold_value=settings.evaluation.accuracy_threshold,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
    logger.info(f"Model Status: {'CHAMPION ✓' if is_champion else 'CANDIDATE'}")
    logger.info(f"MLflow Run ID: {run_id}")
    logger.info("=" * 80)

    return test_metrics, is_champion


if __name__ == "__main__":
    main()
