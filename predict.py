"""
Batch Prediction Script
Make predictions on new delivery data
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.config import settings
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.features.preprocessing import DataPreprocessor
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(input_path: Path, output_path: Path, model_alias: str = "champion"):
    """
    Main prediction pipeline

    Args:
        input_path: Path to input CSV file
        output_path: Path to save predictions
        model_alias: MLflow model alias to use
    """

    logger.info("=" * 80)
    logger.info("DELIVERY LATENESS PREDICTION - BATCH PREDICTION")
    logger.info("=" * 80)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    logger.info("\n[STEP 1/5] Loading data...")

    data_loader = DataLoader()
    df = data_loader.load_raw_data(input_path)

    logger.info(f"Loaded {len(df)} samples")

    # ========================================================================
    # 2. LOAD MODEL AND PREPROCESSOR
    # ========================================================================
    logger.info("\n[STEP 2/5] Loading model and preprocessor...")

    # Load model from MLflow
    registry = ModelRegistry()
    model = registry.load_model_by_alias(
        model_name=settings.mlflow.model_registry_name,
        alias=model_alias,
    )

    model_info = registry.get_model_info(
        model_name=settings.mlflow.model_registry_name,
        alias=model_alias,
    )

    logger.info(f"Loaded model version: {model_info['version']}")

    # Load preprocessor
    preprocessor_path = Path("models/preprocessor.joblib")
    if not preprocessor_path.exists():
        logger.error(f"Preprocessor not found at {preprocessor_path}")
        logger.error("Please train a model first using train.py")
        sys.exit(1)

    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor")

    # ========================================================================
    # 3. FEATURE ENGINEERING
    # ========================================================================
    logger.info("\n[STEP 3/5] Engineering features...")

    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.transform(df)

    logger.info(f"Created {len(feature_engineer.get_feature_names())} features")

    # ========================================================================
    # 4. PREPROCESSING
    # ========================================================================
    logger.info("\n[STEP 4/5] Preprocessing...")

    # Keep trip_id for output
    trip_ids = df_engineered["trip_id"].values if "trip_id" in df_engineered.columns else None

    # Prepare features
    feature_cols = [
        col for col in df_engineered.columns if col not in ["trip_id", "is_late_delivery"]
    ]
    X = df_engineered[feature_cols]

    # Transform
    X_transformed = preprocessor.transform(X)

    logger.info(f"Transformed shape: {X_transformed.shape}")

    # ========================================================================
    # 5. PREDICT
    # ========================================================================
    logger.info("\n[STEP 5/5] Making predictions...")

    # Predict
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)

    # Create output DataFrame
    output_df = pd.DataFrame(
        {
            "trip_id": trip_ids if trip_ids is not None else range(len(predictions)),
            "is_late_delivery_predicted": predictions,
            "probability_on_time": probabilities[:, 0],
            "probability_late": probabilities[:, 1],
        }
    )

    # Add confidence
    output_df["confidence"] = output_df["probability_late"].apply(
        lambda x: "high" if abs(x - 0.5) > 0.3 else ("medium" if abs(x - 0.5) > 0.1 else "low")
    )

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(f"Saved predictions to {output_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(
        f"Predicted late: {predictions.sum()} ({100 * predictions.sum() / len(predictions):.1f}%)"
    )
    logger.info(
        f"Predicted on-time: {(1 - predictions).sum()} ({100 * (1 - predictions).sum() / len(predictions):.1f}%)"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction for delivery lateness")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/predictions.csv"),
        help="Path to save predictions",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default="champion",
        choices=["champion", "candidate"],
        help="MLflow model alias to use",
    )

    args = parser.parse_args()

    main(args.input, args.output, args.model_alias)
