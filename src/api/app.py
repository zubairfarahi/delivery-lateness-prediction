"""
FastAPI Prediction Endpoint
Simple REST API for delivery lateness predictions
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.features.engineering import FeatureEngineer
from src.features.preprocessing import DataPreprocessor
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Delivery Lateness Prediction API",
    description="Predict whether deliveries will be late using ML",
    version="1.0.0",
)

# Global variables for model and preprocessor
model = None
feature_engineer = None
preprocessor = None


class DeliveryRequest(BaseModel):
    """Request schema for prediction"""

    distance_km: float = Field(..., ge=1, le=50, description="Distance in kilometers")
    estimated_duration_min: float = Field(..., ge=5, description="Estimated duration in minutes")
    actual_duration_min: float = Field(..., ge=5, description="Actual duration in minutes")
    num_stops: int = Field(..., ge=1, le=25, description="Number of stops")
    vehicle_type: str = Field(..., description="Vehicle type: van, truck_small, truck_large")
    vehicle_age_years: int = Field(..., ge=1, le=14, description="Vehicle age in years")
    load_weight_kg: float = Field(..., ge=50, le=2000, description="Load weight in kg")
    departure_hour: int = Field(..., ge=0, le=23, description="Departure hour (0-23)")
    day_of_week: str = Field(..., description="Day of week: Mon-Sun")
    weather_condition: str = Field(..., description="Weather: clear, rain, snow, windy")
    region: str = Field(..., description="Region: urban, suburban, rural")

    class Config:
        schema_extra = {
            "example": {
                "distance_km": 25.5,
                "estimated_duration_min": 60.0,
                "actual_duration_min": 65.0,
                "num_stops": 10,
                "vehicle_type": "van",
                "vehicle_age_years": 5,
                "load_weight_kg": 1200.0,
                "departure_hour": 14,
                "day_of_week": "Wed",
                "weather_condition": "rain",
                "region": "urban",
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction"""

    is_late_delivery: int = Field(..., description="Prediction: 0 = on time, 1 = late")
    probability_late: float = Field(..., description="Probability of being late")
    confidence: str = Field(..., description="Confidence level: low, medium, high")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_alias: str


@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global model, feature_engineer, preprocessor

    logger.info("Loading model and preprocessor...")

    try:
        # Initialize registry
        registry = ModelRegistry()

        # Load champion model
        model = registry.load_model_by_alias(
            model_name=settings.mlflow.model_registry_name,
            alias=settings.api.model_alias,
        )

        # Get model info
        model_info = registry.get_model_info(
            model_name=settings.mlflow.model_registry_name,
            alias=settings.api.model_alias,
        )

        logger.info(f"Loaded model version: {model_info['version']}")

        # Initialize feature engineer and preprocessor
        # Note: In production, these should be saved with the model
        feature_engineer = FeatureEngineer()
        preprocessor = DataPreprocessor()

        # Load preprocessor if saved
        preprocessor_path = Path("models/preprocessor.joblib")
        if preprocessor_path.exists():
            import joblib

            preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded saved preprocessor")
        else:
            logger.warning("Preprocessor not found, using default")

        logger.info("Model and preprocessor loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Delivery Lateness Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_alias": settings.api.model_alias,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: DeliveryRequest):
    """
    Predict if delivery will be late

    Args:
        request: Delivery information

    Returns:
        Prediction and probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        data = {
            "trip_id": [0],  # Dummy ID
            "distance_km": [request.distance_km],
            "estimated_duration_min": [request.estimated_duration_min],
            "actual_duration_min": [request.actual_duration_min],
            "num_stops": [request.num_stops],
            "vehicle_type": [request.vehicle_type],
            "vehicle_age_years": [request.vehicle_age_years],
            "load_weight_kg": [request.load_weight_kg],
            "departure_hour": [request.departure_hour],
            "day_of_week": [request.day_of_week],
            "weather_condition": [request.weather_condition],
            "region": [request.region],
        }

        df = pd.DataFrame(data)

        # Feature engineering
        df_engineered = feature_engineer.transform(df)

        # Prepare features (drop trip_id)
        feature_cols = [col for col in df_engineered.columns if col != "trip_id"]
        X = df_engineered[feature_cols]

        # Transform if preprocessor is fitted
        if preprocessor and preprocessor.preprocessor is not None:
            X_transformed = preprocessor.transform(X)
        else:
            # Use raw features if preprocessor not available
            X_transformed = X.values

        # Predict
        prediction = model.predict(X_transformed)[0]
        probability = model.predict_proba(X_transformed)[0]

        # Probability of late delivery (class 1)
        prob_late = probability[1]

        # Confidence level
        if abs(prob_late - 0.5) < 0.1:
            confidence = "low"
        elif abs(prob_late - 0.5) < 0.3:
            confidence = "medium"
        else:
            confidence = "high"

        logger.info(
            f"Prediction: {prediction}, Probability late: {prob_late:.4f}, Confidence: {confidence}"
        )

        return {
            "is_late_delivery": int(prediction),
            "probability_late": float(prob_late),
            "confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[DeliveryRequest]):
    """
    Batch prediction endpoint

    Args:
        requests: List of delivery requests

    Returns:
        List of predictions
    """
    predictions = []

    for request in requests:
        try:
            pred = await predict(request)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Add error response
            predictions.append(
                {
                    "is_late_delivery": -1,
                    "probability_late": 0.0,
                    "confidence": "error",
                }
            )

    return predictions


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
    )
