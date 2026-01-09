# Delivery Lateness Prediction

End-to-end ML system that predicts whether deliveries will be late based on trip characteristics, vehicle info, weather, and timing.

## What It Does

Predicts delivery lateness (on-time vs late) using XGBoost with Optuna hyperparameter tuning. The system handles class imbalance, engineers 25+ features, and serves predictions through a REST API.

**Dataset**: 1,000 delivery trips (77% on-time, 23% late)
**Goal**: ≥75% accuracy for production deployment

## Tech Stack

- **ML**: XGBoost, Optuna, scikit-learn
- **Tracking**: MLflow (experiments & model registry)
- **API**: FastAPI with auto-generated docs
- **Data Validation**: Great Expectations
- **Config**: Pydantic Settings
- **Deployment**: Docker + docker-compose

## Quick Start

### 1. Install

```bash
cd delivery-lateness-prediction
make install
```

Or manually: `pip install -r requirements.txt`

### 2. Train Model

```bash
make train
```

This runs the full pipeline:
- Validates data quality
- Engineers features (duration ratios, time-based, efficiency metrics)
- Tunes hyperparameters with Optuna (50 trials)
- Trains XGBoost with 5-fold CV
- Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
- Registers model in MLflow (champion if ≥75% accuracy)

Takes ~5-10 minutes. For quick testing: `make train-quick` (10 trials)

### 3. View Results

```bash
make mlflow
```

Open http://localhost:8001 to see experiments, metrics, and model artifacts.

### 4. Run API

```bash
make api
```

API runs at http://localhost:8000

**Test it:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "region": "urban"
  }'
```

Response:
```json
{
  "is_late_delivery": 0,
  "probability_late": 0.23,
  "confidence": "medium"
}
```

Visit http://localhost:8000/docs for interactive API documentation.

## Project Structure

```
├── data/
│   ├── delivery_trips.csv      # Raw data
│   └── processed/              # Train/test splits
├── src/
│   ├── data/                   # Loading & validation
│   ├── features/               # Engineering & preprocessing
│   ├── models/                 # Training, eval, registry
│   └── api/                    # FastAPI app
├── notebooks/
│   └── eda.ipynb               # Exploratory analysis
├── train.py                    # Main training script
├── predict.py                  # Batch predictions
└── Makefile                    # All commands
```

## Key Commands

```bash
make help           # Show all commands
make train          # Train model
make predict        # Batch predictions
make api            # Start API server
make mlflow         # Open MLflow UI
make docker-build   # Build container
make format         # Format code
```

## Features Engineered

**Duration**: `duration_diff`, `duration_ratio`, `is_ahead_of_schedule`
**Time**: `is_rush_hour`, `is_weekend`, `time_of_day`
**Efficiency**: `load_per_km`, `stops_per_km`, `min_per_km`
**Interactions**: `bad_weather_rural`, `heavy_load_old_vehicle`

## Model Details

- **Algorithm**: XGBoost with scale_pos_weight for class imbalance
- **Tuning**: Optuna Bayesian optimization (max_depth, learning_rate, n_estimators, etc.)
- **Validation**: 5-fold Stratified K-Fold
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Promotion**: Auto-promoted to "champion" if accuracy ≥75%, else "candidate"

## Docker

Run everything in containers:

```bash
# Full stack (training + API + MLflow)
docker-compose up -d

# Or individually
make docker-build
make docker-run     # Training
make docker-api     # API server
```

## Configuration

Edit `.env` file or set environment variables:

```bash
DATA__TEST_SIZE=0.2
OPTUNA__N_TRIALS=50
EVALUATION__ACCURACY_THRESHOLD=0.75
API__PORT=8000
```

## Development

```bash
make format    # Black + isort
make lint      # Flake8
make test      # Pytest (if tests available)
```

Code follows SOLID principles with modular design. Each component has a single responsibility and can be extended independently.

---

Built with production-grade ML engineering practices. See `docs/SUMMARY.md` for technical details.
