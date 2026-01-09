.PHONY: help install train predict api test clean docker-build docker-run format lint mlflow

# Variables
PYTHON := python
PIP := uv pip
DOCKER_IMAGE := delivery-ml
DOCKER_TAG := latest

# Help
help:
	@echo "Delivery Lateness Prediction - Makefile Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo ""
	@echo "Training:"
	@echo "  make train            Train model with full pipeline"
	@echo "  make train-quick      Train with fewer Optuna trials (faster)"
	@echo ""
	@echo "Prediction:"
	@echo "  make predict          Make predictions on test data"
	@echo "  make api              Start FastAPI prediction server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run training in Docker"
	@echo "  make docker-api       Run API in Docker"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow           Start MLflow UI"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           Format code with black and isort"
	@echo "  make lint             Run linting checks"
	@echo "  make test             Run tests (if available)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files"
	@echo "  make clean-all        Remove all generated files including models"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 pytest
	@echo "✓ Development dependencies installed"

# Training
train:
	@echo "Starting training pipeline..."
	$(PYTHON) train.py
	@echo "✓ Training complete"

train-quick:
	@echo "Starting quick training (10 trials)..."
	OPTUNA__N_TRIALS=10 $(PYTHON) train.py
	@echo "✓ Quick training complete"

# Prediction
predict:
	@echo "Making predictions on test data..."
	$(PYTHON) predict.py --input data/processed/test.csv --output predictions/test_predictions.csv
	@echo "✓ Predictions saved to predictions/test_predictions.csv"

predict-custom:
	@echo "Usage: make predict-custom INPUT=path/to/data.csv OUTPUT=path/to/predictions.csv"
	@if [ -z "$(INPUT)" ]; then \
		echo "ERROR: INPUT not specified"; \
		exit 1; \
	fi
	$(PYTHON) predict.py --input $(INPUT) --output $(or $(OUTPUT),predictions/predictions.csv)

# API
api:
	@echo "Starting FastAPI server..."
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	@echo "Starting FastAPI server (production mode)..."
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE):$(DOCKER_TAG)..."
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "✓ Docker image built"

docker-run:
	@echo "Running training in Docker..."
	docker run --rm -v $(PWD)/mlruns:/app/mlruns -v $(PWD)/models:/app/models $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "✓ Docker training complete"

docker-api:
	@echo "Starting API in Docker..."
	docker run --rm -p 8000:8000 -v $(PWD)/mlruns:/app/mlruns -v $(PWD)/models:/app/models $(DOCKER_IMAGE):$(DOCKER_TAG) python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

docker-compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "✓ Services started"

docker-compose-down:
	@echo "Stopping services..."
	docker-compose down
	@echo "✓ Services stopped"

# MLflow
mlflow:
	@echo "Starting MLflow UI at http://localhost:5000..."
	@if [ ! -d mlruns ]; then mkdir -p mlruns; fi
	mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 8001

# Code Quality
format:
	@echo "Formatting code..."
	black src/ train.py predict.py
	isort src/ train.py predict.py
	@echo "✓ Code formatted"

lint:
	@echo "Running linting checks..."
	flake8 src/ train.py predict.py --max-line-length=100 --extend-ignore=E203,W503
	@echo "✓ Linting complete"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "✓ Tests complete"

# Cleanup
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf logs/*.log
	rm -rf plots/*.png
	@echo "✓ Cleaned"

clean-all: clean
	@echo "Removing all generated files including models..."
	rm -rf mlruns/
	rm -rf models/
	rm -rf data/processed/
	rm -rf predictions/
	rm -rf plots/
	@echo "✓ All generated files removed"

# Data validation
validate-data:
	@echo "Validating data..."
	$(PYTHON) -c "from src.data.loader import DataLoader; from src.data.validation import DataValidator; from src.config import settings; loader = DataLoader(); df = loader.load_raw_data(settings.data.data_path); validator = DataValidator(); result = validator.validate(df); print('Validation:', 'PASSED' if result['is_valid'] else 'FAILED')"

# Directory setup
setup-dirs:
	@echo "Creating project directories..."
	mkdir -p data/processed
	mkdir -p mlruns
	mkdir -p models
	mkdir -p logs
	mkdir -p plots
	mkdir -p predictions
	@echo "✓ Directories created"

# Quick start
quickstart: setup-dirs install train mlflow-ui
	@echo "✓ Quick start complete! MLflow UI should be running at http://localhost:5000"

# All-in-one: install, train, and start API
all: install train api
	@echo "✓ Full pipeline complete"
