# ============================================================================
# Stage 1: Base image with Python and UV
# ============================================================================
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies as app

# Copy source code
COPY src/ ./src/
COPY train.py .
COPY predict.py .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p data models mlruns logs plots predictions

# Set Python path
ENV PYTHONPATH=/app

# Default command: run training
CMD ["python", "train.py"]

# ============================================================================
# Stage 4: API (for production API serving)
# ============================================================================
FROM app as api

# Expose API port
EXPOSE 8000

# Command for API
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
