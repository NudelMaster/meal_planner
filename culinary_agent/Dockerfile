# Multi-stage Dockerfile for Culinary Agent

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Copy source code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files
COPY src/ /app/src/
COPY config/ /app/config/
COPY data/ /app/data/
COPY indices/ /app/indices/
COPY .env.example /app/.env.example

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with production server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
