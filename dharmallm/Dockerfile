# 🕉️ DharmaLLM Spiritual AI Service
# Containerized DharmaLLM service for spiritual wisdom processing
FROM python:3.11-slim

LABEL maintainer="DharmaMind AI Team"
LABEL description="DharmaLLM Spiritual AI Service - Dharmic Wisdom Processing"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/cache /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash dharma
RUN chown -R dharma:dharma /app
USER dharma

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Start command
CMD ["python", "-m", "dharmallm.api.main"]
