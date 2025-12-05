#!/bin/bash

# DharmaMind Backend Quick Start
# Installs essential dependencies and starts the integrated backend

echo "🕉️ Starting DharmaMind Integrated Backend..."

# Navigate to backend directory
cd "$(dirname "$0")"

echo "📦 Installing essential dependencies..."

# Install core dependencies with minimal packages
pip3 install --user --break-system-packages \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    email-validator \
    passlib \
    python-jose \
    python-multipart

echo "🚀 Starting DharmaMind backend with DharmaLLM integration..."

# Set Python path to include dharmallm
export PYTHONPATH="${PWD}/../dharmallm:${PYTHONPATH}"

# Start the server
python3 run_server.py