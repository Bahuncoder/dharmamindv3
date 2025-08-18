#!/bin/bash

# LLM Gateway Startup Script

echo "🔐 Starting LLM Gateway Service..."
echo "======================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Check for environment file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📋 Copying .env.example to .env..."
        cp .env.example .env
        echo "⚠️  Please configure your API keys in .env file"
    else
        echo "⚠️  No .env file found. Please create one with your API keys."
    fi
fi

# Start the service
echo "🚀 Starting LLM Gateway on port 8003..."
echo "📡 Access at: http://localhost:8003"
echo "📖 Documentation: http://localhost:8003/docs"
echo "======================================="

python main.py
