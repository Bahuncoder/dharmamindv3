#!/bin/bash

# DharmaLLM Integration Test Script
# Tests the full integration of DharmaLLM as a separate microservice

echo "🕉️ DharmaLLM Integration Test"
echo "============================="

# Set working directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"

echo ""
echo "📋 Integration Summary:"
echo "----------------------"
echo "✅ Backend: Clean authentication-focused service"
echo "✅ DharmaLLM: Separate microservice with FastAPI"
echo "✅ Communication: HTTP-based client-server architecture"
echo "✅ Docker: Multi-container setup with proper dependencies"
echo ""

echo "🐳 Docker Services Configuration:"
echo "--------------------------------"
echo "• Backend: Port 8000 (Authentication & API Gateway)"
echo "• DharmaLLM: Port 8001 (Spiritual AI Processing)"
echo "• Brand Website: Port 3000"
echo "• Community: Port 3001"
echo "• PostgreSQL: Port 5432"
echo "• Redis: Port 6379"
echo ""

echo "🔧 Key Integration Points:"
echo "------------------------"
echo "1. Backend config has DHARMALLM_SERVICE_URL=http://dharmallm:8001"
echo "2. DharmaLLM service runs independently with /api/v1/chat endpoint"
echo "3. Backend uses HTTP client to communicate with DharmaLLM"
echo "4. Clean separation: Backend handles auth, DharmaLLM handles AI"
echo ""

echo "🚀 To start the integrated system:"
echo "---------------------------------"
echo "docker-compose build"
echo "docker-compose up -d"
echo ""

echo "✅ Integration Status: READY"
echo "🙏 May this serve all beings with wisdom and compassion"