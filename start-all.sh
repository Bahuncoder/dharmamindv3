#!/bin/bash
#
# DharmaMind - Start All Services
# ================================
# This script starts all DharmaMind services in the correct order
#

echo "========================================"
echo "  DharmaMind Platform Startup"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any existing processes
echo -e "${YELLOW}Stopping existing services...${NC}"
pkill -f "uvicorn.*8000" 2>/dev/null
pkill -f "uvicorn.*8001" 2>/dev/null
pkill -f "next.*3000" 2>/dev/null
pkill -f "next.*3001" 2>/dev/null
pkill -f "next.*3002" 2>/dev/null
sleep 2

echo ""
echo "Starting services..."
echo ""

# 1. Start DharmaLLM (AI Service) on port 8001
echo -e "${GREEN}[1/5] Starting DharmaLLM (AI Service) on port 8001...${NC}"
cd "$BASE_DIR/dharmallm"
if [ -d "../dharmallm_env" ]; then
    source ../dharmallm_env/bin/activate
fi
PYTHONPATH="." nohup python -m uvicorn api.production_main:app --host 0.0.0.0 --port 8001 > /tmp/dharmallm.log 2>&1 &
echo "  PID: $!"
sleep 3

# 2. Start Main Backend (API Gateway) on port 8000
echo -e "${GREEN}[2/5] Starting Main Backend (API Gateway) on port 8000...${NC}"
cd "$BASE_DIR/backend"
if [ -d "../backend_env" ]; then
    source ../backend_env/bin/activate
fi
PYTHONPATH="." nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
echo "  PID: $!"
sleep 3

# 3. Start Brand Webpage on port 3001
echo -e "${GREEN}[3/5] Starting Brand Webpage on port 3001...${NC}"
cd "$BASE_DIR/Brand_Webpage"
nohup npm run dev -- --port 3001 > /tmp/brand.log 2>&1 &
echo "  PID: $!"
sleep 2

# 4. Start Community on port 3002
echo -e "${GREEN}[4/5] Starting Community on port 3002...${NC}"
cd "$BASE_DIR/DhramaMind_Community"
nohup npm run dev -- --port 3002 > /tmp/community.log 2>&1 &
echo "  PID: $!"
sleep 2

# 5. Start Chat on port 3000
echo -e "${GREEN}[5/5] Starting Chat on port 3000...${NC}"
cd "$BASE_DIR/dharmamind-chat"
nohup npm run dev -- --port 3000 > /tmp/chat.log 2>&1 &
echo "  PID: $!"
sleep 2

echo ""
echo "========================================"
echo -e "${GREEN}  All Services Started!${NC}"
echo "========================================"
echo ""
echo "Services:"
echo "  - DharmaLLM (AI):      http://localhost:8001"
echo "  - API Gateway:         http://localhost:8000"
echo "  - Brand Webpage:       http://localhost:3001"
echo "  - Community:           http://localhost:3002"
echo "  - Chat:                http://localhost:3000"
echo ""
echo "API Documentation:       http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  - DharmaLLM:  /tmp/dharmallm.log"
echo "  - Backend:    /tmp/backend.log"
echo "  - Brand:      /tmp/brand.log"
echo "  - Community:  /tmp/community.log"
echo "  - Chat:       /tmp/chat.log"
echo ""
echo "To stop all: pkill -f 'uvicorn|next'"
echo "========================================"

