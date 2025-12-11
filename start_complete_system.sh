#!/bin/bash
#
# DharmaMind Complete System Startup Script
# Starts all components: Backend, Frontend, External Gateway, and Monitoring
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Service configuration
BACKEND_PORT=8007
GATEWAY_PORT=8003
MAIN_FRONTEND_PORT=3000
BRAND_FRONTEND_PORT=3001
LOG_DIR="logs"

echo -e "${CYAN}🚀 DharmaMind Complete System Startup${NC}"
echo -e "${CYAN}=======================================${NC}"

# Create logs directory
mkdir -p $LOG_DIR

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}🔄 Killing existing process on port $port (PID: $pid)${NC}"
        kill -9 $pid
        sleep 2
    fi
}

# Function to start service
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="$LOG_DIR/${name}.log"
    
    echo -e "${BLUE}🔄 Starting $name on port $port...${NC}"
    
    # Kill existing process if any
    kill_port $port
    
    # Start the service
    eval "$command" > "$log_file" 2>&1 &
    local pid=$!
    
    # Wait a moment for service to start
    sleep 3
    
    # Check if service is running
    if kill -0 $pid 2>/dev/null && check_port $port; then
        echo -e "${GREEN}✅ $name started successfully (PID: $pid, Port: $port)${NC}"
        echo "   📄 Logs: $log_file"
        return 0
    else
        echo -e "${RED}❌ Failed to start $name${NC}"
        echo -e "${RED}   📄 Check logs: $log_file${NC}"
        return 1
    fi
}

# Function to check service health
check_health() {
    local name=$1
    local url=$2
    local timeout=${3:-5}
    
    echo -e "${BLUE}🩺 Checking $name health...${NC}"
    
    if curl -s --max-time $timeout "$url" > /dev/null; then
        echo -e "${GREEN}✅ $name is healthy${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  $name is not responding${NC}"
        return 1
    fi
}

echo -e "${PURPLE}Phase 1: Starting Backend Services${NC}"
echo "===================================="

# Start External LLM Gateway
start_service "External LLM Gateway" "cd llm-gateway && python3 main.py" $GATEWAY_PORT

# Start Main Backend
start_service "Main Backend" "python3 complete_integration.py" $BACKEND_PORT

echo ""
echo -e "${PURPLE}Phase 2: Starting Frontend Services${NC}"
echo "====================================="

# Start Main Chat Frontend
start_service "DharmaMind Chat" "cd dharmamind-chat && npm run dev" $MAIN_FRONTEND_PORT

# Start Brand Website
start_service "Brand Website" "cd Brand_Webpage && npm run dev" $BRAND_FRONTEND_PORT

echo ""
echo -e "${PURPLE}Phase 3: Health Checks${NC}"
echo "======================"

sleep 5  # Give services time to fully start

# Health checks
check_health "External LLM Gateway" "http://localhost:$GATEWAY_PORT/health"
check_health "Main Backend" "http://localhost:$BACKEND_PORT/system/status"
check_health "DharmaMind Chat" "http://localhost:$MAIN_FRONTEND_PORT"
check_health "Brand Website" "http://localhost:$BRAND_FRONTEND_PORT"

echo ""
echo -e "${CYAN}🌟 DharmaMind Complete System Status${NC}"
echo -e "${CYAN}====================================${NC}"

echo -e "${GREEN}🌉 External LLM Gateway:${NC} http://localhost:$GATEWAY_PORT"
echo -e "   📊 Health: http://localhost:$GATEWAY_PORT/health"
echo -e "   🔑 Providers: http://localhost:$GATEWAY_PORT/providers"

echo -e "${GREEN}🚀 Main Backend (Complete Integration):${NC} http://localhost:$BACKEND_PORT"
echo -e "   📊 Status: http://localhost:$BACKEND_PORT/system/status"
echo -e "   🤖 LLM Chat: http://localhost:$BACKEND_PORT/llm/chat"
echo -e "   📚 API Docs: http://localhost:$BACKEND_PORT/docs"

echo -e "${GREEN}💬 DharmaMind Chat Frontend:${NC} http://localhost:$MAIN_FRONTEND_PORT"
echo -e "   🧘‍♂️ Spiritual chat interface"
echo -e "   📱 PWA capabilities enabled"

echo -e "${GREEN}🌐 Brand Website:${NC} http://localhost:$BRAND_FRONTEND_PORT"
echo -e "   🏠 Landing page and marketing"
echo -e "   📈 SEO optimized"

echo ""
echo -e "${YELLOW}📋 Quick Test Commands:${NC}"
echo "========================"
echo "# Test LLM Chat:"
echo "curl -X POST 'http://localhost:$BACKEND_PORT/llm/chat' -H 'Content-Type: application/json' -d '{\"prompt\": \"How do I meditate?\", \"provider\": \"dharma_quantum\"}'"

echo ""
echo "# Test Providers:"
echo "curl -X GET 'http://localhost:$BACKEND_PORT/llm/providers'"

echo ""
echo "# Check Backend Status:"
echo "curl -X GET 'http://localhost:$BACKEND_PORT/system/status'"

echo ""
echo -e "${CYAN}📁 Log Files:${NC}"
echo "============="
echo "External Gateway: $LOG_DIR/External_LLM_Gateway.log"
echo "Main Backend: $LOG_DIR/Main_Backend.log"
echo "Chat Frontend: $LOG_DIR/DharmaMind_Chat.log"
echo "Brand Website: $LOG_DIR/Brand_Website.log"

echo ""
echo -e "${GREEN}✅ DharmaMind Complete System is now running!${NC}"
echo -e "${PURPLE}🧘‍♂️ May all beings find peace and wisdom through technology${NC}"

# Keep script running to show status
echo ""
echo -e "${BLUE}Press Ctrl+C to stop all services...${NC}"

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping all DharmaMind services...${NC}"
    
    kill_port $BRAND_FRONTEND_PORT
    kill_port $MAIN_FRONTEND_PORT
    kill_port $BACKEND_PORT
    kill_port $GATEWAY_PORT
    
    echo -e "${GREEN}✅ All services stopped. Until next time! 🙏${NC}"
    exit 0
}

trap cleanup SIGINT

# Wait indefinitely
while true; do
    sleep 60
    echo -e "${CYAN}$(date): DharmaMind Complete System running...${NC}"
done
