#!/bin/bash

# =============================================================================
# DharmaMind Quick Start Script - Enterprise Authentication Platform
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🧘 DharmaMind Enterprise Authentication Platform${NC}"
echo -e "${PURPLE}=================================================${NC}"
echo ""

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Windows (Git Bash/WSL)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    IS_WINDOWS=true
else
    IS_WINDOWS=false
fi

# Setup function
setup_environment() {
    log_info "Setting up DharmaMind environment..."
    
    # Create necessary directories
    mkdir -p data logs temp uploads
    
    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "Created .env file from example"
            log_warning "Please edit .env with your configuration before starting!"
        else
            log_error ".env.example not found"
            return 1
        fi
    fi
    
    log_success "Environment setup complete"
}

# Start backend function
start_backend() {
    log_info "Starting DharmaMind Enhanced Authentication Backend..."
    
    if [ ! -d "backend/app" ]; then
        log_error "Backend directory not found"
        return 1
    fi
    
    cd backend/app
    
    # Check if Python is available
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found. Please install Python 3.11+"
        return 1
    fi
    
    # Check if enhanced auth backend exists
    if [ -f "enhanced_enterprise_auth.py" ]; then
        log_info "Starting enhanced enterprise authentication server..."
        $PYTHON_CMD enhanced_enterprise_auth.py &
        BACKEND_PID=$!
        echo $BACKEND_PID > ../../backend.pid
        log_success "Backend started on port 5001 (PID: $BACKEND_PID)"
    else
        log_error "Enhanced authentication backend not found"
        return 1
    fi
    
    cd ../..
}

# Start frontend function
start_frontend() {
    log_info "Starting DharmaMind Frontend..."
    
    if [ ! -d "frontend" ]; then
        log_error "Frontend directory not found"
        return 1
    fi
    
    cd frontend
    
    # Check if Node.js is available
    if ! command -v npm &> /dev/null; then
        log_error "npm not found. Please install Node.js 18+"
        return 1
    fi
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi
    
    # Start development server
    log_info "Starting Next.js development server..."
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../frontend.pid
    log_success "Frontend started on port 3000 (PID: $FRONTEND_PID)"
    
    cd ..
}

# Docker function
start_docker() {
    log_info "Starting DharmaMind with Docker..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker"
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose"
        return 1
    fi
    
    # Check for environment file
    if [ ! -f ".env" ]; then
        log_warning "No .env file found. Using example configuration..."
        cp .env.example .env
    fi
    
    # Start with Docker Compose
    log_info "Starting services with Docker Compose..."
    docker-compose up -d
    
    log_success "Docker services started"
    log_info "Access points:"
    echo "  • Frontend: http://localhost:3000"
    echo "  • Backend API: http://localhost:8000"
    echo "  • API Docs: http://localhost:8000/docs"
}

# Production deployment
deploy_production() {
    log_info "Deploying DharmaMind to Production..."
    
    if [ ! -f "deploy.sh" ]; then
        log_error "Production deployment script not found"
        return 1
    fi
    
    # Make deploy script executable
    chmod +x deploy.sh
    
    # Run deployment
    ./deploy.sh deploy
}

# Stop services function
stop_services() {
    log_info "Stopping DharmaMind services..."
    
    # Stop backend
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            log_success "Backend stopped"
        fi
        rm -f backend.pid
    fi
    
    # Stop frontend
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            log_success "Frontend stopped"
        fi
        rm -f frontend.pid
    fi
    
    # Stop Docker services
    if command -v docker-compose &> /dev/null; then
        docker-compose down
        log_success "Docker services stopped"
    fi
}

# Status function
show_status() {
    log_info "DharmaMind Service Status:"
    echo ""
    
    # Check backend
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Backend (PID: $BACKEND_PID) - http://localhost:5001"
        else
            echo -e "  ${RED}✗${NC} Backend (not running)"
        fi
    else
        echo -e "  ${RED}✗${NC} Backend (not started)"
    fi
    
    # Check frontend
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Frontend (PID: $FRONTEND_PID) - http://localhost:3000"
        else
            echo -e "  ${RED}✗${NC} Frontend (not running)"
        fi
    else
        echo -e "  ${RED}✗${NC} Frontend (not started)"
    fi
    
    # Check Docker
    if command -v docker-compose &> /dev/null; then
        if docker-compose ps | grep -q "Up"; then
            echo -e "  ${GREEN}✓${NC} Docker services running"
        else
            echo -e "  ${YELLOW}○${NC} Docker services available but not running"
        fi
    fi
    
    echo ""
    echo -e "${CYAN}Access Points:${NC}"
    echo "  • Frontend: http://localhost:3000"
    echo "  • Backend API: http://localhost:5001"
    echo "  • API Documentation: http://localhost:5001/docs"
    echo "  • Test Interface: http://localhost:3000/auth-test"
    echo ""
    echo -e "${CYAN}Demo Accounts:${NC}"
    echo "  • Free: demo.free@dharmamind.ai / demo123"
    echo "  • Premium: demo.premium@dharmamind.ai / demo123"
    echo "  • Enterprise: demo.enterprise@dharmamind.ai / demo123"
}

# Show help
show_help() {
    echo -e "${CYAN}DharmaMind Quick Start Commands:${NC}"
    echo ""
    echo "  $0 setup         - Setup environment and configuration"
    echo "  $0 dev           - Start development servers (backend + frontend)"
    echo "  $0 backend       - Start only backend server"
    echo "  $0 frontend      - Start only frontend server"
    echo "  $0 docker        - Start with Docker Compose"
    echo "  $0 production    - Deploy to production"
    echo "  $0 stop          - Stop all services"
    echo "  $0 status        - Show service status"
    echo "  $0 help          - Show this help"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  # Quick development start"
    echo "  $0 setup && $0 dev"
    echo ""
    echo "  # Production deployment"
    echo "  $0 production"
    echo ""
    echo "  # Docker development"
    echo "  $0 docker"
}

# Main script logic
case "${1:-dev}" in
    "setup")
        setup_environment
        ;;
    "dev")
        setup_environment
        start_backend
        sleep 3
        start_frontend
        show_status
        ;;
    "backend")
        setup_environment
        start_backend
        show_status
        ;;
    "frontend")
        start_frontend
        show_status
        ;;
    "docker")
        start_docker
        ;;
    "production")
        deploy_production
        ;;
    "stop")
        stop_services
        ;;
    "status")
        show_status
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
