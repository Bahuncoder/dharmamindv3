#!/bin/bash

# 🚀 DharmaMind Quick Deployment Script
# This script provides automated deployment for the complete DharmaMind system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_info "Checking system requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if ports are available
    if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port 80 is already in use. Please stop the service using this port."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "System requirements check passed"
}

setup_environment() {
    print_info "Setting up environment variables..."
    
    if [ ! -f .env ]; then
        if [ -f .env.production ]; then
            cp .env.production .env
            print_info "Copied .env.production to .env"
        else
            print_error ".env.production template not found. Please create environment configuration."
            exit 1
        fi
    fi
    
    print_warning "Please review and update .env file with your actual configuration:"
    print_warning "- Database credentials"
    print_warning "- API keys"
    print_warning "- Security secrets"
    print_warning "- Domain configuration"
    
    read -p "Have you updated the .env file? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Please update .env file and run this script again."
        exit 1
    fi
    
    print_success "Environment configuration ready"
}

build_services() {
    print_info "Building Docker services..."
    
    # Build all services
    docker-compose build --parallel
    
    print_success "All services built successfully"
}

start_services() {
    print_info "Starting DharmaMind services..."
    
    # Start services in correct order
    print_info "Starting database and cache services..."
    docker-compose up -d postgres redis
    sleep 10  # Wait for database to be ready
    
    print_info "Starting backend API..."
    docker-compose up -d backend
    sleep 15  # Wait for backend to be ready
    
    print_info "Starting frontend services..."
    docker-compose up -d brand_website community
    sleep 10
    
    print_info "Starting monitoring services..."
    docker-compose up -d prometheus grafana
    sleep 5
    
    print_info "Starting reverse proxy..."
    docker-compose up -d nginx
    
    print_success "All services started successfully"
}

run_migrations() {
    print_info "Running database migrations..."
    
    # Wait a bit more for backend to be fully ready
    sleep 10
    
    # Run database migrations
    docker-compose exec -T backend python -m alembic upgrade head
    
    print_success "Database migrations completed"
}

validate_deployment() {
    print_info "Validating deployment..."
    
    # Wait for all services to be fully ready
    sleep 20
    
    # Test health endpoints
    print_info "Testing health endpoints..."
    
    # Nginx health check
    if curl -f -s http://localhost/health >/dev/null 2>&1; then
        print_success "✓ Nginx health check passed"
    else
        print_error "✗ Nginx health check failed"
    fi
    
    # Backend health check
    if curl -f -s http://localhost/api/health >/dev/null 2>&1; then
        print_success "✓ Backend API health check passed"
    else
        print_error "✗ Backend API health check failed"
    fi
    
    # Check if all containers are running
    print_info "Checking container status..."
    docker-compose ps
    
    print_success "Deployment validation completed"
}

show_access_info() {
    print_success "\n🎉 DharmaMind deployment completed successfully!"
    
    echo -e "\n${GREEN}🌐 Access URLs:${NC}"
    echo -e "${BLUE}Main Website:${NC} http://localhost"
    echo -e "${BLUE}API Documentation:${NC} http://localhost/api/docs"
    echo -e "${BLUE}Community Platform:${NC} http://localhost:3002"
    echo -e "${BLUE}Grafana Dashboard:${NC} http://localhost:3000 (admin/admin)"
    echo -e "${BLUE}Prometheus Metrics:${NC} http://localhost:9090"
    
    echo -e "\n${GREEN}🔧 Management Commands:${NC}"
    echo -e "${BLUE}View logs:${NC} docker-compose logs -f [service_name]"
    echo -e "${BLUE}Stop services:${NC} docker-compose down"
    echo -e "${BLUE}Restart services:${NC} docker-compose restart [service_name]"
    echo -e "${BLUE}Check status:${NC} docker-compose ps"
    
    echo -e "\n${GREEN}🧪 Testing:${NC}"
    echo -e "${BLUE}Run tests:${NC} cd backend && python -m pytest tests/ -v"
    
    echo -e "\n${YELLOW}📖 For detailed documentation, see:${NC}"
    echo -e "- COMPLETE_DEPLOYMENT_GUIDE.md"
    echo -e "- TESTING_INFRASTRUCTURE_COMPLETE.md"
    echo -e "- MONITORING_SYSTEM_ANALYSIS.md"
}

cleanup_on_error() {
    print_error "Deployment failed. Cleaning up..."
    docker-compose down
    exit 1
}

# Main deployment flow
main() {
    echo -e "${GREEN}"
    echo "  ____  _                           __  __ _           _ "
    echo " |  _ \\| |__   __ _ _ __ _ __ ___   |  \\/  (_)_ __   __| |"
    echo " | | | | '_ \\ / _\` | '__| '_ \` _ \\  | |\\/| | | '_ \\ / _\` |"
    echo " | |_| | | | | (_| | |  | | | | | | | |  | | | | | | (_| |"
    echo " |____/|_| |_|\\__,_|_|  |_| |_| |_| |_|  |_|_|_| |_|\\__,_|"
    echo -e "${NC}"
    echo -e "${BLUE}🚀 DharmaMind Quick Deployment Script${NC}"
    echo -e "${BLUE}======================================${NC}\n"
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Deployment steps
    check_requirements
    setup_environment
    build_services
    start_services
    run_migrations
    validate_deployment
    show_access_info
    
    print_success "\n🎯 Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-}" in
    "help" | "-h" | "--help")
        echo "DharmaMind Quick Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  help    Show this help message"
        echo "  stop    Stop all services"
        echo "  logs    Show logs for all services"
        echo "  status  Show status of all services"
        echo ""
        echo "Default: Run full deployment"
        exit 0
        ;;
    "stop")
        print_info "Stopping DharmaMind services..."
        docker-compose down
        print_success "All services stopped"
        exit 0
        ;;
    "logs")
        docker-compose logs -f
        exit 0
        ;;
    "status")
        docker-compose ps
        exit 0
        ;;
    *)
        main
        ;;
esac
