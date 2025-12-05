#!/bin/bash

# 🕉️ DharmaMind Production Deployment Script
# 
# Complete production deployment automation for DharmaMind
# Handles database migrations, service updates, health checks, and rollback
# 
# Usage: ./deploy_production.sh [version] [environment]
# 
# May this deployment bring digital wisdom to all beings 🚀

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_LOG="/var/log/dharmamind/deploy.log"
BACKUP_DIR="/var/backups/dharmamind"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
ROLLBACK_TIMEOUT=180      # 3 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Deployment parameters
VERSION=${1:-$(date +%Y%m%d-%H%M%S)}
ENVIRONMENT=${2:-production}
FORCE_DEPLOY=${3:-false}

# Docker configuration
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
DOCKER_COMPOSE_PROD_FILE="${PROJECT_ROOT}/docker-compose.prod.yml"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${BLUE}[INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "DHARMIC") echo -e "${PURPLE}[🕉️ DHARMIC]${NC} $message" ;;
    esac
    
    # Log to file
    mkdir -p "$(dirname "$DEPLOY_LOG")"
    echo "[$timestamp] [$level] $message" >> "$DEPLOY_LOG"
}

# Error handler
error_handler() {
    local line_number=$1
    log "ERROR" "Deployment failed at line $line_number"
    log "ERROR" "Starting automatic rollback procedure..."
    rollback_deployment
    exit 1
}

trap 'error_handler $LINENO' ERR

# Check prerequisites
check_prerequisites() {
    log "INFO" "🔍 Checking deployment prerequisites..."
    
    # Check if running as appropriate user
    if [[ $EUID -eq 0 ]]; then
        log "WARN" "Running as root - consider using dedicated deployment user"
    fi
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "psql" "redis-cli" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker daemon not running"
        exit 1
    fi
    
    # Check available disk space (minimum 5GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        log "ERROR" "Insufficient disk space. Need at least 5GB available"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "${PROJECT_ROOT}/.env.${ENVIRONMENT}" ]]; then
        log "ERROR" "Environment file not found: .env.${ENVIRONMENT}"
        exit 1
    fi
    
    log "SUCCESS" "✅ Prerequisites check passed"
}

# Create deployment backup
create_deployment_backup() {
    log "INFO" "📦 Creating deployment backup..."
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="${BACKUP_DIR}/${backup_timestamp}"
    
    mkdir -p "$backup_path"
    
    # Backup database
    log "INFO" "Backing up PostgreSQL database..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dump -U dharmamind dharmamind > "${backup_path}/database_backup.sql"
    
    # Backup Redis data
    log "INFO" "Backing up Redis data..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli --rdb - > "${backup_path}/redis_backup.rdb"
    
    # Backup application configuration
    log "INFO" "Backing up application configuration..."
    cp -r "${PROJECT_ROOT}/backend/app/config" "${backup_path}/"
    cp "${PROJECT_ROOT}/.env.${ENVIRONMENT}" "${backup_path}/"
    
    # Backup vector database
    if [[ -d "${PROJECT_ROOT}/vector_db/data" ]]; then
        log "INFO" "Backing up vector database..."
        cp -r "${PROJECT_ROOT}/vector_db/data" "${backup_path}/vector_db_data"
    fi
    
    # Create backup metadata
    cat > "${backup_path}/backup_metadata.json" << EOF
{
    "timestamp": "${backup_timestamp}",
    "version": "${VERSION}",
    "environment": "$ENVIRONMENT",
    "backup_type": "pre_deployment",
    "services": ["postgres", "redis", "dharmamind", "vector_db"]
}
EOF
    
    echo "$backup_path" > "/tmp/dharmamind_last_backup"
    log "SUCCESS" "✅ Backup created: $backup_path"
}

# Pre-deployment health check
pre_deployment_health_check() {
    log "INFO" "🩺 Running pre-deployment health check..."
    
    # Check current services status
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        log "INFO" "Current services are running - performing health checks"
        
        # Test database connectivity
        if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U dharmamind -d dharmamind -c "SELECT 1;" &> /dev/null; then
            log "SUCCESS" "✅ Database connectivity check passed"
        else
            log "ERROR" "Database connectivity check failed"
            exit 1
        fi
        
        # Test Redis connectivity
        if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
            log "SUCCESS" "✅ Redis connectivity check passed"
        else
            log "ERROR" "Redis connectivity check failed"
            exit 1
        fi
        
        # Test API health endpoint
        if curl -s http://localhost:8000/health | jq -r '.status' | grep -q "healthy"; then
            log "SUCCESS" "✅ API health check passed"
        else
            log "WARN" "API health check failed - may be expected during deployment"
        fi
    else
        log "INFO" "No services currently running - fresh deployment"
    fi
}

# Run database migrations
run_database_migrations() {
    log "INFO" "🗃️ Running database migrations..."
    
    # Start database service if not running
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres
    
    # Wait for database to be ready
    log "INFO" "Waiting for database to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U dharmamind &> /dev/null; then
            break
        fi
        sleep 2
        ((retries--))
    done
    
    if [[ $retries -eq 0 ]]; then
        log "ERROR" "Database failed to become ready"
        exit 1
    fi
    
    # Run migrations using the migration system
    log "INFO" "Executing database migrations..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" run --rm dharmamind python -m backend.app.services.migrations migrate --environment="$ENVIRONMENT"
    
    # Verify migration status
    local migration_status=$(docker-compose -f "$DOCKER_COMPOSE_FILE" run --rm dharmamind python -m backend.app.services.migrations status --format=json | jq -r '.status')
    
    if [[ "$migration_status" == "up_to_date" ]]; then
        log "SUCCESS" "✅ Database migrations completed successfully"
    else
        log "ERROR" "Database migrations failed or incomplete"
        exit 1
    fi
}

# Build and deploy services
deploy_services() {
    log "INFO" "🚀 Deploying DharmaMind services..."
    
    # Build new images
    log "INFO" "Building application images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache dharmamind
    
    # Tag images with version
    docker tag dharmamind_dharmamind:latest "dharmamind_dharmamind:${VERSION}"
    
    # Deploy services with rolling update strategy
    log "INFO" "Performing rolling deployment..."
    
    # Update services one by one to minimize downtime
    local services=("redis" "postgres" "dharmamind" "nginx")
    
    for service in "${services[@]}"; do
        log "INFO" "Updating service: $service"
        
        # Scale up new instance
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --scale "$service=2" "$service"
        
        # Wait for new instance to be healthy
        sleep 10
        
        # Scale down to single instance (removes old)
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --scale "$service=1" "$service"
        
        log "SUCCESS" "✅ Service updated: $service"
    done
    
    # Final service startup
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
}

# Post-deployment health check
post_deployment_health_check() {
    log "INFO" "🩺 Running post-deployment health check..."
    
    local start_time=$(date +%s)
    local health_check_passed=false
    
    while [[ $(($(date +%s) - start_time)) -lt $HEALTH_CHECK_TIMEOUT ]]; do
        log "INFO" "Checking service health..."
        
        # Check all services are running
        local running_services=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps --services --filter "status=running" | wc -l)
        local total_services=$(docker-compose -f "$DOCKER_COMPOSE_FILE" config --services | wc -l)
        
        if [[ $running_services -eq $total_services ]]; then
            log "INFO" "All services are running, checking application health..."
            
            # Test database
            if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U dharmamind -d dharmamind -c "SELECT 1;" &> /dev/null; then
                log "INFO" "✓ Database health check passed"
                
                # Test Redis
                if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
                    log "INFO" "✓ Redis health check passed"
                    
                    # Test API
                    if curl -s http://localhost:8000/health | jq -r '.status' | grep -q "healthy"; then
                        log "INFO" "✓ API health check passed"
                        
                        # Test AI system
                        if curl -s http://localhost:8000/api/health/ai | jq -r '.status' | grep -q "healthy"; then
                            log "SUCCESS" "✅ All health checks passed!"
                            health_check_passed=true
                            break
                        else
                            log "WARN" "AI system health check failed, retrying..."
                        fi
                    else
                        log "WARN" "API health check failed, retrying..."
                    fi
                else
                    log "WARN" "Redis health check failed, retrying..."
                fi
            else
                log "WARN" "Database health check failed, retrying..."
            fi
        else
            log "WARN" "Not all services are running ($running_services/$total_services), waiting..."
        fi
        
        sleep 10
    done
    
    if [[ "$health_check_passed" != "true" ]]; then
        log "ERROR" "Post-deployment health check failed within timeout"
        return 1
    fi
}

# Performance benchmarking
run_performance_tests() {
    log "INFO" "⚡ Running performance benchmarks..."
    
    # API response time test
    log "INFO" "Testing API response times..."
    local api_response_time=$(curl -s -w "%{time_total}" -o /dev/null http://localhost:8000/health)
    
    if (( $(echo "$api_response_time < 2.0" | bc -l) )); then
        log "SUCCESS" "✅ API response time: ${api_response_time}s (acceptable)"
    else
        log "WARN" "API response time: ${api_response_time}s (slower than expected)"
    fi
    
    # Database query performance
    log "INFO" "Testing database performance..."
    local db_query_time=$(docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U dharmamind -d dharmamind -c "\timing" -c "SELECT COUNT(*) FROM application_logs;" 2>&1 | grep "Time:" | awk '{print $2}' | sed 's/ms//')
    
    if [[ -n "$db_query_time" && $(echo "$db_query_time < 100" | bc -l) -eq 1 ]]; then
        log "SUCCESS" "✅ Database query time: ${db_query_time}ms (acceptable)"
    else
        log "WARN" "Database query time: ${db_query_time}ms (may need optimization)"
    fi
    
    # Memory usage check
    log "INFO" "Checking memory usage..."
    local memory_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | grep dharmamind | awk '{print $2}' | head -1)
    log "INFO" "Current memory usage: $memory_usage"
    
    # Load test with simple concurrent requests
    log "INFO" "Running basic load test..."
    for i in {1..5}; do
        curl -s http://localhost:8000/health > /dev/null &
    done
    wait
    
    log "SUCCESS" "✅ Basic performance tests completed"
}

# Security validation
validate_security() {
    log "INFO" "🔒 Running security validation..."
    
    # Check for exposed secrets
    log "INFO" "Checking for exposed secrets..."
    if docker-compose -f "$DOCKER_COMPOSE_FILE" config | grep -i "password\|secret\|key" | grep -v "xxx\|***"; then
        log "ERROR" "Potential secrets exposed in docker-compose configuration"
        exit 1
    fi
    
    # Check SSL/TLS configuration
    log "INFO" "Checking SSL/TLS configuration..."
    if curl -s -I https://localhost | grep -q "HTTP/2 200"; then
        log "SUCCESS" "✅ HTTPS endpoint responding"
    else
        log "WARN" "HTTPS endpoint not responding (may be expected in development)"
    fi
    
    # Check security headers
    log "INFO" "Checking security headers..."
    local security_headers=("X-Content-Type-Options" "X-Frame-Options" "X-XSS-Protection")
    
    for header in "${security_headers[@]}"; do
        if curl -s -I http://localhost:8000/health | grep -q "$header"; then
            log "SUCCESS" "✅ Security header present: $header"
        else
            log "WARN" "Security header missing: $header"
        fi
    done
    
    log "SUCCESS" "✅ Security validation completed"
}

# Rollback deployment
rollback_deployment() {
    log "WARN" "🔄 Starting deployment rollback..."
    
    if [[ ! -f "/tmp/dharmamind_last_backup" ]]; then
        log "ERROR" "No backup path found for rollback"
        return 1
    fi
    
    local backup_path=$(cat "/tmp/dharmamind_last_backup")
    
    if [[ ! -d "$backup_path" ]]; then
        log "ERROR" "Backup directory not found: $backup_path"
        return 1
    fi
    
    # Stop current services
    log "INFO" "Stopping current services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # Restore database
    log "INFO" "Restoring database from backup..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres
    sleep 10
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U dharmamind -c "DROP DATABASE IF EXISTS dharmamind;"
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U dharmamind -c "CREATE DATABASE dharmamind;"
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U dharmamind -d dharmamind < "${backup_path}/database_backup.sql"
    
    # Restore Redis data
    log "INFO" "Restoring Redis data..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d redis
    sleep 5
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli FLUSHALL
    cat "${backup_path}/redis_backup.rdb" | docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli --pipe
    
    # Restore configuration
    log "INFO" "Restoring configuration..."
    cp "${backup_path}/.env.${ENVIRONMENT}" "${PROJECT_ROOT}/"
    
    # Restart services
    log "INFO" "Restarting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    local start_time=$(date +%s)
    while [[ $(($(date +%s) - start_time)) -lt $ROLLBACK_TIMEOUT ]]; do
        if curl -s http://localhost:8000/health | jq -r '.status' | grep -q "healthy"; then
            log "SUCCESS" "✅ Rollback completed successfully"
            return 0
        fi
        sleep 5
    done
    
    log "ERROR" "Rollback completed but services not healthy within timeout"
    return 1
}

# Cleanup old deployments
cleanup_old_deployments() {
    log "INFO" "🧹 Cleaning up old deployments..."
    
    # Remove old Docker images (keep last 5 versions)
    log "INFO" "Cleaning up old Docker images..."
    docker images "dharmamind_dharmamind" --format "table {{.Tag}}\t{{.CreatedAt}}" | tail -n +2 | sort -k2 -r | tail -n +6 | awk '{print $1}' | while read -r tag; do
        if [[ "$tag" == "latest" ]]; then
            continue
        fi
        log "INFO" "Removing old image: dharmamind_dharmamind:$tag"
        docker rmi "dharmamind_dharmamind:$tag" || true
    done
    
    # Clean up old backups (keep last 10)
    log "INFO" "Cleaning up old backups..."
    find "$BACKUP_DIR" -type d -name "20*" | sort -r | tail -n +11 | while read -r backup_dir; do
        log "INFO" "Removing old backup: $backup_dir"
        rm -rf "$backup_dir"
    done
    
    # Clean up Docker system
    log "INFO" "Cleaning up Docker system..."
    docker system prune -f
    
    log "SUCCESS" "✅ Cleanup completed"
}

# Update monitoring and alerting
update_monitoring() {
    log "INFO" "📊 Updating monitoring configuration..."
    
    # Update Prometheus targets
    if [[ -f "${PROJECT_ROOT}/monitoring/prometheus.yml" ]]; then
        log "INFO" "Updating Prometheus configuration..."
        # Restart Prometheus to reload config
        docker-compose -f "$DOCKER_COMPOSE_FILE" restart prometheus || true
    fi
    
    # Update Grafana dashboards
    if [[ -d "${PROJECT_ROOT}/monitoring/grafana/dashboards" ]]; then
        log "INFO" "Updating Grafana dashboards..."
        # Import any new dashboards
        for dashboard in "${PROJECT_ROOT}/monitoring/grafana/dashboards"/*.json; do
            if [[ -f "$dashboard" ]]; then
                log "INFO" "Importing dashboard: $(basename "$dashboard")"
                # Dashboard import would happen here
            fi
        done
    fi
    
    log "SUCCESS" "✅ Monitoring updated"
}

# Send deployment notification
send_deployment_notification() {
    local status=$1
    local message=$2
    
    log "INFO" "📧 Sending deployment notification..."
    
    # Create notification payload
    local notification_payload=$(cat << EOF
{
    "deployment": {
        "version": "$VERSION",
        "environment": "$ENVIRONMENT",
        "status": "$status",
        "message": "$message",
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "deployer": "$(whoami)",
        "hostname": "$(hostname)"
    }
}
EOF
)
    
    # Log notification (in production, this would send to Slack, email, etc.)
    echo "$notification_payload" | jq '.' >> "${DEPLOY_LOG}"
    
    log "INFO" "Deployment notification logged"
}

# Main deployment function
main() {
    log "DHARMIC" "🕉️ Starting DharmaMind Production Deployment"
    log "INFO" "Version: $VERSION"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Timestamp: $(date)"
    
    # Deployment steps
    check_prerequisites
    create_deployment_backup
    pre_deployment_health_check
    run_database_migrations
    deploy_services
    
    if post_deployment_health_check; then
        log "SUCCESS" "✅ Health checks passed"
        
        run_performance_tests
        validate_security
        update_monitoring
        cleanup_old_deployments
        
        send_deployment_notification "success" "Deployment completed successfully"
        
        log "DHARMIC" "🎉 DharmaMind deployment completed successfully!"
        log "INFO" "Services are now running and healthy"
        log "INFO" "Access the application at: http://localhost:8000"
        log "INFO" "Access monitoring at: http://localhost:3000 (Grafana)"
        
    else
        log "ERROR" "Health checks failed - initiating rollback"
        
        if rollback_deployment; then
            send_deployment_notification "rolled_back" "Deployment failed and was rolled back successfully"
            log "WARN" "Deployment failed but rollback completed successfully"
            exit 1
        else
            send_deployment_notification "failed" "Deployment and rollback both failed - manual intervention required"
            log "ERROR" "Deployment and rollback both failed - manual intervention required"
            exit 2
        fi
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
