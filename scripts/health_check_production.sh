#!/bin/bash

# 🕉️ DharmaMind Production Health Check Script
#
# Comprehensive health monitoring for production deployment
# Tests all services, endpoints, and system health
# Used by load balancers and monitoring systems
#
# Usage: ./health_check_production.sh [--verbose] [--json]
# Exit codes: 0 = healthy, 1 = unhealthy, 2 = critical
#
# May this script maintain the pulse of digital dharma ❤️‍🩹

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HEALTH_LOG="/var/log/dharmamind/health.log"
TIMEOUT=30
VERBOSE=false
JSON_OUTPUT=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --json|-j)
            JSON_OUTPUT=true
            shift
            ;;
        --timeout|-t)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--json] [--timeout SECONDS]"
            echo "  --verbose    Show detailed output"
            echo "  --json       Output results in JSON format"
            echo "  --timeout    Set timeout for checks (default: 30s)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        return  # Skip logging in JSON mode
    fi
    
    if [[ "$VERBOSE" == "true" ]] || [[ "$level" != "DEBUG" ]]; then
        case $level in
            "INFO")    echo -e "${BLUE}[INFO]${NC} $message" ;;
            "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
            "WARN")    echo -e "${YELLOW}[WARN]${NC} $message" ;;
            "ERROR")   echo -e "${RED}[ERROR]${NC} $message" ;;
            "DHARMIC") echo -e "${PURPLE}[🕉️]${NC} $message" ;;
            "DEBUG")   [[ "$VERBOSE" == "true" ]] && echo -e "[DEBUG] $message" ;;
        esac
    fi
    
    # Log to file
    mkdir -p "$(dirname "$HEALTH_LOG")"
    echo "[$timestamp] [$level] $message" >> "$HEALTH_LOG"
}

# Health check results
declare -A HEALTH_RESULTS
OVERALL_STATUS="healthy"
CRITICAL_ISSUES=0
WARNING_ISSUES=0

# Add health check result
add_result() {
    local component=$1
    local status=$2
    local message=$3
    local response_time=${4:-"0"}
    
    HEALTH_RESULTS["$component"]="$status|$message|$response_time"
    
    case $status in
        "critical"|"down")
            CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
            OVERALL_STATUS="critical"
            ;;
        "warning")
            WARNING_ISSUES=$((WARNING_ISSUES + 1))
            if [[ "$OVERALL_STATUS" == "healthy" ]]; then
                OVERALL_STATUS="warning"
            fi
            ;;
    esac
}

# Test service connectivity
test_service_connectivity() {
    local service_name=$1
    local host=$2
    local port=$3
    local protocol=${4:-"tcp"}
    
    log "DEBUG" "Testing $service_name connectivity ($host:$port)"
    
    local start_time=$(date +%s.%N)
    
    if command -v nc &> /dev/null; then
        if timeout "$TIMEOUT" nc -z "$host" "$port" &> /dev/null; then
            local end_time=$(date +%s.%N)
            local response_time=$(echo "$end_time - $start_time" | bc -l)
            add_result "$service_name" "healthy" "Service is reachable" "$response_time"
            log "SUCCESS" "$service_name connectivity check passed (${response_time}s)"
            return 0
        else
            add_result "$service_name" "critical" "Service unreachable"
            log "ERROR" "$service_name connectivity check failed"
            return 1
        fi
    else
        # Fallback to curl/wget for HTTP services
        if [[ "$port" == "80" || "$port" == "443" || "$port" == "8000" ]]; then
            local url="http://$host:$port"
            if [[ "$port" == "443" ]]; then
                url="https://$host:$port"
            fi
            
            if timeout "$TIMEOUT" curl -s --fail "$url" &> /dev/null; then
                local end_time=$(date +%s.%N)
                local response_time=$(echo "$end_time - $start_time" | bc -l)
                add_result "$service_name" "healthy" "HTTP service responding" "$response_time"
                log "SUCCESS" "$service_name HTTP check passed"
                return 0
            fi
        fi
        
        add_result "$service_name" "critical" "Service check failed - no connectivity tools"
        log "ERROR" "$service_name check failed - nc/curl not available"
        return 1
    fi
}

# Test HTTP endpoint
test_http_endpoint() {
    local endpoint_name=$1
    local url=$2
    local expected_status=${3:-"200"}
    local expected_content=${4:-""}
    
    log "DEBUG" "Testing HTTP endpoint: $url"
    
    local start_time=$(date +%s.%N)
    local response
    local status_code
    
    if response=$(timeout "$TIMEOUT" curl -s -w "%{http_code}" "$url" 2>/dev/null); then
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc -l)
        
        status_code="${response: -3}"
        local content="${response%???}"
        
        if [[ "$status_code" == "$expected_status" ]]; then
            if [[ -n "$expected_content" ]] && [[ "$content" != *"$expected_content"* ]]; then
                add_result "$endpoint_name" "warning" "Unexpected response content" "$response_time"
                log "WARN" "$endpoint_name content check failed"
                return 1
            else
                add_result "$endpoint_name" "healthy" "HTTP endpoint responding correctly" "$response_time"
                log "SUCCESS" "$endpoint_name HTTP check passed (${response_time}s)"
                return 0
            fi
        else
            add_result "$endpoint_name" "critical" "HTTP status $status_code (expected $expected_status)" "$response_time"
            log "ERROR" "$endpoint_name returned status $status_code"
            return 1
        fi
    else
        add_result "$endpoint_name" "critical" "HTTP request failed"
        log "ERROR" "$endpoint_name HTTP request failed"
        return 1
    fi
}

# Test database connectivity
test_database() {
    log "DEBUG" "Testing PostgreSQL database"
    
    local start_time=$(date +%s.%N)
    
    # First test basic connectivity
    if ! test_service_connectivity "postgres_connectivity" "localhost" "5432"; then
        return 1
    fi
    
    # Test database query if psql is available
    if command -v psql &> /dev/null; then
        local db_url="postgresql://dharmamind:${DATABASE_PASSWORD:-dharmamind}@localhost:5432/dharmamind"
        
        if timeout "$TIMEOUT" psql "$db_url" -c "SELECT 1;" &> /dev/null; then
            local end_time=$(date +%s.%N)
            local response_time=$(echo "$end_time - $start_time" | bc -l)
            add_result "postgres_query" "healthy" "Database queries working" "$response_time"
            log "SUCCESS" "PostgreSQL query test passed"
            
            # Test read replica if available
            if timeout 5 psql "postgresql://dharmamind:${DATABASE_PASSWORD:-dharmamind}@localhost:5433/dharmamind" -c "SELECT 1;" &> /dev/null; then
                add_result "postgres_replica" "healthy" "Read replica working"
                log "SUCCESS" "PostgreSQL replica test passed"
            else
                add_result "postgres_replica" "warning" "Read replica not available"
                log "WARN" "PostgreSQL replica not available"
            fi
            
            return 0
        else
            add_result "postgres_query" "critical" "Database query failed"
            log "ERROR" "PostgreSQL query test failed"
            return 1
        fi
    else
        add_result "postgres_query" "warning" "psql not available for query test"
        log "WARN" "psql not available - skipping query test"
        return 0
    fi
}

# Test Redis connectivity
test_redis() {
    log "DEBUG" "Testing Redis cache"
    
    local start_time=$(date +%s.%N)
    
    # Test basic connectivity
    if ! test_service_connectivity "redis_connectivity" "localhost" "6379"; then
        return 1
    fi
    
    # Test Redis commands if redis-cli is available
    if command -v redis-cli &> /dev/null; then
        local redis_auth=""
        if [[ -n "${REDIS_PASSWORD}" ]]; then
            redis_auth="-a ${REDIS_PASSWORD}"
        fi
        
        if timeout "$TIMEOUT" redis-cli $redis_auth ping | grep -q "PONG"; then
            local end_time=$(date +%s.%N)
            local response_time=$(echo "$end_time - $start_time" | bc -l)
            add_result "redis_ping" "healthy" "Redis responding to commands" "$response_time"
            log "SUCCESS" "Redis ping test passed"
            
            # Test basic set/get operation
            local test_key="health_check_$(date +%s)"
            if timeout 5 redis-cli $redis_auth set "$test_key" "test" EX 60 &> /dev/null && \
               timeout 5 redis-cli $redis_auth get "$test_key" | grep -q "test"; then
                add_result "redis_operations" "healthy" "Redis operations working"
                log "SUCCESS" "Redis operations test passed"
                
                # Cleanup test key
                redis-cli $redis_auth del "$test_key" &> /dev/null || true
            else
                add_result "redis_operations" "warning" "Redis operations not working properly"
                log "WARN" "Redis operations test failed"
            fi
            
            return 0
        else
            add_result "redis_ping" "critical" "Redis not responding"
            log "ERROR" "Redis ping test failed"
            return 1
        fi
    else
        add_result "redis_ping" "warning" "redis-cli not available for command test"
        log "WARN" "redis-cli not available - skipping command test"
        return 0
    fi
}

# Test application endpoints
test_application() {
    log "DEBUG" "Testing DharmaMind application"
    
    # Test main health endpoint
    if test_http_endpoint "app_health" "http://localhost:8000/health" "200" "healthy"; then
        log "SUCCESS" "Application health endpoint passed"
    else
        return 1
    fi
    
    # Test API documentation
    if test_http_endpoint "api_docs" "http://localhost:8000/docs" "200"; then
        log "SUCCESS" "API documentation accessible"
    else
        add_result "api_docs" "warning" "API documentation not accessible"
        log "WARN" "API documentation check failed"
    fi
    
    # Test specific API endpoints
    local api_endpoints=(
        "api_root:http://localhost:8000/api/"
        "auth_health:http://localhost:8000/api/auth/health"
        "chat_health:http://localhost:8000/api/chat/health"
    )
    
    for endpoint_def in "${api_endpoints[@]}"; do
        IFS=':' read -r endpoint_name endpoint_url <<< "$endpoint_def"
        
        if test_http_endpoint "$endpoint_name" "$endpoint_url" "200"; then
            log "SUCCESS" "$endpoint_name endpoint passed"
        else
            log "WARN" "$endpoint_name endpoint check failed"
        fi
    done
    
    return 0
}

# Test load balancer
test_load_balancer() {
    log "DEBUG" "Testing Nginx load balancer"
    
    # Test main HTTP endpoint
    if test_http_endpoint "nginx_http" "http://localhost" "200"; then
        log "SUCCESS" "Nginx HTTP endpoint passed"
    else
        return 1
    fi
    
    # Test HTTPS if available
    if test_http_endpoint "nginx_https" "https://localhost" "200"; then
        log "SUCCESS" "Nginx HTTPS endpoint passed"
    else
        add_result "nginx_https" "warning" "HTTPS not available or not configured"
        log "WARN" "Nginx HTTPS check failed"
    fi
    
    return 0
}

# Test monitoring services
test_monitoring() {
    log "DEBUG" "Testing monitoring services"
    
    # Test Prometheus
    if test_http_endpoint "prometheus" "http://localhost:9090/-/healthy" "200"; then
        log "SUCCESS" "Prometheus health check passed"
    else
        add_result "prometheus" "warning" "Prometheus not available"
        log "WARN" "Prometheus check failed"
    fi
    
    # Test Grafana
    if test_http_endpoint "grafana" "http://localhost:3000/api/health" "200"; then
        log "SUCCESS" "Grafana health check passed"
    else
        add_result "grafana" "warning" "Grafana not available"
        log "WARN" "Grafana check failed"
    fi
    
    # Test Loki
    if test_http_endpoint "loki" "http://localhost:3100/ready" "200"; then
        log "SUCCESS" "Loki health check passed"
    else
        add_result "loki" "warning" "Loki not available"
        log "WARN" "Loki check failed"
    fi
    
    return 0
}

# Test system resources
test_system_resources() {
    log "DEBUG" "Testing system resources"
    
    # Check disk space
    local disk_usage
    if command -v df &> /dev/null; then
        disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
        
        if [[ "$disk_usage" -lt 85 ]]; then
            add_result "disk_space" "healthy" "Disk usage: ${disk_usage}%"
            log "SUCCESS" "Disk space check passed (${disk_usage}%)"
        elif [[ "$disk_usage" -lt 95 ]]; then
            add_result "disk_space" "warning" "Disk usage: ${disk_usage}%"
            log "WARN" "Disk space warning (${disk_usage}%)"
        else
            add_result "disk_space" "critical" "Disk usage: ${disk_usage}%"
            log "ERROR" "Disk space critical (${disk_usage}%)"
        fi
    else
        add_result "disk_space" "warning" "Cannot check disk space - df not available"
        log "WARN" "df command not available"
    fi
    
    # Check memory usage
    if command -v free &> /dev/null; then
        local memory_usage
        memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        
        if [[ "$memory_usage" -lt 80 ]]; then
            add_result "memory_usage" "healthy" "Memory usage: ${memory_usage}%"
            log "SUCCESS" "Memory usage check passed (${memory_usage}%)"
        elif [[ "$memory_usage" -lt 90 ]]; then
            add_result "memory_usage" "warning" "Memory usage: ${memory_usage}%"
            log "WARN" "Memory usage warning (${memory_usage}%)"
        else
            add_result "memory_usage" "critical" "Memory usage: ${memory_usage}%"
            log "ERROR" "Memory usage critical (${memory_usage}%)"
        fi
    else
        add_result "memory_usage" "warning" "Cannot check memory usage - free not available"
        log "WARN" "free command not available"
    fi
    
    # Check CPU load
    if command -v uptime &> /dev/null; then
        local load_avg
        load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
        local cpu_cores
        cpu_cores=$(nproc 2>/dev/null || echo "1")
        local load_per_core
        load_per_core=$(echo "scale=2; $load_avg / $cpu_cores" | bc -l 2>/dev/null || echo "0")
        
        if (( $(echo "$load_per_core < 0.8" | bc -l) )); then
            add_result "cpu_load" "healthy" "Load average: $load_avg (${load_per_core} per core)"
            log "SUCCESS" "CPU load check passed ($load_avg)"
        elif (( $(echo "$load_per_core < 1.5" | bc -l) )); then
            add_result "cpu_load" "warning" "Load average: $load_avg (${load_per_core} per core)"
            log "WARN" "CPU load warning ($load_avg)"
        else
            add_result "cpu_load" "critical" "Load average: $load_avg (${load_per_core} per core)"
            log "ERROR" "CPU load critical ($load_avg)"
        fi
    else
        add_result "cpu_load" "warning" "Cannot check CPU load - uptime not available"
        log "WARN" "uptime command not available"
    fi
    
    return 0
}

# Test Docker containers
test_docker_containers() {
    log "DEBUG" "Testing Docker containers"
    
    if ! command -v docker &> /dev/null; then
        add_result "docker_check" "warning" "Docker not available for container checks"
        log "WARN" "Docker not available"
        return 0
    fi
    
    # List expected containers
    local expected_containers=(
        "dharmamind_app_primary"
        "dharmamind_postgres_primary"
        "dharmamind_redis_master"
        "dharmamind_nginx"
    )
    
    local running_containers=0
    local total_containers=${#expected_containers[@]}
    
    for container in "${expected_containers[@]}"; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            running_containers=$((running_containers + 1))
            log "DEBUG" "Container $container is running"
        else
            log "WARN" "Container $container is not running"
        fi
    done
    
    if [[ $running_containers -eq $total_containers ]]; then
        add_result "docker_containers" "healthy" "All containers running ($running_containers/$total_containers)"
        log "SUCCESS" "All Docker containers are running"
    elif [[ $running_containers -gt $((total_containers / 2)) ]]; then
        add_result "docker_containers" "warning" "Some containers not running ($running_containers/$total_containers)"
        log "WARN" "Some Docker containers are not running"
    else
        add_result "docker_containers" "critical" "Most containers not running ($running_containers/$total_containers)"
        log "ERROR" "Most Docker containers are not running"
    fi
    
    return 0
}

# Output results in JSON format
output_json() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    echo "{"
    echo "  \"timestamp\": \"$timestamp\","
    echo "  \"overall_status\": \"$OVERALL_STATUS\","
    echo "  \"critical_issues\": $CRITICAL_ISSUES,"
    echo "  \"warning_issues\": $WARNING_ISSUES,"
    echo "  \"components\": {"
    
    local first=true
    for component in "${!HEALTH_RESULTS[@]}"; do
        if [[ "$first" == "false" ]]; then
            echo ","
        fi
        first=false
        
        IFS='|' read -r status message response_time <<< "${HEALTH_RESULTS[$component]}"
        echo -n "    \"$component\": {"
        echo -n "\"status\": \"$status\", "
        echo -n "\"message\": \"$message\", "
        echo -n "\"response_time\": $response_time"
        echo -n "}"
    done
    
    echo ""
    echo "  }"
    echo "}"
}

# Output results in human-readable format
output_human() {
    echo
    log "DHARMIC" "🕉️ DharmaMind Health Check Results"
    echo "============================================"
    echo "Timestamp: $(date)"
    echo "Overall Status: $OVERALL_STATUS"
    echo "Critical Issues: $CRITICAL_ISSUES"
    echo "Warning Issues: $WARNING_ISSUES"
    echo
    echo "Component Details:"
    echo "------------------"
    
    for component in "${!HEALTH_RESULTS[@]}"; do
        IFS='|' read -r status message response_time <<< "${HEALTH_RESULTS[$component]}"
        
        local status_color=""
        case $status in
            "healthy") status_color="${GREEN}" ;;
            "warning") status_color="${YELLOW}" ;;
            "critical"|"down") status_color="${RED}" ;;
        esac
        
        printf "%-25s: ${status_color}%-8s${NC} - %s" "$component" "$status" "$message"
        if [[ "$response_time" != "0" ]] && (( $(echo "$response_time > 0" | bc -l) )); then
            printf " (%.3fs)" "$response_time"
        fi
        echo
    done
    
    echo
    echo "Health check completed at $(date)"
}

# Main health check function
main_health_check() {
    log "DHARMIC" "🕉️ Starting DharmaMind Production Health Check"
    
    # Run all health checks
    test_database
    test_redis  
    test_application
    test_load_balancer
    test_monitoring
    test_system_resources
    test_docker_containers
    
    # Output results
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        output_json
    else
        output_human
    fi
    
    # Set exit code based on overall status
    case $OVERALL_STATUS in
        "healthy")
            log "SUCCESS" "All systems healthy"
            exit 0
            ;;
        "warning")
            log "WARN" "Some warnings detected"
            exit 0  # Still considered healthy for load balancer purposes
            ;;
        "critical")
            log "ERROR" "Critical issues detected"
            exit 1
            ;;
        *)
            log "ERROR" "Unknown status: $OVERALL_STATUS"
            exit 2
            ;;
    esac
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main_health_check "$@"
fi
