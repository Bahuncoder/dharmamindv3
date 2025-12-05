"""
Prometheus Metrics Exporter for DharmaMind
==========================================

Exports application metrics in Prometheus format for monitoring:
- HTTP request metrics (count, duration, status codes)
- Authentication metrics (login attempts, failures)
- Database metrics (query count, duration, errors)
- Cache metrics (hits, misses, evictions)
- Custom business metrics (dharmic scores, user satisfaction)

Endpoints:
- GET /metrics - Prometheus metrics endpoint

Author: DharmaMind Infrastructure Team
Date: October 27, 2025
"""

import time
import logging
from typing import Optional
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)

logger = logging.getLogger(__name__)


# ==================
# METRIC DEFINITIONS
# ==================

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests currently being processed',
    ['method', 'endpoint']
)

# Authentication Metrics
auth_attempts_total = Counter(
    'auth_attempts_total',
    'Total authentication attempts',
    ['result', 'type']  # result: success/failure, type: login/register
)

auth_lockouts_total = Counter(
    'auth_lockouts_total',
    'Total account lockouts due to failed attempts'
)

active_sessions = Gauge(
    'active_sessions',
    'Number of active user sessions'
)

# Database Metrics
db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['operation', 'table']
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

db_errors_total = Counter(
    'db_errors_total',
    'Total database errors',
    ['error_type']
)

db_connections = Gauge(
    'db_connections',
    'Number of active database connections'
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits'
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses'
)

cache_evictions_total = Counter(
    'cache_evictions_total',
    'Total cache evictions'
)

cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes'
)

# Rate Limiting Metrics
rate_limit_hits_total = Counter(
    'rate_limit_hits_total',
    'Total rate limit hits (blocked requests)',
    ['endpoint']
)

rate_limit_blocked_ips = Gauge(
    'rate_limit_blocked_ips',
    'Number of currently blocked IP addresses'
)

# LLM Metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM inference requests',
    ['model', 'result']
)

llm_inference_duration_seconds = Histogram(
    'llm_inference_duration_seconds',
    'LLM inference duration in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

llm_token_count = Histogram(
    'llm_token_count',
    'Number of tokens generated',
    ['model'],
    buckets=(10, 50, 100, 200, 500, 1000, 2000)
)

dharmic_score = Histogram(
    'dharmic_score',
    'Dharmic alignment score of generated responses',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# Application Info
app_info = Info('dharmamind_app', 'Application information')
app_info.info({
    'version': '1.0.0',
    'environment': 'production'
})


# ==================
# MIDDLEWARE
# ==================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP metrics
    
    Automatically tracks:
    - Request count by method, endpoint, status
    - Request duration
    - Requests in progress
    
    Example:
        app.add_middleware(PrometheusMiddleware)
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics"""
        
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Extract endpoint (remove query params, route params)
        endpoint = self._clean_endpoint(request.url.path)
        method = request.method
        
        # Track requests in progress
        http_requests_in_progress.labels(
            method=method,
            endpoint=endpoint
        ).inc()
        
        # Start timer
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            status = response.status_code
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=500
            ).inc()
            
            raise
            
        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(
                method=method,
                endpoint=endpoint
            ).dec()
    
    def _clean_endpoint(self, path: str) -> str:
        """Clean endpoint path for metric labels"""
        # Remove trailing slash
        if path.endswith('/') and path != '/':
            path = path[:-1]
        
        # Replace path parameters with placeholder
        # e.g., /users/123 -> /users/{id}
        parts = path.split('/')
        cleaned_parts = []
        
        for part in parts:
            # If part looks like an ID (numeric or UUID), replace with placeholder
            if part.isdigit() or self._is_uuid(part):
                cleaned_parts.append('{id}')
            else:
                cleaned_parts.append(part)
        
        return '/'.join(cleaned_parts)
    
    def _is_uuid(self, value: str) -> bool:
        """Check if value looks like a UUID"""
        try:
            import uuid
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False


# ==================
# METRIC HELPERS
# ==================

class MetricsCollector:
    """
    Helper class for collecting application metrics
    
    Provides convenient methods for recording metrics throughout the application.
    
    Example:
        metrics = MetricsCollector()
        metrics.record_auth_attempt(success=True, auth_type="login")
        metrics.record_db_query("SELECT", "users", duration=0.05)
    """
    
    @staticmethod
    def record_auth_attempt(success: bool, auth_type: str = "login"):
        """Record authentication attempt"""
        result = "success" if success else "failure"
        auth_attempts_total.labels(result=result, type=auth_type).inc()
    
    @staticmethod
    def record_auth_lockout():
        """Record account lockout"""
        auth_lockouts_total.inc()
    
    @staticmethod
    def set_active_sessions(count: int):
        """Set number of active sessions"""
        active_sessions.set(count)
    
    @staticmethod
    def record_db_query(operation: str, table: str, duration: float):
        """Record database query"""
        db_queries_total.labels(operation=operation, table=table).inc()
        db_query_duration_seconds.labels(operation=operation).observe(duration)
    
    @staticmethod
    def record_db_error(error_type: str):
        """Record database error"""
        db_errors_total.labels(error_type=error_type).inc()
    
    @staticmethod
    def set_db_connections(count: int):
        """Set number of database connections"""
        db_connections.set(count)
    
    @staticmethod
    def record_cache_hit():
        """Record cache hit"""
        cache_hits_total.inc()
    
    @staticmethod
    def record_cache_miss():
        """Record cache miss"""
        cache_misses_total.inc()
    
    @staticmethod
    def record_cache_eviction():
        """Record cache eviction"""
        cache_evictions_total.inc()
    
    @staticmethod
    def set_cache_size(size_bytes: int):
        """Set cache size"""
        cache_size_bytes.set(size_bytes)
    
    @staticmethod
    def record_rate_limit_hit(endpoint: str):
        """Record rate limit hit"""
        rate_limit_hits_total.labels(endpoint=endpoint).inc()
    
    @staticmethod
    def set_blocked_ips(count: int):
        """Set number of blocked IPs"""
        rate_limit_blocked_ips.set(count)
    
    @staticmethod
    def record_llm_request(model: str, success: bool, duration: float,
                          token_count: int, dharmic_score_value: float):
        """Record LLM inference request"""
        result = "success" if success else "failure"
        llm_requests_total.labels(model=model, result=result).inc()
        llm_inference_duration_seconds.labels(model=model).observe(duration)
        llm_token_count.labels(model=model).observe(token_count)
        dharmic_score.observe(dharmic_score_value)


# Decorator for timing functions
def track_duration(metric: Histogram, **labels):
    """
    Decorator to track function duration
    
    Example:
        @track_duration(db_query_duration_seconds, operation="SELECT")
        async def fetch_users():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ==================
# METRICS ENDPOINT
# ==================

async def metrics_endpoint():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus text format.
    Configure Prometheus to scrape this endpoint.
    
    Example prometheus.yml:
        scrape_configs:
          - job_name: 'dharmamind'
            scrape_interval: 15s
            static_configs:
              - targets: ['localhost:8000']
            metrics_path: '/metrics'
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


# ==================
# INITIALIZATION
# ==================

def setup_metrics(app):
    """
    Set up metrics for FastAPI application
    
    Args:
        app: FastAPI application instance
    
    Example:
        from fastapi import FastAPI
        from services.monitoring.metrics_exporter import setup_metrics
        
        app = FastAPI()
        setup_metrics(app)
    """
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Add metrics endpoint
    app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])
    
    logger.info("✅ Prometheus metrics configured")
    logger.info("   Metrics endpoint: /metrics")


# Example usage
if __name__ == "__main__":
    print("📊 Prometheus Metrics Exporter")
    print("\nAvailable metrics:")
    print("  - http_requests_total")
    print("  - http_request_duration_seconds")
    print("  - http_requests_in_progress")
    print("  - auth_attempts_total")
    print("  - db_queries_total")
    print("  - cache_hits_total")
    print("  - llm_requests_total")
    print("  - dharmic_score")
    print("\nUsage:")
    print("  from services.monitoring.metrics_exporter import setup_metrics")
    print("  setup_metrics(app)")
