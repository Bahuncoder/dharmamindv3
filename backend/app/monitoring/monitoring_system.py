# 📊 Advanced Monitoring System
# Comprehensive observability for DharmaMind production

import os
import time
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import redis
from fastapi import Request, Response
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import structlog
from elasticsearch import Elasticsearch
import aiohttp

# ================================
# 📈 METRICS CONFIGURATION
# ================================
@dataclass
class MetricsConfig:
    """Monitoring configuration"""
    
    # Prometheus settings
    PROMETHEUS_PORT: int = 9090
    METRICS_NAMESPACE: str = "dharmamind"
    
    # Performance thresholds
    RESPONSE_TIME_THRESHOLD: float = 2.0  # seconds
    ERROR_RATE_THRESHOLD: float = 0.05    # 5%
    CPU_THRESHOLD: float = 80.0           # 80%
    MEMORY_THRESHOLD: float = 85.0        # 85%
    
    # Retention periods
    METRICS_RETENTION_DAYS: int = 30
    LOGS_RETENTION_DAYS: int = 7
    TRACE_RETENTION_HOURS: int = 24
    
    # Alert settings
    ALERT_COOLDOWN_MINUTES: int = 15
    CRITICAL_ALERT_THRESHOLD: int = 3
    
    # External services
    ELASTICSEARCH_URL: str = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    GRAFANA_URL: str = os.getenv('GRAFANA_URL', 'http://localhost:3000')
    
    # Health check intervals
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    DEEP_HEALTH_CHECK_INTERVAL: int = 300  # 5 minutes

# ================================
# 📊 PROMETHEUS METRICS
# ================================
class PrometheusMetrics:
    """Centralized Prometheus metrics collector"""
    
    def __init__(self, namespace: str = "dharmamind"):
        self.namespace = namespace
        self.registry = CollectorRegistry()
        
        # HTTP Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # AI Model Metrics
        self.ai_requests_total = Counter(
            'ai_requests_total',
            'Total AI model requests',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.ai_response_time = Histogram(
            'ai_response_time_seconds',
            'AI model response time',
            ['model'],
            buckets=[1, 5, 10, 30, 60, 120],
            registry=self.registry
        )
        
        self.ai_token_usage = Counter(
            'ai_token_usage_total',
            'Total AI tokens used',
            ['model', 'type'],  # type: prompt, completion
            registry=self.registry
        )
        
        # Database Metrics
        self.db_connections_active = Gauge(
            'db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['operation'],
            registry=self.registry
        )
        
        # Application Metrics
        self.active_users = Gauge(
            'active_users_current',
            'Currently active users',
            registry=self.registry
        )
        
        self.user_sessions = Counter(
            'user_sessions_total',
            'Total user sessions',
            ['auth_method'],
            registry=self.registry
        )
        
        self.spiritual_interactions = Counter(
            'spiritual_interactions_total',
            'Spiritual interactions by type',
            ['interaction_type', 'spiritual_path'],
            registry=self.registry
        )
        
        # System Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['mount_point'],
            registry=self.registry
        )
        
        # Business Metrics
        self.wisdom_quality_score = Histogram(
            'wisdom_quality_score',
            'Quality score of spiritual wisdom responses',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.user_satisfaction = Histogram(
            'user_satisfaction_score',
            'User satisfaction ratings',
            buckets=[1, 2, 3, 4, 5],
            registry=self.registry
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_operations = Counter(
            'cache_operations_total',
            'Cache operations',
            ['operation', 'status'],  # operation: get, set, delete; status: hit, miss
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )

# ================================
# 📝 STRUCTURED LOGGING
# ================================
class StructuredLogger:
    """Advanced structured logging with multiple outputs"""
    
    def __init__(self, 
                 service_name: str = "dharmamind",
                 elasticsearch_client: Optional[Elasticsearch] = None):
        self.service_name = service_name
        self.elasticsearch = elasticsearch_client
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(service_name)
    
    async def log_request(self, 
                         request: Request, 
                         response: Response, 
                         duration: float,
                         user_id: Optional[str] = None):
        """Log HTTP request with full context"""
        
        log_data = {
            'event_type': 'http_request',
            'method': request.method,
            'url': str(request.url),
            'status_code': response.status_code,
            'duration': duration,
            'user_agent': request.headers.get('user-agent'),
            'ip_address': self._get_client_ip(request),
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add to Elasticsearch if available
        if self.elasticsearch:
            try:
                await self._index_to_elasticsearch('http_requests', log_data)
            except Exception as e:
                self.logger.error("Failed to index to Elasticsearch", error=str(e))
        
        # Structured log
        if response.status_code >= 400:
            self.logger.error("HTTP request error", **log_data)
        elif duration > MetricsConfig.RESPONSE_TIME_THRESHOLD:
            self.logger.warning("Slow HTTP request", **log_data)
        else:
            self.logger.info("HTTP request", **log_data)
    
    async def log_ai_interaction(self, 
                                model: str, 
                                prompt_tokens: int,
                                completion_tokens: int,
                                duration: float,
                                quality_score: float,
                                user_id: str,
                                spiritual_path: Optional[str] = None):
        """Log AI model interactions"""
        
        log_data = {
            'event_type': 'ai_interaction',
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'duration': duration,
            'quality_score': quality_score,
            'user_id': user_id,
            'spiritual_path': spiritual_path,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.elasticsearch:
            try:
                await self._index_to_elasticsearch('ai_interactions', log_data)
            except Exception as e:
                self.logger.error("Failed to index AI interaction", error=str(e))
        
        self.logger.info("AI interaction", **log_data)
    
    async def log_spiritual_event(self, 
                                 event_type: str,
                                 user_id: str,
                                 spiritual_path: str,
                                 details: Dict[str, Any]):
        """Log spiritual guidance events"""
        
        log_data = {
            'event_type': 'spiritual_event',
            'spiritual_event_type': event_type,
            'user_id': user_id,
            'spiritual_path': spiritual_path,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.elasticsearch:
            try:
                await self._index_to_elasticsearch('spiritual_events', log_data)
            except Exception:
                pass
        
        self.logger.info("Spiritual event", **log_data)
    
    async def log_error(self, 
                       error_type: str, 
                       error_message: str,
                       component: str,
                       user_id: Optional[str] = None,
                       traceback: Optional[str] = None):
        """Log application errors"""
        
        log_data = {
            'event_type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'component': component,
            'user_id': user_id,
            'traceback': traceback,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.elasticsearch:
            try:
                await self._index_to_elasticsearch('errors', log_data)
            except Exception:
                pass
        
        self.logger.error("Application error", **log_data)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'
    
    async def _index_to_elasticsearch(self, index: str, document: Dict[str, Any]):
        """Index document to Elasticsearch"""
        if not self.elasticsearch:
            return
        
        index_name = f"{self.service_name}-{index}-{datetime.utcnow().strftime('%Y.%m.%d')}"
        
        await self.elasticsearch.index(
            index=index_name,
            body=document
        )

# ================================
# 🔍 PERFORMANCE MONITORING
# ================================
class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.metrics = PrometheusMetrics()
        self.logger = structlog.get_logger(__name__)
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.active_requests = 0
    
    async def start_request_monitoring(self) -> str:
        """Start monitoring a new request"""
        request_id = f"req_{int(time.time() * 1000000)}"
        self.active_requests += 1
        self.metrics.active_users.set(self.active_requests)
        return request_id
    
    async def end_request_monitoring(self, 
                                   request_id: str, 
                                   method: str,
                                   endpoint: str,
                                   status_code: int,
                                   duration: float):
        """End request monitoring and record metrics"""
        
        self.active_requests = max(0, self.active_requests - 1)
        self.metrics.active_users.set(self.active_requests)
        
        # Record metrics
        self.metrics.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
        
        self.metrics.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        # Track response times
        self.response_times.append(duration)
        
        # Track errors
        if status_code >= 400:
            self.error_counts[status_code] += 1
        
        # Store in Redis for real-time analysis
        await self._store_request_data(request_id, {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration': duration,
            'timestamp': time.time()
        })
    
    async def monitor_ai_request(self, 
                               model: str, 
                               duration: float,
                               token_usage: Dict[str, int],
                               success: bool):
        """Monitor AI model requests"""
        
        status = 'success' if success else 'error'
        
        self.metrics.ai_requests_total.labels(
            model=model, 
            status=status
        ).inc()
        
        self.metrics.ai_response_time.labels(model=model).observe(duration)
        
        # Track token usage
        if 'prompt_tokens' in token_usage:
            self.metrics.ai_token_usage.labels(
                model=model, 
                type='prompt'
            ).inc(token_usage['prompt_tokens'])
        
        if 'completion_tokens' in token_usage:
            self.metrics.ai_token_usage.labels(
                model=model, 
                type='completion'
            ).inc(token_usage['completion_tokens'])
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        
        now = time.time()
        recent_responses = [rt for rt in self.response_times if now - rt < 300]  # Last 5 minutes
        
        return {
            'active_requests': self.active_requests,
            'avg_response_time': sum(recent_responses) / len(recent_responses) if recent_responses else 0,
            'p95_response_time': sorted(recent_responses)[int(len(recent_responses) * 0.95)] if recent_responses else 0,
            'error_rate': sum(self.error_counts.values()) / max(1, len(self.response_times)),
            'total_requests': len(self.response_times),
            'recent_errors': dict(self.error_counts)
        }
    
    async def _store_request_data(self, request_id: str, data: Dict[str, Any]):
        """Store request data in Redis for analysis"""
        key = f"request_data:{request_id}"
        await self.redis_client.hset(key, mapping=data)
        await self.redis_client.expire(key, 3600)  # 1 hour retention

# ================================
# 🏥 HEALTH MONITORING
# ================================
class HealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, redis_client: redis.Redis, metrics: PrometheusMetrics):
        self.redis_client = redis_client
        self.metrics = metrics
        self.logger = structlog.get_logger(__name__)
        self.health_checks = {}
        self.last_check_time = {}
    
    async def register_health_check(self, 
                                  name: str, 
                                  check_function,
                                  interval: int = 60):
        """Register a health check function"""
        self.health_checks[name] = {
            'function': check_function,
            'interval': interval,
            'last_run': 0,
            'status': 'unknown',
            'details': {}
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        now = time.time()
        overall_status = 'healthy'
        results = {}
        
        for name, check_config in self.health_checks.items():
            # Check if it's time to run this check
            if now - check_config['last_run'] < check_config['interval']:
                results[name] = {
                    'status': check_config['status'],
                    'details': check_config['details'],
                    'last_check': check_config['last_run']
                }
                continue
            
            try:
                # Run the health check
                check_result = await check_config['function']()
                
                check_config['status'] = check_result.get('status', 'unknown')
                check_config['details'] = check_result.get('details', {})
                check_config['last_run'] = now
                
                results[name] = {
                    'status': check_config['status'],
                    'details': check_config['details'],
                    'last_check': now
                }
                
                # Update overall status
                if check_config['status'] in ['critical', 'down']:
                    overall_status = 'critical'
                elif check_config['status'] == 'warning' and overall_status == 'healthy':
                    overall_status = 'warning'
                    
            except Exception as e:
                self.logger.error(f"Health check {name} failed", error=str(e))
                check_config['status'] = 'error'
                check_config['details'] = {'error': str(e)}
                check_config['last_run'] = now
                
                results[name] = {
                    'status': 'error',
                    'details': {'error': str(e)},
                    'last_check': now
                }
                
                overall_status = 'critical'
        
        return {
            'overall_status': overall_status,
            'checks': results,
            'timestamp': now
        }
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            # This would connect to your actual database
            # For now, simulating the check
            
            start_time = time.time()
            # await database.execute("SELECT 1")
            query_time = time.time() - start_time
            
            if query_time > 1.0:
                return {
                    'status': 'warning',
                    'details': {
                        'query_time': query_time,
                        'message': 'Database responding slowly'
                    }
                }
            
            return {
                'status': 'healthy',
                'details': {
                    'query_time': query_time,
                    'connection_pool': 'available'
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'details': {
                    'error': str(e),
                    'message': 'Database connection failed'
                }
            }
    
    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = time.time() - start_time
            
            # Get Redis info
            info = await self.redis_client.info()
            memory_usage = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            memory_percent = (memory_usage / max_memory * 100) if max_memory > 0 else 0
            
            status = 'healthy'
            if memory_percent > 90:
                status = 'critical'
            elif memory_percent > 80:
                status = 'warning'
            
            return {
                'status': status,
                'details': {
                    'ping_time': ping_time,
                    'memory_usage_percent': memory_percent,
                    'connected_clients': info.get('connected_clients', 0)
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'details': {
                    'error': str(e),
                    'message': 'Redis connection failed'
                }
            }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system CPU, memory, and disk usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update Prometheus metrics
            self.metrics.system_cpu_usage.set(cpu_percent)
            self.metrics.system_memory_usage.set(memory.percent)
            self.metrics.system_disk_usage.labels(mount_point='/').set(disk.percent)
            
            status = 'healthy'
            warnings = []
            
            if cpu_percent > MetricsConfig.CPU_THRESHOLD:
                status = 'warning'
                warnings.append(f'High CPU usage: {cpu_percent}%')
            
            if memory.percent > MetricsConfig.MEMORY_THRESHOLD:
                status = 'critical' if memory.percent > 95 else 'warning'
                warnings.append(f'High memory usage: {memory.percent}%')
            
            if disk.percent > 90:
                status = 'critical'
                warnings.append(f'High disk usage: {disk.percent}%')
            
            return {
                'status': status,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'warnings': warnings
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'details': {
                    'error': str(e),
                    'message': 'Failed to get system resources'
                }
            }
    
    async def check_ai_services(self) -> Dict[str, Any]:
        """Check AI service availability"""
        services = ['openai', 'anthropic', 'google']
        results = {}
        overall_status = 'healthy'
        
        for service in services:
            try:
                # Simulate AI service health check
                # In reality, you'd make a simple API call to each service
                start_time = time.time()
                
                # Placeholder for actual health check
                # response = await ai_service.health_check()
                
                response_time = time.time() - start_time
                
                results[service] = {
                    'status': 'healthy',
                    'response_time': response_time
                }
                
            except Exception as e:
                results[service] = {
                    'status': 'down',
                    'error': str(e)
                }
                overall_status = 'warning'  # Can still function with some services down
        
        return {
            'status': overall_status,
            'details': results
        }

# ================================
# 📊 ANALYTICS ENGINE
# ================================
class AnalyticsEngine:
    """Advanced analytics and insights"""
    
    def __init__(self, redis_client: redis.Redis, elasticsearch: Optional[Elasticsearch] = None):
        self.redis_client = redis_client
        self.elasticsearch = elasticsearch
        self.logger = structlog.get_logger(__name__)
    
    async def track_user_journey(self, 
                                user_id: str, 
                                event: str, 
                                properties: Dict[str, Any]):
        """Track user journey events"""
        
        journey_data = {
            'user_id': user_id,
            'event': event,
            'properties': properties,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in Redis for real-time analytics
        await self.redis_client.lpush(f"user_journey:{user_id}", json.dumps(journey_data))
        await self.redis_client.expire(f"user_journey:{user_id}", 86400 * 30)  # 30 days
        
        # Index to Elasticsearch for complex analytics
        if self.elasticsearch:
            try:
                await self._index_user_event(journey_data)
            except Exception as e:
                self.logger.error("Failed to index user event", error=str(e))
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights for a specific user"""
        
        # Get user journey from Redis
        journey_data = await self.redis_client.lrange(f"user_journey:{user_id}", 0, -1)
        
        events = []
        for data in journey_data:
            events.append(json.loads(data.decode()))
        
        # Calculate insights
        total_sessions = len([e for e in events if e['event'] == 'session_start'])
        spiritual_interactions = len([e for e in events if e['event'] == 'spiritual_question'])
        avg_session_length = self._calculate_avg_session_length(events)
        most_used_path = self._get_most_used_spiritual_path(events)
        
        return {
            'user_id': user_id,
            'total_sessions': total_sessions,
            'spiritual_interactions': spiritual_interactions,
            'avg_session_length_minutes': avg_session_length,
            'most_used_spiritual_path': most_used_path,
            'engagement_score': self._calculate_engagement_score(events),
            'last_activity': events[0]['timestamp'] if events else None
        }
    
    async def get_platform_analytics(self) -> Dict[str, Any]:
        """Get overall platform analytics"""
        
        now = datetime.utcnow()
        today = now.strftime('%Y-%m-%d')
        
        # Get daily metrics from Redis
        daily_users = await self.redis_client.get(f"daily_active_users:{today}") or 0
        daily_sessions = await self.redis_client.get(f"daily_sessions:{today}") or 0
        daily_questions = await self.redis_client.get(f"daily_questions:{today}") or 0
        
        # Get spiritual path distribution
        path_distribution = await self._get_spiritual_path_distribution()
        
        # Get AI model usage stats
        model_usage = await self._get_ai_model_usage()
        
        return {
            'daily_active_users': int(daily_users),
            'daily_sessions': int(daily_sessions),
            'daily_spiritual_questions': int(daily_questions),
            'spiritual_path_distribution': path_distribution,
            'ai_model_usage': model_usage,
            'platform_health_score': await self._calculate_platform_health_score(),
            'user_satisfaction_average': await self._get_avg_user_satisfaction()
        }
    
    def _calculate_avg_session_length(self, events: List[Dict]) -> float:
        """Calculate average session length in minutes"""
        session_starts = [e for e in events if e['event'] == 'session_start']
        session_ends = [e for e in events if e['event'] == 'session_end']
        
        if not session_starts or not session_ends:
            return 0.0
        
        total_duration = 0
        sessions = 0
        
        for start_event in session_starts:
            start_time = datetime.fromisoformat(start_event['timestamp'])
            
            # Find corresponding end event
            for end_event in session_ends:
                end_time = datetime.fromisoformat(end_event['timestamp'])
                if end_time > start_time:
                    duration = (end_time - start_time).total_seconds() / 60
                    total_duration += duration
                    sessions += 1
                    break
        
        return total_duration / sessions if sessions > 0 else 0.0
    
    def _get_most_used_spiritual_path(self, events: List[Dict]) -> str:
        """Get the most frequently used spiritual path"""
        path_counts = defaultdict(int)
        
        for event in events:
            if 'spiritual_path' in event.get('properties', {}):
                path = event['properties']['spiritual_path']
                path_counts[path] += 1
        
        if not path_counts:
            return 'unknown'
        
        return max(path_counts, key=path_counts.get)
    
    def _calculate_engagement_score(self, events: List[Dict]) -> float:
        """Calculate user engagement score (0-100)"""
        if not events:
            return 0.0
        
        score = 0
        
        # Base score for activity
        score += min(30, len(events))
        
        # Bonus for diverse interactions
        event_types = set(e['event'] for e in events)
        score += len(event_types) * 5
        
        # Bonus for recent activity
        recent_events = [e for e in events 
                        if (datetime.utcnow() - datetime.fromisoformat(e['timestamp'])).days <= 7]
        score += len(recent_events) * 2
        
        # Bonus for spiritual engagement
        spiritual_events = [e for e in events if 'spiritual' in e['event']]
        score += len(spiritual_events) * 3
        
        return min(100, score)
    
    async def _get_spiritual_path_distribution(self) -> Dict[str, int]:
        """Get distribution of spiritual paths"""
        paths = ['karma_yoga', 'bhakti_yoga', 'raja_yoga', 'jnana_yoga', 'general']
        distribution = {}
        
        for path in paths:
            count = await self.redis_client.get(f"spiritual_path_users:{path}") or 0
            distribution[path] = int(count)
        
        return distribution
    
    async def _get_ai_model_usage(self) -> Dict[str, Dict[str, int]]:
        """Get AI model usage statistics"""
        models = ['gpt-4', 'claude-3', 'gemini-pro']
        usage = {}
        
        for model in models:
            requests = await self.redis_client.get(f"ai_requests:{model}:today") or 0
            tokens = await self.redis_client.get(f"ai_tokens:{model}:today") or 0
            
            usage[model] = {
                'requests': int(requests),
                'tokens': int(tokens)
            }
        
        return usage
    
    async def _calculate_platform_health_score(self) -> float:
        """Calculate overall platform health score"""
        # This would integrate with your health monitoring
        # For now, returning a placeholder
        return 95.5
    
    async def _get_avg_user_satisfaction(self) -> float:
        """Get average user satisfaction score"""
        # This would come from user feedback
        # For now, returning a placeholder
        return 4.3
    
    async def _index_user_event(self, event_data: Dict[str, Any]):
        """Index user event to Elasticsearch"""
        if not self.elasticsearch:
            return
        
        index_name = f"dharmamind-user-events-{datetime.utcnow().strftime('%Y.%m.%d')}"
        
        await self.elasticsearch.index(
            index=index_name,
            body=event_data
        )

# ================================
# 🚨 ALERTING SYSTEM
# ================================
class AlertingSystem:
    """Intelligent alerting and notification system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = structlog.get_logger(__name__)
        self.alert_channels = []
        self.alert_rules = {}
        self.cooldown_tracker = {}
    
    def add_alert_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add an alert notification channel"""
        self.alert_channels.append({
            'type': channel_type,
            'config': config
        })
    
    def add_alert_rule(self, 
                      rule_name: str, 
                      condition_func, 
                      severity: str,
                      cooldown_minutes: int = 15):
        """Add an alert rule"""
        self.alert_rules[rule_name] = {
            'condition': condition_func,
            'severity': severity,
            'cooldown': cooldown_minutes * 60
        }
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        
        current_time = time.time()
        
        for rule_name, rule_config in self.alert_rules.items():
            try:
                # Check if we're in cooldown period
                last_alert = self.cooldown_tracker.get(rule_name, 0)
                if current_time - last_alert < rule_config['cooldown']:
                    continue
                
                # Evaluate condition
                if await rule_config['condition'](metrics):
                    await self._trigger_alert(rule_name, rule_config, metrics)
                    self.cooldown_tracker[rule_name] = current_time
                    
            except Exception as e:
                self.logger.error(f"Alert rule {rule_name} failed", error=str(e))
    
    async def _trigger_alert(self, 
                           rule_name: str, 
                           rule_config: Dict[str, Any],
                           metrics: Dict[str, Any]):
        """Trigger an alert notification"""
        
        alert_data = {
            'rule_name': rule_name,
            'severity': rule_config['severity'],
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'message': f"Alert triggered: {rule_name}"
        }
        
        # Store alert
        await self.redis_client.lpush('alerts', json.dumps(alert_data))
        await self.redis_client.ltrim('alerts', 0, 999)  # Keep last 1000 alerts
        
        # Log alert
        self.logger.critical("ALERT TRIGGERED", **alert_data)
        
        # Send notifications
        for channel in self.alert_channels:
            try:
                await self._send_notification(channel, alert_data)
            except Exception as e:
                self.logger.error(f"Failed to send alert to {channel['type']}", error=str(e))
    
    async def _send_notification(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send alert notification to a channel"""
        
        if channel['type'] == 'slack':
            await self._send_slack_notification(channel['config'], alert_data)
        elif channel['type'] == 'email':
            await self._send_email_notification(channel['config'], alert_data)
        # Add more notification types as needed
    
    async def _send_slack_notification(self, config: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send Slack notification"""
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return
        
        severity_emoji = {
            'critical': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        message = {
            'text': f"{severity_emoji.get(alert_data['severity'], '🔔')} DharmaMind Alert",
            'attachments': [{
                'color': 'danger' if alert_data['severity'] == 'critical' else 'warning',
                'fields': [
                    {
                        'title': 'Rule',
                        'value': alert_data['rule_name'],
                        'short': True
                    },
                    {
                        'title': 'Severity',
                        'value': alert_data['severity'].upper(),
                        'short': True
                    },
                    {
                        'title': 'Time',
                        'value': alert_data['timestamp'],
                        'short': True
                    }
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status != 200:
                    raise Exception(f"Slack notification failed with status {response.status}")
    
    async def _send_email_notification(self, config: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send email notification"""
        # Implement email sending logic
        pass

# ================================
# 🔧 MONITORING SYSTEM INITIALIZATION
# ================================
async def initialize_monitoring_system(redis_client: redis.Redis) -> Dict[str, Any]:
    """Initialize the complete monitoring system"""
    
    # Initialize components
    metrics = PrometheusMetrics()
    logger = StructuredLogger("dharmamind")
    performance_monitor = PerformanceMonitor(redis_client)
    health_monitor = HealthMonitor(redis_client, metrics)
    analytics_engine = AnalyticsEngine(redis_client)
    alerting_system = AlertingSystem(redis_client)
    
    # Register health checks
    await health_monitor.register_health_check("database", health_monitor.check_database_health)
    await health_monitor.register_health_check("redis", health_monitor.check_redis_health)
    await health_monitor.register_health_check("system", health_monitor.check_system_resources)
    await health_monitor.register_health_check("ai_services", health_monitor.check_ai_services)
    
    # Setup alert rules
    alerting_system.add_alert_rule(
        "high_error_rate",
        lambda m: m.get('error_rate', 0) > MetricsConfig.ERROR_RATE_THRESHOLD,
        "critical"
    )
    
    alerting_system.add_alert_rule(
        "slow_response_time",
        lambda m: m.get('avg_response_time', 0) > MetricsConfig.RESPONSE_TIME_THRESHOLD,
        "warning"
    )
    
    return {
        'metrics': metrics,
        'logger': logger,
        'performance_monitor': performance_monitor,
        'health_monitor': health_monitor,
        'analytics_engine': analytics_engine,
        'alerting_system': alerting_system
    }

# ================================
# 📊 MONITORING DASHBOARD DATA
# ================================
async def get_monitoring_dashboard_data(monitoring_system: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data"""
    
    performance_monitor = monitoring_system['performance_monitor']
    health_monitor = monitoring_system['health_monitor']
    analytics_engine = monitoring_system['analytics_engine']
    
    return {
        'performance': await performance_monitor.get_performance_summary(),
        'health': await health_monitor.run_health_checks(),
        'analytics': await analytics_engine.get_platform_analytics(),
        'timestamp': datetime.utcnow().isoformat()
    }
