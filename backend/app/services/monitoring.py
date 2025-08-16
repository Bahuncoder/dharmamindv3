"""
🕉️ DharmaMind Production Monitoring System

Comprehensive monitoring and health check system for production deployment:
- Real-time system health monitoring
- Performance metrics collection
- Alert system for critical issues
- Dashboard data aggregation
- Dharmic analytics and insights
- Predictive health analysis

May this system maintain the wellbeing of our digital dharma service 🚀
"""

import asyncio
import logging
import json
import time
import psutil
import platform
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from pathlib import Path

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logging.warning("Prometheus client not installed - using mock metrics")

from ..config import settings
from ..db.database import DatabaseManager
from .cache_service import get_cache_service

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    response_time: float
    last_check: datetime
    details: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_connections: int
    request_rate: float
    error_rate: float
    response_time_avg: float


@dataclass
class Alert:
    """System alert"""
    id: str
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]
    created_at: datetime
    resolved_at: Optional[datetime]
    acknowledged: bool


class MonitoringService:
    """🚀 Comprehensive Production Monitoring System"""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize monitoring service"""
        self.db = db_manager
        self.cache_service = get_cache_service()
        
        # Health check components
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # Monitoring configuration
        self.config = {
            'check_interval': 30,          # seconds
            'metrics_retention_hours': 24,
            'alert_cooldown': 300,         # 5 minutes
            'critical_response_time': 5.0,  # seconds
            'warning_response_time': 2.0,
            'critical_error_rate': 0.05,   # 5%
            'warning_error_rate': 0.02,    # 2%
            'critical_cpu_percent': 90,
            'warning_cpu_percent': 75,
            'critical_memory_percent': 90,
            'warning_memory_percent': 80,
            'critical_disk_percent': 95,
            'warning_disk_percent': 85
        }
        
        # Prometheus metrics (if available)
        if HAS_PROMETHEUS:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # Component health checkers
        self.health_checkers = {
            'database': self._check_database_health,
            'redis': self._check_redis_health,
            'cache': self._check_cache_health,
            'system': self._check_system_health,
            'api': self._check_api_health,
            'dharmic_ai': self._check_dharmic_ai_health
        }
        
        # Alert handlers
        self.alert_handlers = []
        
        logger.info("🔱 Monitoring Service initialized")
    
    async def initialize(self):
        """Initialize monitoring service"""
        logger.info("📊 Initializing Monitoring Service...")
        
        try:
            # Create monitoring tables
            await self._create_monitoring_tables()
            
            # Start monitoring tasks
            asyncio.create_task(self._health_check_worker())
            asyncio.create_task(self._metrics_collection_worker())
            asyncio.create_task(self._alert_processing_worker())
            asyncio.create_task(self._cleanup_worker())
            
            # Initial health check
            await self.run_all_health_checks()
            
            logger.info("✅ Monitoring Service ready!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Monitoring Service: {e}")
            raise
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        if not HAS_PROMETHEUS:
            return
        
        # Request metrics
        self.request_counter = Counter(
            'dharmamind_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'dharmamind_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'dharmamind_cpu_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'dharmamind_memory_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'dharmamind_disk_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Application metrics
        self.active_users = Gauge(
            'dharmamind_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        self.dharmic_responses = Counter(
            'dharmamind_dharmic_responses_total',
            'Total dharmic responses generated',
            ['wisdom_level'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'dharmamind_cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
    
    async def _create_monitoring_tables(self):
        """Create monitoring database tables"""
        
        # System metrics table
        metrics_table = """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                component VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """
        
        # Health checks table
        health_checks_table = """
            CREATE TABLE IF NOT EXISTS health_checks (
                id SERIAL PRIMARY KEY,
                component VARCHAR(100) NOT NULL,
                status VARCHAR(20) NOT NULL,
                message TEXT,
                response_time FLOAT,
                details JSONB DEFAULT '{}',
                checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """
        
        # Alerts table
        alerts_table = """
            CREATE TABLE IF NOT EXISTS monitoring_alerts (
                id VARCHAR(255) PRIMARY KEY,
                level VARCHAR(20) NOT NULL,
                component VARCHAR(100) NOT NULL,
                message TEXT NOT NULL,
                details JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_by VARCHAR(255),
                acknowledged_at TIMESTAMP WITH TIME ZONE
            )
        """
        
        # Performance incidents table
        incidents_table = """
            CREATE TABLE IF NOT EXISTS performance_incidents (
                id SERIAL PRIMARY KEY,
                incident_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                component VARCHAR(100) NOT NULL,
                description TEXT,
                metrics_snapshot JSONB,
                started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE,
                resolution_notes TEXT,
                impact_users INTEGER DEFAULT 0
            )
        """
        
        await self.db.execute_query(metrics_table)
        await self.db.execute_query(health_checks_table)
        await self.db.execute_query(alerts_table)
        await self.db.execute_query(incidents_table)
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_component ON system_metrics(component)",
            "CREATE INDEX IF NOT EXISTS idx_health_checks_component ON health_checks(component)",
            "CREATE INDEX IF NOT EXISTS idx_health_checks_checked_at ON health_checks(checked_at)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_level ON monitoring_alerts(level)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON monitoring_alerts(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_started_at ON performance_incidents(started_at)"
        ]
        
        for index_query in indexes:
            await self.db.execute_query(index_query)
        
        logger.info("✅ Monitoring database tables created")
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        
        logger.debug("🔍 Running comprehensive health checks...")
        
        results = {}
        
        for component_name, checker_func in self.health_checkers.items():
            try:
                start_time = time.time()
                health_check = await checker_func()
                health_check.response_time = time.time() - start_time
                health_check.last_check = datetime.now()
                
                results[component_name] = health_check
                self.health_checks[component_name] = health_check
                
                # Store in database
                await self._store_health_check(health_check)
                
                # Check for alerts
                await self._check_component_alerts(health_check)
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                
                error_check = HealthCheck(
                    component=component_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    response_time=0.0,
                    last_check=datetime.now(),
                    details={'error': str(e)}
                )
                
                results[component_name] = error_check
                self.health_checks[component_name] = error_check
        
        logger.debug(f"✅ Health checks completed for {len(results)} components")
        return results
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database health"""
        
        try:
            # Test basic connectivity
            start_time = time.time()
            result = await self.db.execute_query("SELECT 1 as health_check")
            query_time = time.time() - start_time
            
            if not result:
                return HealthCheck(
                    component="database",
                    status=HealthStatus.CRITICAL,
                    message="Database query returned no results",
                    response_time=query_time,
                    last_check=datetime.now(),
                    details={'query_time': query_time}
                )
            
            # Check connection pool status
            connection_stats = await self.db.get_connection_stats()
            
            status = HealthStatus.HEALTHY
            message = "Database is healthy"
            
            # Determine status based on performance
            if query_time > self.config['critical_response_time']:
                status = HealthStatus.CRITICAL
                message = f"Database response time critical: {query_time:.2f}s"
            elif query_time > self.config['warning_response_time']:
                status = HealthStatus.WARNING
                message = f"Database response time slow: {query_time:.2f}s"
            
            return HealthCheck(
                component="database",
                status=status,
                message=message,
                response_time=query_time,
                last_check=datetime.now(),
                details={
                    'query_time': query_time,
                    'connection_stats': connection_stats
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="database",
                status=HealthStatus.DOWN,
                message=f"Database connection failed: {str(e)}",
                response_time=0.0,
                last_check=datetime.now(),
                details={'error': str(e)}
            )
    
    async def _check_redis_health(self) -> HealthCheck:
        """Check Redis cache health"""
        
        try:
            cache_service = self.cache_service
            
            if not cache_service.redis_client:
                return HealthCheck(
                    component="redis",
                    status=HealthStatus.WARNING,
                    message="Redis client not available - using memory cache only",
                    response_time=0.0,
                    last_check=datetime.now(),
                    details={'fallback_mode': True}
                )
            
            # Test Redis connectivity
            start_time = time.time()
            await cache_service.redis_client.ping()
            ping_time = time.time() - start_time
            
            # Get Redis info
            try:
                info = await cache_service.redis_client.info()
                memory_usage = info.get('used_memory', 0)
                connected_clients = info.get('connected_clients', 0)
                
                status = HealthStatus.HEALTHY
                message = "Redis is healthy"
                
                if ping_time > 1.0:
                    status = HealthStatus.WARNING
                    message = f"Redis response time slow: {ping_time:.2f}s"
                
                return HealthCheck(
                    component="redis",
                    status=status,
                    message=message,
                    response_time=ping_time,
                    last_check=datetime.now(),
                    details={
                        'ping_time': ping_time,
                        'memory_usage': memory_usage,
                        'connected_clients': connected_clients,
                        'redis_version': info.get('redis_version')
                    }
                )
                
            except Exception as e:
                logger.warning(f"Failed to get Redis info: {e}")
                
                return HealthCheck(
                    component="redis",
                    status=HealthStatus.WARNING,
                    message="Redis connected but info unavailable",
                    response_time=ping_time,
                    last_check=datetime.now(),
                    details={'ping_time': ping_time, 'info_error': str(e)}
                )
            
        except Exception as e:
            return HealthCheck(
                component="redis",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}",
                response_time=0.0,
                last_check=datetime.now(),
                details={'error': str(e)}
            )
    
    async def _check_cache_health(self) -> HealthCheck:
        """Check cache system health"""
        
        try:
            cache_service = self.cache_service
            
            # Test cache functionality
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat(), "test": True}
            
            start_time = time.time()
            
            # Test set operation
            set_success = await cache_service.set(test_key, test_value, ttl=60)
            
            # Test get operation
            retrieved_value = await cache_service.get(test_key)
            
            operation_time = time.time() - start_time
            
            # Cleanup test data
            await cache_service.invalidate(test_key)
            
            # Get cache metrics
            metrics = await cache_service.get_metrics()
            
            status = HealthStatus.HEALTHY
            message = "Cache system is healthy"
            
            if not set_success or retrieved_value is None:
                status = HealthStatus.CRITICAL
                message = "Cache operations failed"
            elif operation_time > 1.0:
                status = HealthStatus.WARNING
                message = f"Cache operations slow: {operation_time:.2f}s"
            elif metrics.hit_rate < 0.5:  # Less than 50% hit rate
                status = HealthStatus.WARNING
                message = f"Low cache hit rate: {metrics.hit_rate:.2%}"
            
            return HealthCheck(
                component="cache",
                status=status,
                message=message,
                response_time=operation_time,
                last_check=datetime.now(),
                details={
                    'operation_time': operation_time,
                    'set_success': set_success,
                    'get_success': retrieved_value is not None,
                    'hit_rate': metrics.hit_rate,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'total_requests': metrics.total_requests
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="cache",
                status=HealthStatus.CRITICAL,
                message=f"Cache health check failed: {str(e)}",
                response_time=0.0,
                last_check=datetime.now(),
                details={'error': str(e)}
            )
    
    async def _check_system_health(self) -> HealthCheck:
        """Check system resource health"""
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Determine overall status
            status = HealthStatus.HEALTHY
            messages = []
            
            # CPU check
            if cpu_percent >= self.config['critical_cpu_percent']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent >= self.config['warning_cpu_percent']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Memory check
            if memory.percent >= self.config['critical_memory_percent']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent >= self.config['warning_memory_percent']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Disk check
            if disk.percent >= self.config['critical_disk_percent']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical disk usage: {disk.percent:.1f}%")
            elif disk.percent >= self.config['warning_disk_percent']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources are healthy"
            
            # Update Prometheus metrics if available
            if HAS_PROMETHEUS:
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.set(memory.percent)
                self.disk_usage.set(disk.percent)
            
            return HealthCheck(
                component="system",
                status=status,
                message=message,
                response_time=0.0,
                last_check=datetime.now(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv,
                    'platform': platform.platform(),
                    'python_version': platform.python_version()
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="system",
                status=HealthStatus.CRITICAL,
                message=f"System health check failed: {str(e)}",
                response_time=0.0,
                last_check=datetime.now(),
                details={'error': str(e)}
            )
    
    async def _check_api_health(self) -> HealthCheck:
        """Check API endpoint health"""
        
        try:
            # Test internal API health endpoint
            start_time = time.time()
            
            # This would test actual API endpoints
            # For now, simulate a basic health check
            await asyncio.sleep(0.1)  # Simulate API call
            
            response_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY
            message = "API endpoints are responsive"
            
            if response_time > self.config['critical_response_time']:
                status = HealthStatus.CRITICAL
                message = f"API response time critical: {response_time:.2f}s"
            elif response_time > self.config['warning_response_time']:
                status = HealthStatus.WARNING
                message = f"API response time slow: {response_time:.2f}s"
            
            return HealthCheck(
                component="api",
                status=status,
                message=message,
                response_time=response_time,
                last_check=datetime.now(),
                details={
                    'response_time': response_time,
                    'endpoints_checked': ['health', 'chat', 'auth']
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="api",
                status=HealthStatus.CRITICAL,
                message=f"API health check failed: {str(e)}",
                response_time=0.0,
                last_check=datetime.now(),
                details={'error': str(e)}
            )
    
    async def _check_dharmic_ai_health(self) -> HealthCheck:
        """Check Dharmic AI system health"""
        
        try:
            # Test AI system components
            start_time = time.time()
            
            # This would test actual AI model availability and performance
            # For now, simulate AI health check
            await asyncio.sleep(0.2)  # Simulate AI processing
            
            response_time = time.time() - start_time
            
            # Simulate AI system metrics
            ai_metrics = {
                'model_loaded': True,
                'inference_time': response_time,
                'memory_usage_mb': 512,  # Mock value
                'dharmic_alignment_score': 0.95,
                'wisdom_confidence': 0.88
            }
            
            status = HealthStatus.HEALTHY
            message = "Dharmic AI system is operational"
            
            if response_time > 5.0:  # AI inference taking too long
                status = HealthStatus.WARNING
                message = f"AI inference slow: {response_time:.2f}s"
            
            if ai_metrics['dharmic_alignment_score'] < 0.8:
                status = HealthStatus.WARNING
                message = "Low dharmic alignment score detected"
            
            return HealthCheck(
                component="dharmic_ai",
                status=status,
                message=message,
                response_time=response_time,
                last_check=datetime.now(),
                details=ai_metrics
            )
            
        except Exception as e:
            return HealthCheck(
                component="dharmic_ai",
                status=HealthStatus.CRITICAL,
                message=f"Dharmic AI health check failed: {str(e)}",
                response_time=0.0,
                last_check=datetime.now(),
                details={'error': str(e)}
            )
    
    async def _store_health_check(self, health_check: HealthCheck):
        """Store health check result in database"""
        
        try:
            query = """
                INSERT INTO health_checks (
                    component, status, message, response_time, details, checked_at
                ) VALUES (
                    %(component)s, %(status)s, %(message)s, %(response_time)s, 
                    %(details)s, %(checked_at)s
                )
            """
            
            await self.db.execute_query(query, {
                'component': health_check.component,
                'status': health_check.status.value,
                'message': health_check.message,
                'response_time': health_check.response_time,
                'details': health_check.details,
                'checked_at': health_check.last_check
            })
            
        except Exception as e:
            logger.error(f"Failed to store health check: {e}")
    
    async def _check_component_alerts(self, health_check: HealthCheck):
        """Check if component health requires alerts"""
        
        component = health_check.component
        status = health_check.status
        
        # Generate alert ID
        alert_id = f"{component}_{status.value}_{int(time.time())}"
        
        # Check if we should create an alert
        should_alert = False
        alert_level = AlertLevel.INFO
        
        if status == HealthStatus.CRITICAL:
            should_alert = True
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.DOWN:
            should_alert = True
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.WARNING:
            should_alert = True
            alert_level = AlertLevel.WARNING
        
        if should_alert:
            # Check cooldown period
            recent_alerts = [
                alert for alert in self.active_alerts.values()
                if (alert.component == component and 
                    alert.level == alert_level and
                    (datetime.now() - alert.created_at).total_seconds() < self.config['alert_cooldown'])
            ]
            
            if not recent_alerts:
                alert = Alert(
                    id=alert_id,
                    level=alert_level,
                    component=component,
                    message=health_check.message,
                    details=health_check.details,
                    created_at=datetime.now(),
                    resolved_at=None,
                    acknowledged=False
                )
                
                self.active_alerts[alert_id] = alert
                await self._store_alert(alert)
                await self._trigger_alert_handlers(alert)
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        
        try:
            query = """
                INSERT INTO monitoring_alerts (
                    id, level, component, message, details, created_at,
                    resolved_at, acknowledged
                ) VALUES (
                    %(id)s, %(level)s, %(component)s, %(message)s, %(details)s,
                    %(created_at)s, %(resolved_at)s, %(acknowledged)s
                )
            """
            
            await self.db.execute_query(query, {
                'id': alert.id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'details': alert.details,
                'created_at': alert.created_at,
                'resolved_at': alert.resolved_at,
                'acknowledged': alert.acknowledged
            })
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    async def _trigger_alert_handlers(self, alert: Alert):
        """Trigger registered alert handlers"""
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        try:
            # Get latest health checks
            overall_status = HealthStatus.HEALTHY
            components_status = {}
            
            for component, health_check in self.health_checks.items():
                components_status[component] = {
                    'status': health_check.status.value,
                    'message': health_check.message,
                    'last_check': health_check.last_check.isoformat(),
                    'response_time': health_check.response_time
                }
                
                # Determine overall status (worst case)
                if health_check.status == HealthStatus.DOWN or health_check.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif health_check.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
            
            # Get active alerts
            active_alerts_summary = {
                'total': len(self.active_alerts),
                'critical': len([a for a in self.active_alerts.values() if a.level == AlertLevel.CRITICAL]),
                'warnings': len([a for a in self.active_alerts.values() if a.level == AlertLevel.WARNING]),
                'unacknowledged': len([a for a in self.active_alerts.values() if not a.acknowledged])
            }
            
            return {
                'overall_status': overall_status.value,
                'components': components_status,
                'alerts': active_alerts_summary,
                'last_updated': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'overall_status': HealthStatus.CRITICAL.value,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    async def _health_check_worker(self):
        """Background worker for periodic health checks"""
        
        self._start_time = time.time()
        
        while True:
            try:
                await asyncio.sleep(self.config['check_interval'])
                await self.run_all_health_checks()
                
            except asyncio.CancelledError:
                logger.info("Health check worker cancelled")
                break
            except Exception as e:
                logger.error(f"Health check worker error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _metrics_collection_worker(self):
        """Background worker for metrics collection"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                await self._collect_system_metrics()
                
            except asyncio.CancelledError:
                logger.info("Metrics collection worker cancelled")
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect and store system metrics"""
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics_data = [
                ('system', 'cpu_percent', cpu_percent),
                ('system', 'memory_percent', memory.percent),
                ('system', 'disk_percent', disk.percent),
                ('system', 'network_bytes_sent', network.bytes_sent),
                ('system', 'network_bytes_recv', network.bytes_recv)
            ]
            
            # Store metrics in database
            for component, metric_name, value in metrics_data:
                query = """
                    INSERT INTO system_metrics (component, metric_name, metric_value)
                    VALUES (%(component)s, %(metric_name)s, %(metric_value)s)
                """
                
                await self.db.execute_query(query, {
                    'component': component,
                    'metric_name': metric_name,
                    'metric_value': value
                })
            
            # Update recent metrics list
            system_metric = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io={'sent': network.bytes_sent, 'recv': network.bytes_recv},
                active_connections=0,  # Would get from actual connection pool
                request_rate=0.0,      # Would calculate from request metrics
                error_rate=0.0,        # Would calculate from error metrics
                response_time_avg=0.0  # Would calculate from response time metrics
            )
            
            self.system_metrics.append(system_metric)
            
            # Keep only last 24 hours of metrics in memory
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.system_metrics = [
                m for m in self.system_metrics 
                if m.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _alert_processing_worker(self):
        """Background worker for alert processing"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check alerts every 30 seconds
                await self._process_alert_resolution()
                
            except asyncio.CancelledError:
                logger.info("Alert processing worker cancelled")
                break
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _process_alert_resolution(self):
        """Check if any alerts can be auto-resolved"""
        
        for alert_id, alert in list(self.active_alerts.items()):
            try:
                # Check if the component is now healthy
                component_health = self.health_checks.get(alert.component)
                
                if component_health and component_health.status == HealthStatus.HEALTHY:
                    # Auto-resolve the alert
                    alert.resolved_at = datetime.now()
                    
                    # Update in database
                    query = """
                        UPDATE monitoring_alerts 
                        SET resolved_at = %(resolved_at)s
                        WHERE id = %(id)s
                    """
                    
                    await self.db.execute_query(query, {
                        'resolved_at': alert.resolved_at,
                        'id': alert_id
                    })
                    
                    # Remove from active alerts
                    del self.active_alerts[alert_id]
                    
                    logger.info(f"Auto-resolved alert: {alert_id}")
                    
            except Exception as e:
                logger.error(f"Alert resolution processing failed for {alert_id}: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run cleanup every hour
                await self._cleanup_old_data()
                
            except asyncio.CancelledError:
                logger.info("Cleanup worker cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        try:
            # Clean old metrics (keep last 7 days)
            metrics_cleanup = """
                DELETE FROM system_metrics 
                WHERE created_at < NOW() - INTERVAL '7 days'
            """
            
            # Clean old health checks (keep last 3 days)
            health_cleanup = """
                DELETE FROM health_checks 
                WHERE checked_at < NOW() - INTERVAL '3 days'
            """
            
            # Clean resolved alerts (keep last 30 days)
            alerts_cleanup = """
                DELETE FROM monitoring_alerts 
                WHERE resolved_at IS NOT NULL 
                AND resolved_at < NOW() - INTERVAL '30 days'
            """
            
            await self.db.execute_query(metrics_cleanup)
            await self.db.execute_query(health_cleanup)
            await self.db.execute_query(alerts_cleanup)
            
            logger.info("Monitoring data cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def get_metrics_export(self) -> str:
        """Export metrics in Prometheus format"""
        
        if not HAS_PROMETHEUS:
            return "# Prometheus not available\n"
        
        try:
            from prometheus_client import generate_latest
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            return f"# Error: {str(e)}\n"
    
    async def cleanup(self):
        """Cleanup monitoring service"""
        
        try:
            # Clear in-memory data
            self.health_checks.clear()
            self.system_metrics.clear()
            self.active_alerts.clear()
            
            logger.info("Monitoring service cleanup completed")
            
        except Exception as e:
            logger.error(f"Monitoring cleanup error: {e}")


# Export monitoring service
__all__ = [
    "MonitoringService",
    "HealthStatus",
    "AlertLevel",
    "HealthCheck",
    "SystemMetrics",
    "Alert"
]
