"""
📊 Advanced Performance Monitoring & Metrics System

Comprehensive performance tracking, metrics collection, and real-time monitoring
for enterprise-grade system observability.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
import redis
from fastapi import Request, Response
import threading

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class PerformanceAlert:
    """Performance alert configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 1000", "< 0.5"
    threshold: float
    level: AlertLevel
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: Optional[datetime] = None

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    uptime: float
    timestamp: datetime

class MetricsCollector:
    """Advanced metrics collection and aggregation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics_prefix = "metrics:"
        self.alerts_prefix = "alerts:"
        self.retention_days = 7
        
        # In-memory metrics for real-time access
        self.real_time_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Performance alerts
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # System monitoring
        self.system_metrics_enabled = True
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        
    async def start_collection(self):
        """Start metrics collection background task"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                await asyncio.sleep(self.collection_interval)
                
                # Collect system metrics
                if self.system_metrics_enabled:
                    await self._collect_system_metrics()
                
                # Process and store metrics
                await self._process_metrics()
                
                # Check alerts
                await self._check_alerts()
                
                # Cleanup old data
                await self._cleanup_old_metrics()
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge("system.cpu.percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system.memory.percent", memory.percent)
            self.record_gauge("system.memory.used", memory.used)
            self.record_gauge("system.memory.available", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_gauge("system.disk.percent", disk_percent)
            self.record_gauge("system.disk.used", disk.used)
            self.record_gauge("system.disk.free", disk.free)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.record_counter("system.network.bytes_sent", network.bytes_sent)
            self.record_counter("system.network.bytes_recv", network.bytes_recv)
            self.record_counter("system.network.packets_sent", network.packets_sent)
            self.record_counter("system.network.packets_recv", network.packets_recv)
            
            # Process metrics
            process_count = len(psutil.pids())
            self.record_gauge("system.processes.count", process_count)
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
                self.record_gauge("system.load.1min", load_avg[0])
                self.record_gauge("system.load.5min", load_avg[1])
                self.record_gauge("system.load.15min", load_avg[2])
            except AttributeError:
                # Not available on all systems
                pass
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record counter metric"""
        self.counters[name] += value
        self._add_metric_point(name, value, MetricType.COUNTER, tags)
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record gauge metric"""
        self.gauges[name] = value
        self._add_metric_point(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record histogram metric"""
        self.histograms[name].append(value)
        
        # Keep only recent values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        self._add_metric_point(name, value, MetricType.HISTOGRAM, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record timer metric"""
        self.timers[name].append(duration)
        
        # Keep only recent values
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
        
        self._add_metric_point(name, duration, MetricType.TIMER, tags)
    
    def _add_metric_point(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str] = None):
        """Add metric point to real-time collection"""
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        self.real_time_metrics[name].append(point)
    
    async def _process_metrics(self):
        """Process and aggregate metrics"""
        
        current_time = int(time.time())
        
        # Store aggregated metrics in Redis
        for name, points in self.real_time_metrics.items():
            if not points:
                continue
            
            # Calculate aggregations for the last minute
            recent_points = [p for p in points if (datetime.utcnow() - p.timestamp).total_seconds() < 60]
            
            if recent_points:
                values = [p.value for p in recent_points]
                
                aggregated = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "timestamp": current_time
                }
                
                if len(values) > 1:
                    aggregated["stddev"] = statistics.stdev(values)
                    aggregated["p50"] = statistics.median(values)
                    aggregated["p95"] = self._percentile(values, 0.95)
                    aggregated["p99"] = self._percentile(values, 0.99)
                
                # Store in Redis
                metric_key = f"{self.metrics_prefix}{name}:{current_time // 60}"  # Per minute
                self.redis.setex(metric_key, 86400 * self.retention_days, json.dumps(aggregated))
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    async def _check_alerts(self):
        """Check performance alerts"""
        
        current_time = datetime.utcnow()
        
        for alert in self.alerts:
            if not alert.enabled:
                continue
            
            # Check cooldown period
            if (alert.last_triggered and 
                (current_time - alert.last_triggered).total_seconds() < alert.cooldown_seconds):
                continue
            
            # Get current metric value
            current_value = self.gauges.get(alert.metric_name)
            if current_value is None:
                continue
            
            # Check condition
            condition_met = self._evaluate_condition(current_value, alert.condition, alert.threshold)
            
            if condition_met:
                alert.last_triggered = current_time
                await self._trigger_alert(alert, current_value)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        
        if condition.startswith(">"):
            return value > threshold
        elif condition.startswith("<"):
            return value < threshold
        elif condition.startswith(">="):
            return value >= threshold
        elif condition.startswith("<="):
            return value <= threshold
        elif condition.startswith("=="):
            return abs(value - threshold) < 0.001  # Float equality
        else:
            return False
    
    async def _trigger_alert(self, alert: PerformanceAlert, current_value: float):
        """Trigger performance alert"""
        
        alert_data = {
            "alert_name": alert.name,
            "metric_name": alert.metric_name,
            "current_value": current_value,
            "threshold": alert.threshold,
            "condition": alert.condition,
            "level": alert.level.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log alert
        log_level = getattr(logger, alert.level.value.lower(), logger.info)
        log_level(f"Performance Alert: {alert.name} - {alert.metric_name} = {current_value}")
        
        # Store alert in Redis
        alert_key = f"{self.alerts_prefix}{int(time.time())}"
        self.redis.setex(alert_key, 86400, json.dumps(alert_data))  # Keep for 24 hours
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metric data"""
        
        cutoff_time = int(time.time()) - (86400 * self.retention_days)
        
        # Clean up metric keys
        for key in self.redis.scan_iter(match=f"{self.metrics_prefix}*"):
            try:
                key_str = key.decode()
                timestamp = int(key_str.split(':')[-1]) * 60  # Convert back to seconds
                if timestamp < cutoff_time:
                    self.redis.delete(key)
            except:
                continue
        
        # Clean up old alerts
        for key in self.redis.scan_iter(match=f"{self.alerts_prefix}*"):
            try:
                key_str = key.decode()
                timestamp = int(key_str.split(':')[-1])
                if timestamp < cutoff_time:
                    self.redis.delete(key)
            except:
                continue
    
    def add_alert(self, alert: PerformanceAlert):
        """Add performance alert"""
        self.alerts.append(alert)
        logger.info(f"Added performance alert: {alert.name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    async def get_metrics_summary(self, time_range: int = 3600) -> Dict[str, Any]:
        """Get metrics summary for specified time range"""
        
        end_time = int(time.time())
        start_time = end_time - time_range
        
        summary = {
            "time_range": time_range,
            "start_time": start_time,
            "end_time": end_time,
            "metrics": {},
            "system_health": await self._get_current_system_health(),
            "alerts": await self._get_recent_alerts(time_range)
        }
        
        # Aggregate metrics for time range
        for key in self.redis.scan_iter(match=f"{self.metrics_prefix}*"):
            try:
                key_str = key.decode()
                parts = key_str.split(':')
                metric_name = ':'.join(parts[1:-1])
                timestamp = int(parts[-1]) * 60
                
                if start_time <= timestamp <= end_time:
                    data = json.loads(self.redis.get(key))
                    
                    if metric_name not in summary["metrics"]:
                        summary["metrics"][metric_name] = {
                            "data_points": [],
                            "avg": 0,
                            "min": float('inf'),
                            "max": float('-inf'),
                            "total": 0
                        }
                    
                    metric_summary = summary["metrics"][metric_name]
                    metric_summary["data_points"].append(data)
                    metric_summary["min"] = min(metric_summary["min"], data["min"])
                    metric_summary["max"] = max(metric_summary["max"], data["max"])
                    metric_summary["total"] += data["sum"]
                    
            except Exception as e:
                logger.error(f"Error processing metric key {key}: {e}")
                continue
        
        # Calculate overall averages
        for metric_name, metric_data in summary["metrics"].items():
            if metric_data["data_points"]:
                total_avg = sum(dp["avg"] for dp in metric_data["data_points"])
                metric_data["avg"] = total_avg / len(metric_data["data_points"])
        
        return summary
    
    async def _get_current_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        
        try:
            return {
                "cpu_percent": self.gauges.get("system.cpu.percent", 0),
                "memory_percent": self.gauges.get("system.memory.percent", 0),
                "disk_percent": self.gauges.get("system.disk.percent", 0),
                "process_count": self.gauges.get("system.processes.count", 0),
                "load_1min": self.gauges.get("system.load.1min", 0),
                "load_5min": self.gauges.get("system.load.5min", 0),
                "load_15min": self.gauges.get("system.load.15min", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {}
    
    async def _get_recent_alerts(self, time_range: int) -> List[Dict[str, Any]]:
        """Get recent alerts within time range"""
        
        alerts = []
        cutoff_time = int(time.time()) - time_range
        
        for key in self.redis.scan_iter(match=f"{self.alerts_prefix}*"):
            try:
                key_str = key.decode()
                timestamp = int(key_str.split(':')[-1])
                
                if timestamp >= cutoff_time:
                    alert_data = json.loads(self.redis.get(key))
                    alerts.append(alert_data)
                    
            except Exception as e:
                logger.error(f"Error processing alert key {key}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return alerts

class PerformanceProfiler:
    """Request performance profiler"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_requests: Dict[str, Dict[str, Any]] = {}
    
    def start_request(self, request_id: str, endpoint: str, method: str) -> Dict[str, Any]:
        """Start profiling a request"""
        
        profile_data = {
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            "start_time": time.time(),
            "phases": {},
            "database_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.active_requests[request_id] = profile_data
        return profile_data
    
    def start_phase(self, request_id: str, phase_name: str):
        """Start timing a request phase"""
        
        if request_id in self.active_requests:
            self.active_requests[request_id]["phases"][phase_name] = {
                "start_time": time.time()
            }
    
    def end_phase(self, request_id: str, phase_name: str):
        """End timing a request phase"""
        
        if (request_id in self.active_requests and 
            phase_name in self.active_requests[request_id]["phases"]):
            
            phase = self.active_requests[request_id]["phases"][phase_name]
            phase["duration"] = time.time() - phase["start_time"]
            phase["end_time"] = time.time()
    
    def record_database_query(self, request_id: str, duration: float):
        """Record database query for request"""
        
        if request_id in self.active_requests:
            self.active_requests[request_id]["database_queries"] += 1
            
            if "database_time" not in self.active_requests[request_id]:
                self.active_requests[request_id]["database_time"] = 0
            
            self.active_requests[request_id]["database_time"] += duration
    
    def record_cache_hit(self, request_id: str):
        """Record cache hit for request"""
        
        if request_id in self.active_requests:
            self.active_requests[request_id]["cache_hits"] += 1
    
    def record_cache_miss(self, request_id: str):
        """Record cache miss for request"""
        
        if request_id in self.active_requests:
            self.active_requests[request_id]["cache_misses"] += 1
    
    def end_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """End profiling a request and collect metrics"""
        
        if request_id not in self.active_requests:
            return None
        
        profile_data = self.active_requests.pop(request_id)
        profile_data["end_time"] = time.time()
        profile_data["total_duration"] = profile_data["end_time"] - profile_data["start_time"]
        
        # Record metrics
        endpoint = profile_data["endpoint"]
        method = profile_data["method"]
        
        self.metrics.record_timer(
            f"request.duration",
            profile_data["total_duration"],
            {"endpoint": endpoint, "method": method}
        )
        
        self.metrics.record_counter(
            f"request.count",
            1.0,
            {"endpoint": endpoint, "method": method}
        )
        
        if "database_time" in profile_data:
            self.metrics.record_timer(
                f"request.database_time",
                profile_data["database_time"],
                {"endpoint": endpoint, "method": method}
            )
        
        self.metrics.record_gauge(
            f"request.database_queries",
            profile_data["database_queries"],
            {"endpoint": endpoint, "method": method}
        )
        
        cache_hit_rate = 0
        total_cache = profile_data["cache_hits"] + profile_data["cache_misses"]
        if total_cache > 0:
            cache_hit_rate = profile_data["cache_hits"] / total_cache
        
        self.metrics.record_gauge(
            f"request.cache_hit_rate",
            cache_hit_rate,
            {"endpoint": endpoint, "method": method}
        )
        
        return profile_data

# Performance monitoring middleware
class PerformanceMiddleware:
    """FastAPI middleware for performance monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.profiler = PerformanceProfiler(metrics_collector)
    
    async def __call__(self, request: Request, call_next):
        """Process request through performance monitoring"""
        
        # Generate request ID
        request_id = f"{int(time.time() * 1000000)}-{id(request)}"
        
        # Start profiling
        profile = self.profiler.start_request(
            request_id=request_id,
            endpoint=str(request.url.path),
            method=request.method
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.performance_profile = profile
        
        try:
            # Process request
            self.profiler.start_phase(request_id, "request_processing")
            response = await call_next(request)
            self.profiler.end_phase(request_id, "request_processing")
            
            # Add performance headers
            final_profile = self.profiler.end_request(request_id)
            if final_profile:
                response.headers["X-Response-Time"] = f"{final_profile['total_duration']:.3f}s"
                response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.metrics.record_counter(
                "request.errors",
                1.0,
                {"endpoint": str(request.url.path), "method": request.method}
            )
            
            self.profiler.end_request(request_id)
            raise

# Global instances
metrics_collector: Optional[MetricsCollector] = None
performance_profiler: Optional[PerformanceProfiler] = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    if metrics_collector is None:
        raise RuntimeError("Metrics collector not initialized")
    return metrics_collector

def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler"""
    if performance_profiler is None:
        raise RuntimeError("Performance profiler not initialized")
    return performance_profiler

def init_performance_monitoring(redis_client: redis.Redis) -> tuple[MetricsCollector, PerformanceProfiler]:
    """Initialize performance monitoring system"""
    global metrics_collector, performance_profiler
    
    metrics_collector = MetricsCollector(redis_client)
    performance_profiler = PerformanceProfiler(metrics_collector)
    
    # Add default alerts
    metrics_collector.add_alert(PerformanceAlert(
        name="High CPU Usage",
        metric_name="system.cpu.percent",
        condition="> 80",
        threshold=80.0,
        level=AlertLevel.WARNING
    ))
    
    metrics_collector.add_alert(PerformanceAlert(
        name="High Memory Usage",
        metric_name="system.memory.percent", 
        condition="> 85",
        threshold=85.0,
        level=AlertLevel.WARNING
    ))
    
    metrics_collector.add_alert(PerformanceAlert(
        name="High Disk Usage",
        metric_name="system.disk.percent",
        condition="> 90",
        threshold=90.0,
        level=AlertLevel.ERROR
    ))
    
    return metrics_collector, performance_profiler
