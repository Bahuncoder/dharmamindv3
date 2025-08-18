#!/usr/bin/env python3
"""
DharmaMind Performance Monitoring System
Real-time system health, metrics, and performance tracking
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import json
import os

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]

@dataclass
class APIMetrics:
    """API performance metrics"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    timestamp: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter('dharmamind_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
        self.response_time = Histogram('dharmamind_response_time_seconds', 'Response time', ['method', 'endpoint'])
        self.cpu_usage = Gauge('dharmamind_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('dharmamind_memory_usage_percent', 'Memory usage percentage')
        self.active_connections = Gauge('dharmamind_active_connections', 'Active connections')
        
        # Storage for metrics
        self.system_metrics_history: List[SystemMetrics] = []
        self.api_metrics_history: List[APIMetrics] = []
        
        # Configuration
        self.max_history_size = 1000
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance thresholds
        self.cpu_warning_threshold = 80.0
        self.memory_warning_threshold = 85.0
        self.response_time_warning_ms = 500.0
        
        print("🔧 Performance Monitor initialized")

    def start_monitoring(self, port: int = 8001):
        """Start the monitoring system"""
        try:
            # Start Prometheus metrics server
            start_http_server(port)
            print(f"📊 Prometheus metrics server started on port {port}")
            
            # Start system monitoring thread
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._system_monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            print("🚀 Performance monitoring started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            return False

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        print("🛑 Performance monitoring stopped")

    def _system_monitoring_loop(self):
        """Background system monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self._update_prometheus_metrics(metrics)
                self._store_system_metrics(metrics)
                self._check_performance_alerts(metrics)
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                print(f"⚠️ Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics  
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]  # Windows doesn't have getloadavg
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
        except Exception as e:
            print(f"❌ Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0, memory_percent=0.0, memory_used_mb=0.0,
                memory_available_mb=0.0, disk_usage_percent=0.0,
                network_bytes_sent=0, network_bytes_recv=0,
                process_count=0, load_average=[0.0, 0.0, 0.0]
            )

    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics"""
        self.cpu_usage.set(metrics.cpu_percent)
        self.memory_usage.set(metrics.memory_percent)

    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in history"""
        self.system_metrics_history.append(metrics)
        
        # Limit history size
        if len(self.system_metrics_history) > self.max_history_size:
            self.system_metrics_history = self.system_metrics_history[-self.max_history_size:]

    def _check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.cpu_warning_threshold:
            alerts.append(f"HIGH CPU: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.memory_warning_threshold:
            alerts.append(f"HIGH MEMORY: {metrics.memory_percent:.1f}%")
        
        if alerts:
            print(f"⚠️ PERFORMANCE ALERT: {' | '.join(alerts)}")

    def record_api_request(self, endpoint: str, method: str, response_time_ms: float, 
                          status_code: int, user_agent: str = None, ip_address: str = None):
        """Record API request metrics"""
        # Update Prometheus metrics
        self.request_count.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.response_time.labels(method=method, endpoint=endpoint).observe(response_time_ms / 1000)
        
        # Store in history
        api_metric = APIMetrics(
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time_ms,
            status_code=status_code,
            timestamp=datetime.now().isoformat(),
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        self.api_metrics_history.append(api_metric)
        
        # Limit history size
        if len(self.api_metrics_history) > self.max_history_size:
            self.api_metrics_history = self.api_metrics_history[-self.max_history_size:]
        
        # Check for slow responses
        if response_time_ms > self.response_time_warning_ms:
            print(f"🐌 SLOW REQUEST: {method} {endpoint} took {response_time_ms:.1f}ms")

    def get_current_metrics(self) -> Dict:
        """Get current system performance"""
        if not self.system_metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.system_metrics_history[-1]
        
        # Calculate recent averages
        recent_metrics = self.system_metrics_history[-6:]  # Last 6 readings (1 minute)
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            "current": asdict(latest),
            "averages": {
                "cpu_percent_1min": round(avg_cpu, 2),
                "memory_percent_1min": round(avg_memory, 2)
            },
            "health_status": self._get_health_status(latest),
            "total_requests": len(self.api_metrics_history),
            "monitoring_uptime": self._get_monitoring_uptime()
        }

    def _get_health_status(self, metrics: SystemMetrics) -> str:
        """Determine system health status"""
        if metrics.cpu_percent > 90 or metrics.memory_percent > 95:
            return "critical"
        elif metrics.cpu_percent > 80 or metrics.memory_percent > 85:
            return "warning"
        else:
            return "healthy"

    def _get_monitoring_uptime(self) -> str:
        """Get monitoring uptime"""
        if self.system_metrics_history:
            start_time = datetime.fromisoformat(self.system_metrics_history[0].timestamp)
            uptime = datetime.now() - start_time
            return str(uptime).split('.')[0]  # Remove microseconds
        return "0:00:00"

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.system_metrics_history or not self.api_metrics_history:
            return {"error": "Insufficient data for report"}
        
        # Recent API metrics (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_api_metrics = [
            m for m in self.api_metrics_history 
            if datetime.fromisoformat(m.timestamp) > one_hour_ago
        ]
        
        # Calculate API statistics
        if recent_api_metrics:
            response_times = [m.response_time_ms for m in recent_api_metrics]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # Status code distribution
            status_codes = {}
            for metric in recent_api_metrics:
                status_codes[metric.status_code] = status_codes.get(metric.status_code, 0) + 1
        else:
            avg_response_time = max_response_time = min_response_time = 0
            status_codes = {}
        
        # System performance summary
        recent_system = self.system_metrics_history[-36:]  # Last 6 minutes
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        
        return {
            "report_generated": datetime.now().isoformat(),
            "monitoring_period": {
                "start": self.system_metrics_history[0].timestamp,
                "end": self.system_metrics_history[-1].timestamp,
                "duration_hours": len(self.system_metrics_history) * 10 / 3600  # 10 sec intervals
            },
            "system_performance": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "current_health": self._get_health_status(self.system_metrics_history[-1])
            },
            "api_performance": {
                "total_requests": len(recent_api_metrics),
                "avg_response_time_ms": round(avg_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2),
                "min_response_time_ms": round(min_response_time, 2),
                "status_code_distribution": status_codes,
                "slow_requests_count": len([m for m in recent_api_metrics if m.response_time_ms > self.response_time_warning_ms])
            },
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not self.system_metrics_history:
            return ["Insufficient data for recommendations"]
        
        latest = self.system_metrics_history[-1]
        
        if latest.cpu_percent > 80:
            recommendations.append("Consider CPU optimization or scaling")
        
        if latest.memory_percent > 85:
            recommendations.append("Monitor memory usage, consider optimization")
        
        if self.api_metrics_history:
            slow_requests = [m for m in self.api_metrics_history[-100:] if m.response_time_ms > 500]
            if len(slow_requests) > 10:
                recommendations.append("Multiple slow API responses detected - optimize endpoints")
        
        if not recommendations:
            recommendations.append("System performance is good - continue monitoring")
        
        return recommendations

    def save_metrics_to_file(self, filename: str = None):
        """Save metrics to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        data = {
            "export_time": datetime.now().isoformat(),
            "system_metrics": [asdict(m) for m in self.system_metrics_history],
            "api_metrics": [asdict(m) for m in self.api_metrics_history],
            "summary": self.get_performance_report()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Metrics saved to: {filename}")
        return filename

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return performance_monitor

if __name__ == "__main__":
    # Test the performance monitor
    monitor = PerformanceMonitor()
    
    print("🚀 Starting Performance Monitor Test...")
    monitor.start_monitoring(port=8001)
    
    try:
        # Simulate some API requests
        for i in range(5):
            monitor.record_api_request(
                endpoint="/api/test",
                method="GET",
                response_time_ms=50.0 + (i * 10),
                status_code=200
            )
            time.sleep(1)
        
        # Wait for some system metrics
        time.sleep(15)
        
        # Get current metrics
        current = monitor.get_current_metrics()
        print("\n📊 Current Metrics:")
        print(json.dumps(current, indent=2))
        
        # Generate performance report
        report = monitor.get_performance_report()
        print("\n📈 Performance Report:")
        print(json.dumps(report, indent=2))
        
        # Save metrics
        filename = monitor.save_metrics_to_file()
        print(f"\n💾 Metrics saved to: {filename}")
        
        print("\n✅ Performance Monitor test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
    finally:
        monitor.stop_monitoring()
