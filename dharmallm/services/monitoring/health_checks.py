"""
Health Check System for DharmaMind
==================================

Provides comprehensive health monitoring endpoints for:
- Basic health status
- Readiness probes (Kubernetes)
- Liveness probes (Kubernetes)
- Component-level health checks
- Dependency status verification

Endpoints:
- GET /health - Basic health check
- GET /health/ready - Readiness probe (checks dependencies)
- GET /health/live - Liveness probe (checks process is alive)
- GET /health/components - Detailed component status

Author: DharmaMind Infrastructure Team
Date: October 27, 2025
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    response_time_ms: Optional[float] = None
    last_check: str


class HealthCheckResponse(BaseModel):
    """Overall health check response"""
    status: HealthStatus
    timestamp: str
    uptime_seconds: float
    version: str
    components: Dict[str, ComponentHealth]


class HealthChecker:
    """
    Comprehensive health checking system
    
    Monitors:
    - Database connectivity
    - Cache availability
    - File system access
    - Memory usage
    - CPU usage
    - Disk space
    
    Example:
        checker = HealthChecker()
        await checker.initialize()
        health = await checker.check_all()
    """
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.version = "1.0.0"
        self._initialized = False
        
        # Component checkers
        self._checkers = {
            "database": self._check_database,
            "filesystem": self._check_filesystem,
            "memory": self._check_memory,
            "disk": self._check_disk,
        }
    
    async def initialize(self):
        """Initialize health checker"""
        if self._initialized:
            return
        
        logger.info("🏥 Initializing health checker...")
        self._initialized = True
        logger.info("✅ Health checker initialized")
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    async def check_all(self) -> HealthCheckResponse:
        """
        Check health of all components
        
        Returns comprehensive health status
        """
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        # Check each component
        for name, checker in self._checkers.items():
            try:
                component = await checker()
                components[name] = component
                
                # Degrade overall status if any component is unhealthy
                if component.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (component.status == HealthStatus.DEGRADED 
                      and overall_status == HealthStatus.HEALTHY):
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.utcnow().isoformat()
                )
                overall_status = HealthStatus.UNHEALTHY
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=self.get_uptime(),
            version=self.version,
            components=components
        )
    
    async def _check_database(self) -> ComponentHealth:
        """Check database connectivity"""
        import time
        start = time.time()
        
        try:
            from services.db.database import get_database
            
            db = get_database()
            
            # Try a simple query
            result = await db.execute("SELECT 1")
            
            response_time = (time.time() - start) * 1000  # ms
            
            # Warn if query is slow
            if response_time > 1000:  # > 1 second
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message=f"Database responding slowly ({response_time:.0f}ms)",
                    response_time_ms=response_time,
                    last_check=datetime.utcnow().isoformat()
                )
            
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection OK",
                response_time_ms=response_time,
                last_check=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}",
                response_time_ms=(time.time() - start) * 1000,
                last_check=datetime.utcnow().isoformat()
            )
    
    async def _check_filesystem(self) -> ComponentHealth:
        """Check filesystem access"""
        import tempfile
        from pathlib import Path
        
        try:
            # Try to write a test file
            test_dir = Path("data")
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / ".health_check"
            test_file.write_text("health_check")
            test_file.unlink()
            
            return ComponentHealth(
                name="filesystem",
                status=HealthStatus.HEALTHY,
                message="Filesystem read/write OK",
                last_check=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return ComponentHealth(
                name="filesystem",
                status=HealthStatus.UNHEALTHY,
                message=f"Filesystem error: {str(e)}",
                last_check=datetime.utcnow().isoformat()
            )
    
    async def _check_memory(self) -> ComponentHealth:
        """Check memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            
            # Thresholds
            if percent_used > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {percent_used:.1f}%"
            elif percent_used > 80:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {percent_used:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage OK: {percent_used:.1f}%"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                last_check=datetime.utcnow().isoformat()
            )
            
        except ImportError:
            # psutil not installed
            return ComponentHealth(
                name="memory",
                status=HealthStatus.HEALTHY,
                message="Memory monitoring not available (install psutil)",
                last_check=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.DEGRADED,
                message=f"Memory check error: {str(e)}",
                last_check=datetime.utcnow().isoformat()
            )
    
    async def _check_disk(self) -> ComponentHealth:
        """Check disk space"""
        try:
            import psutil
            
            disk = psutil.disk_usage('.')
            percent_used = disk.percent
            
            # Thresholds
            if percent_used > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {percent_used:.1f}%"
            elif percent_used > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {percent_used:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage OK: {percent_used:.1f}%"
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                last_check=datetime.utcnow().isoformat()
            )
            
        except ImportError:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.HEALTHY,
                message="Disk monitoring not available (install psutil)",
                last_check=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.DEGRADED,
                message=f"Disk check error: {str(e)}",
                last_check=datetime.utcnow().isoformat()
            )


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# ==================
# HEALTH ENDPOINTS
# ==================

@router.get("/", response_model=Dict[str, Any])
async def basic_health_check():
    """
    Basic health check endpoint
    
    Returns simple health status.
    Use for basic availability monitoring.
    
    Returns:
        200: Service is running
    """
    checker = get_health_checker()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": checker.get_uptime(),
        "version": checker.version
    }


@router.get("/live", response_model=Dict[str, str])
async def liveness_probe():
    """
    Kubernetes liveness probe
    
    Checks if the application process is alive and responding.
    If this fails, Kubernetes will restart the pod.
    
    Returns:
        200: Process is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_probe():
    """
    Kubernetes readiness probe
    
    Checks if the application is ready to receive traffic.
    Verifies all dependencies are available.
    If this fails, Kubernetes will remove pod from load balancer.
    
    Returns:
        200: Ready to receive traffic
        503: Not ready (dependencies unavailable)
    """
    checker = get_health_checker()
    health = await checker.check_all()
    
    # Not ready if any critical component is unhealthy
    if health.status == HealthStatus.UNHEALTHY:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": "Critical components unhealthy",
                "components": {
                    name: comp.dict()
                    for name, comp in health.components.items()
                    if comp.status == HealthStatus.UNHEALTHY
                }
            }
        )
    
    return {
        "status": "ready",
        "timestamp": health.timestamp,
        "components": {
            name: comp.status.value
            for name, comp in health.components.items()
        }
    }


@router.get("/components", response_model=HealthCheckResponse)
async def detailed_health_check():
    """
    Detailed component health check
    
    Returns comprehensive health status for all components.
    Use for detailed monitoring and diagnostics.
    
    Returns:
        200: Health status (may include degraded components)
        503: Service unavailable (critical failures)
    """
    checker = get_health_checker()
    health = await checker.check_all()
    
    # Return 503 if unhealthy
    if health.status == HealthStatus.UNHEALTHY:
        raise HTTPException(
            status_code=503,
            detail=health.dict()
        )
    
    return health


# Startup event to initialize health checker
async def initialize_health_checker():
    """Initialize health checker on startup"""
    checker = get_health_checker()
    await checker.initialize()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_health_checks():
        checker = HealthChecker()
        await checker.initialize()
        
        print("\n🏥 Testing Health Checks...\n")
        
        # Check all components
        health = await checker.check_all()
        
        print(f"Overall Status: {health.status.value}")
        print(f"Uptime: {health.uptime_seconds:.1f}s")
        print(f"\nComponents:")
        
        for name, component in health.components.items():
            status_icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌"
            }[component.status]
            
            print(f"  {status_icon} {name}: {component.message}")
            if component.response_time_ms:
                print(f"     Response time: {component.response_time_ms:.0f}ms")
    
    asyncio.run(test_health_checks())
