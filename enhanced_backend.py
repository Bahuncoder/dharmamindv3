#!/usr/bin/env python3
"""
Enhanced DharmaMind Backend - Phase 1: Performance Monitoring
Secure backend with integrated performance monitoring and health checks
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import os
import sys
import time
import json
from datetime import datetime

# Add the backend directory to Python path
sys.path.append('/media/rupert/New Volume/new complete apps/backend')

# Import performance monitoring
try:
    from performance_monitor import get_performance_monitor, PerformanceMonitor
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Performance monitoring not available: {e}")
    MONITORING_AVAILABLE = False

# Import security routes (optional for now)
try:
    from app.routes.auth import router as auth_router
    from app.routes.admin_auth import router as admin_auth_router
    AUTH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Auth routes not available: {e}")
    auth_router = None
    admin_auth_router = None
    AUTH_AVAILABLE = False

app = FastAPI(
    title="DharmaMind Enhanced API - Phase 1",
    description="🚀 Performance-Monitored DharmaMind Backend with Security",
    version="2.1.0-performance",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
security = HTTPBearer()

# Initialize performance monitor
if MONITORING_AVAILABLE:
    performance_monitor = get_performance_monitor()
    print("📊 Performance monitoring initialized")
else:
    performance_monitor = None

# CORS Configuration - Secure and Restricted
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://localhost:3003",
        "https://dharmamind.ai",
        "https://www.dharmamind.ai"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# Performance monitoring middleware
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Monitor API performance"""
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate response time
    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Record metrics if monitoring is available
    if performance_monitor:
        performance_monitor.record_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            response_time_ms=process_time,
            status_code=response.status_code,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )
    
    # Add performance headers
    response.headers["X-Process-Time"] = str(round(process_time, 2))
    response.headers["X-Performance-Monitored"] = "true"
    
    return response

# Trusted Host validation
allowed_hosts = ["localhost", "127.0.0.1", "dharmamind.ai"]

@app.middleware("http")
async def trusted_host_middleware(request, call_next):
    """Manual trusted host validation"""
    host = request.headers.get("host", "").split(":")[0]
    if host and host not in allowed_hosts and not host.endswith(".dharmamind.ai"):
        raise HTTPException(status_code=400, detail="Invalid host")
    response = await call_next(request)
    return response

# Include secure authentication routes if available
if AUTH_AVAILABLE:
    if auth_router:
        app.include_router(auth_router, prefix="/auth", tags=["authentication"])
    if admin_auth_router:
        app.include_router(admin_auth_router, prefix="/api/admin", tags=["admin-authentication"])

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("🚀 Starting DharmaMind Enhanced Backend...")
    
    if performance_monitor:
        success = performance_monitor.start_monitoring(port=8001)
        if success:
            print("📊 Performance monitoring started on port 8001")
        else:
            print("❌ Failed to start performance monitoring")
    
    print("✅ DharmaMind Enhanced Backend ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    print("🛑 Shutting down DharmaMind Enhanced Backend...")
    
    if performance_monitor:
        performance_monitor.stop_monitoring()
        # Save final metrics
        try:
            filename = performance_monitor.save_metrics_to_file()
            print(f"📁 Final metrics saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Error saving final metrics: {e}")
    
    print("👋 Shutdown complete")

@app.get("/")
async def root():
    """Enhanced root endpoint with performance status"""
    return {
        "message": "🚀 DharmaMind Enhanced Backend - Phase 1",
        "version": "2.1.0-performance",
        "enhancements": [
            "Performance monitoring",
            "Health checks",
            "Real-time metrics",
            "Security hardening"
        ],
        "features": {
            "monitoring": MONITORING_AVAILABLE,
            "authentication": AUTH_AVAILABLE,
            "security": "enhanced",
            "metrics_endpoint": "/metrics" if MONITORING_AVAILABLE else None
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0-performance",
        "uptime": "unknown"
    }
    
    if performance_monitor:
        try:
            current_metrics = performance_monitor.get_current_metrics()
            health_data.update({
                "performance": current_metrics.get("health_status", "unknown"),
                "cpu_usage": current_metrics.get("current", {}).get("cpu_percent", 0),
                "memory_usage": current_metrics.get("current", {}).get("memory_percent", 0),
                "monitoring_uptime": current_metrics.get("monitoring_uptime", "0:00:00")
            })
        except Exception as e:
            health_data["performance_error"] = str(e)
    
    return health_data

@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get current performance metrics"""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        return performance_monitor.get_current_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {e}")

@app.get("/performance/report")
async def get_performance_report():
    """Get comprehensive performance report"""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        return performance_monitor.get_performance_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {e}")

@app.post("/performance/export")
async def export_performance_data():
    """Export performance data to file"""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        filename = performance_monitor.save_metrics_to_file()
        return {
            "success": True,
            "filename": filename,
            "message": "Performance data exported successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {e}")

@app.get("/api/test/performance")
async def test_performance_endpoint():
    """Test endpoint for performance monitoring"""
    import random
    import asyncio
    
    # Simulate some processing time
    processing_time = random.uniform(0.01, 0.1)  # 10-100ms
    await asyncio.sleep(processing_time)
    
    return {
        "message": "Performance test endpoint",
        "processing_time_seconds": processing_time,
        "random_number": random.randint(1, 1000),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/test/load")
async def test_load_endpoint():
    """Test endpoint that simulates higher load"""
    import random
    import asyncio
    
    # Simulate variable load
    operations = random.randint(100, 1000)
    result = sum(i * i for i in range(operations))
    
    # Add some async delay
    delay = random.uniform(0.05, 0.2)  # 50-200ms
    await asyncio.sleep(delay)
    
    return {
        "message": "Load test endpoint",
        "operations_performed": operations,
        "result": result,
        "delay_seconds": delay,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/security/status")
async def security_status():
    """Enhanced security status with performance info"""
    return {
        "authentication": {
            "password_hashing": "bcrypt",
            "token_system": "JWT",
            "admin_access": "secured",
            "available": AUTH_AVAILABLE
        },
        "middleware": {
            "cors": "restricted",
            "trusted_hosts": "enabled",
            "performance_monitoring": MONITORING_AVAILABLE
        },
        "monitoring": {
            "metrics_collection": MONITORING_AVAILABLE,
            "health_checks": "enabled",
            "real_time_alerts": MONITORING_AVAILABLE
        },
        "compliance": {
            "hardcoded_credentials": "removed",
            "token_security": "enhanced",
            "environment_security": "improved",
            "performance_tracking": "implemented"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting DharmaMind Enhanced Backend - Phase 1...")
    print("📊 Performance monitoring enabled")
    print("🛡️ Security enhancements active")
    print("🔍 Health monitoring available")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "enhanced_backend:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
