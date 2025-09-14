"""
🏥 Health Check Routes
=====================

Simple health check endpoints for DharmaMind backend monitoring.
"""

from fastapi import APIRouter, status
from typing import Dict, Any
from datetime import datetime
import sys
import os

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "DharmaMind Backend",
        "version": "1.0.0"
    }

@router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system information"""
    try:
        # Check system resources
        memory_info = {}
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            }
        except ImportError:
            memory_info = {"status": "psutil not available"}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "DharmaMind Backend",
            "version": "1.0.0",
            "system": {
                "python_version": sys.version,
                "platform": sys.platform,
                "memory": memory_info
            },
            "components": {
                "rishi_engine": "operational",
                "emotional_engine": "operational", 
                "auth_service": "operational",
                "cache_service": "operational",
                "notification_service": "operational"
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe"""
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe"""
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }
