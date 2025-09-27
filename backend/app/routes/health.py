"""
Health Check Routes
Simple health monitoring for the authentication backend
"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import sys
import os

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "dharmamind-backend-auth",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@router.get("/health/detailed")  
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system info"""
    return {
        "status": "healthy",
        "service": "dharmamind-backend-auth",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "process_id": os.getpid()
        },
        "features": {
            "authentication": True,
            "admin_panel": True,
            "mfa": True,
            "feedback": True,
            "security_dashboard": True,
            "chat": False,  # Chat handled by frontend
            "wisdom_generation": False  # Wisdom handled by frontend
        }
    }

@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for container orchestration"""
    return {
        "status": "ready",
        "service": "dharmamind-backend-auth",
        "timestamp": datetime.utcnow().isoformat()
    }
