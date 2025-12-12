"""
DharmaMind API Routes Package

This package contains all API route handlers for the DharmaMind platform.
"""

from fastapi import APIRouter

# Import individual routers
from .auth import router as auth_router
from .health import router as health_router

__all__ = [
    "auth_router",
    "health_router",
]
