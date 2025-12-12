#!/usr/bin/env python3
"""
DharmaMind Backend - Enterprise API Gateway
============================================

Central API gateway for all DharmaMind services:
- Authentication & User Management
- Subscription & Billing
- Chat Routing to DharmaLLM
- Usage Tracking & Rate Limiting
- Admin & Security
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
from .routes.auth import router as auth_router
from .routes.health import router as health_router

# Import config
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    logger.info("Starting DharmaMind API Gateway...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    yield
    logger.info("Shutting down DharmaMind API Gateway...")


# Create FastAPI app
app = FastAPI(
    title="DharmaMind API Gateway",
    description="Enterprise API gateway for DharmaMind platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS - Allow all frontends
cors_origins = [
    "http://localhost:3000",   # Chat
    "http://localhost:3001",   # Brand Webpage
    "http://localhost:3002",   # Community
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "https://dharmamind.com",
    "https://dharmamind.ai",
    "https://dharmamind.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(health_router, prefix="/api/v1", tags=["health"])

# Try to include optional routers
try:
    from .routes.llm_router import router as llm_router
    app.include_router(llm_router, tags=["chat"])
except ImportError:
    logger.warning("LLM router not available")

try:
    from .routes.mfa_auth import router as mfa_router
    app.include_router(mfa_router, tags=["mfa"])
except ImportError:
    logger.warning("MFA router not available")

try:
    from .routes.feedback import router as feedback_router
    app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
except ImportError:
    logger.warning("Feedback router not available")

try:
    from .routes.admin_auth import router as admin_auth_router
    app.include_router(admin_auth_router, prefix="/admin", tags=["admin"])
except ImportError:
    logger.warning("Admin auth router not available")

try:
    from .routes.subscription import router as subscription_router
    app.include_router(subscription_router, prefix="/api/v1/subscription", tags=["subscription"])
except ImportError:
    logger.warning("Subscription router not available")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DharmaMind API Gateway",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "auth": "/auth/login",
            "register": "/auth/register",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Endpoint {request.url.path} not found",
            "available_endpoints": ["/", "/docs", "/auth/login", "/api/v1/health"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal errors"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
