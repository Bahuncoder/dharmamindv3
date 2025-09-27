#!/usr/bin/env python3
"""
DharmaMind Backend - Authentication Service
Clean authentication-only backend service.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import routers
from .routes.auth import router as auth_router
from .routes.mfa_auth import router as mfa_router  
from .routes.admin_auth import router as admin_auth_router
from .routes.feedback import router as feedback_router
from .routes.health import router as health_router
from .routes.llm_router import router as llm_router  # Simple LLM routing

# Import database and config
from .db.database import DatabaseManager
from .config import settings

# Import security middleware
from .middleware.security import setup_security_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for state management
database_manager: Optional[DatabaseManager] = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - startup and shutdown."""
    try:
        # Initialize database manager
        global database_manager
        database_manager = DatabaseManager()
        await database_manager.initialize()
        logger.info("✅ Database manager initialized successfully")
        
        # Log startup
        logger.info("🚀 DharmaMind Authentication Backend started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Cleanup
        if database_manager:
            await database_manager.close()
        logger.info("🔄 DharmaMind Authentication Backend shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="DharmaMind Authentication Backend",
    description="Clean authentication and user management service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Setup security middleware
setup_security_middleware(app)

# Include routers - Authentication, Chat, and Admin
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(mfa_router, prefix="/api/v1/mfa", tags=["multi-factor-auth"])
app.include_router(admin_auth_router, prefix="/api/admin", tags=["admin-authentication"])
app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
app.include_router(llm_router, tags=["llm-routing"])  # Simple authenticated LLM routing
app.include_router(health_router, prefix="/api/v1", tags=["health"])

# Note: Chat functionality now handled entirely by frontend
# Note: Wisdom functionality also handled by frontend with comprehensive fallback responses

# Import and include security dashboard router
from .routes.security_dashboard import router as security_dashboard_router
app.include_router(security_dashboard_router, prefix="/api/v1", tags=["security-dashboard"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "DharmaMind Complete Backend",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "Authentication, chat, and AI-powered spiritual guidance",
        "features": ["authentication", "dharmic_chat", "spiritual_ai", "user_management"]
    }

@app.get("/api/v1/status")
async def status():
    """Service status endpoint."""
    try:
        # Check database connection
        db_status = "connected" if database_manager and await database_manager.health_check() else "disconnected"
        
        return {
            "service": "dharmic_chat_complete",
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "features": ["authentication", "dharmic_chat", "spiritual_ai", "dharmallm_integration"]
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url.path),
            "service": "authentication"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error", 
            "message": "An unexpected error occurred",
            "service": "authentication"
        }
    )

# Dependency injection helpers
async def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    if not database_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    return database_manager

# Application metadata
def get_app_info() -> Dict[str, Any]:
    """Get application information."""
    return {
        "name": "DharmaMind Authentication Backend",
        "version": "2.0.0",
        "description": "Clean authentication and user management service",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )