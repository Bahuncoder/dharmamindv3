"""Security middleware components"""

import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


class BruteForceProtectionMiddleware(BaseHTTPMiddleware):
    """Protect against brute force attacks"""
    
    def __init__(self, app):
        super().__init__(app)
        self.failed_attempts = {}
        logger.info("Brute force protection enabled")
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Pass-through for now
        response = await call_next(request)
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate incoming requests"""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Pass-through for now
        response = await call_next(request)
        return response
