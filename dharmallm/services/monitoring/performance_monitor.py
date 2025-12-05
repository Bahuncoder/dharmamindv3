"""Performance monitoring middleware and utilities"""

import logging
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API performance"""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.debug(f"{request.method} {request.url.path} - {process_time:.3f}s")
        return response


async def init_performance_monitoring():
    """Initialize performance monitoring system"""
    logger.info("Performance monitoring initialized")
    return True
