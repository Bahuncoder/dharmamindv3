"""Distributed tracing for request tracking"""

import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TracingConfig(BaseModel):
    """Configuration for distributed tracing"""
    enabled: bool = False
    service_name: str = "dharmamind"
    endpoint: str = "http://localhost:9411"


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for distributed tracing"""
    
    def __init__(self, app, config: TracingConfig):
        super().__init__(app)
        self.config = config
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Pass-through when tracing disabled
        response = await call_next(request)
        return response


async def initialize_tracing(config: TracingConfig):
    """Initialize distributed tracing system"""
    if config.enabled:
        logger.info(f"Tracing enabled for service: {config.service_name}")
    else:
        logger.info("Tracing disabled")
    return True
