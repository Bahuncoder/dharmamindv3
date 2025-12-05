"""Intelligent caching system with TTL and invalidation"""

import logging
from typing import Optional, Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IntelligentCache:
    """Intelligent caching system"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl
        logger.info(f"Intelligent Cache initialized (TTL: {default_ttl}s)")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            item = self.cache[key]
            if datetime.now() < item["expires"]:
                logger.debug(f"Cache hit: {key}")
                return item["value"]
            else:
                del self.cache[key]
                logger.debug(f"Cache expired: {key}")
        return None
    
    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            "value": value,
            "expires": datetime.now() + timedelta(seconds=ttl)
        }
        logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache deleted: {key}")
    
    async def clear(self):
        """Clear all cache"""
        self.cache.clear()
        logger.info("Cache cleared")


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for caching responses"""
    
    def __init__(self, app, cache: IntelligentCache):
        super().__init__(app)
        self.cache = cache
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Pass-through for now (caching logic can be added)
        response = await call_next(request)
        return response


_intelligent_cache: Optional[IntelligentCache] = None


async def init_intelligent_cache(
    default_ttl: int = 3600
) -> IntelligentCache:
    """Initialize the intelligent cache"""
    global _intelligent_cache
    _intelligent_cache = IntelligentCache(default_ttl)
    return _intelligent_cache


def get_intelligent_cache() -> Optional[IntelligentCache]:
    """Get the current cache instance"""
    return _intelligent_cache
