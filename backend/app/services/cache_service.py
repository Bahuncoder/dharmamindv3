"""
🕉️ Cache Service
================

Simple cache service for DharmaMind that provides:
- In-memory caching for fast access
- TTL (Time-To-Live) support
- Simple key-value storage
- Cache statistics and monitoring

Features:
- Fast in-memory storage
- Automatic expiry handling
- Cache hit/miss tracking
- Memory usage monitoring
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class CacheCategory(str, Enum):
    """Cache categories for organized caching"""
    USER_SESSION = "user_session"
    RISHI_RESPONSE = "rishi_response"
    EMOTIONAL_ANALYSIS = "emotional_analysis"
    SPIRITUAL_CONTENT = "spiritual_content"
    LLM_RESPONSE = "llm_response"
    SYSTEM_CONFIG = "system_config"
    ANALYTICS = "analytics"
    TEMPORARY = "temporary"

class CacheEntry:
    """Cache entry with TTL support"""
    def __init__(self, value: Any, ttl_seconds: Optional[int] = None):
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

class CacheService:
    """📦 Simple cache service"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.stats["misses"] += 1
                    return None
                else:
                    self.stats["hits"] += 1
                    return entry.value
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            self.cache[key] = CacheEntry(value, ttl_seconds)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.cache.clear()
            return True
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_entries": len(self.cache),
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            **self.stats
        }

class AdvancedCacheService(CacheService):
    """🚀 Advanced cache service with additional features"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def get_or_set(self, key: str, value_func, ttl_seconds: Optional[int] = None) -> Any:
        """Get from cache or set if not exists"""
        value = await self.get(key)
        if value is None:
            value = await value_func() if callable(value_func) else value_func
            await self.set(key, value, ttl_seconds)
        return value
    
    async def increment(self, key: str, amount: int = 1, ttl_seconds: Optional[int] = None) -> int:
        """Increment a numeric value in cache"""
        current = await self.get(key) or 0
        new_value = current + amount
        await self.set(key, new_value, ttl_seconds)
        return new_value
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache and not self.cache[key].is_expired()
    
    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiration time for existing key"""
        if key in self.cache:
            entry = self.cache[key]
            entry.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            return True
        return False

# Global instance
_cache_service: Optional[CacheService] = None
_advanced_cache_service: Optional[AdvancedCacheService] = None

def get_cache_service() -> CacheService:
    """Get global cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service

def get_advanced_cache_service() -> AdvancedCacheService:
    """Get global advanced cache service instance"""
    global _advanced_cache_service
    if _advanced_cache_service is None:
        _advanced_cache_service = AdvancedCacheService()
    return _advanced_cache_service

def create_cache_service() -> CacheService:
    """Create new cache service instance"""
    return CacheService()

def create_advanced_cache_service() -> AdvancedCacheService:
    """Create new advanced cache service instance"""
    return AdvancedCacheService()

# Export commonly used classes and functions
__all__ = [
    'CacheService',
    'AdvancedCacheService',
    'CacheCategory',
    'CacheEntry',
    'get_cache_service',
    'get_advanced_cache_service',
    'create_cache_service',
    'create_advanced_cache_service'
]
