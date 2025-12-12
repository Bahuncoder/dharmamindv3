"""
Cache Service for DharmaMind platform

Provides intelligent caching for improved performance and reduced
external API calls. Supports multiple cache categories and TTL policies.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    import fakeredis

logger = logging.getLogger(__name__)

class CacheCategory(str, Enum):
    """Cache categories for organized storage"""
    USER_DATA = "user_data"
    WISDOM_RESPONSES = "wisdom_responses"
    API_RESPONSES = "api_responses"
    SYSTEM_METRICS = "system_metrics"
    SESSION_DATA = "session_data"
    CONFIGURATION = "configuration"
    DHARMIC_CONTENT = "dharmic_content"
    ANALYTICS = "analytics"
    TEMPORARY = "temporary"

class CacheService:
    """Intelligent cache service with category-based organization"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = 3600  # 1 hour default TTL
        self.category_ttls = {
            CacheCategory.USER_DATA: 7200,  # 2 hours
            CacheCategory.WISDOM_RESPONSES: 86400,  # 24 hours
            CacheCategory.API_RESPONSES: 1800,  # 30 minutes
            CacheCategory.SYSTEM_METRICS: 300,  # 5 minutes
            CacheCategory.SESSION_DATA: 3600,  # 1 hour
            CacheCategory.CONFIGURATION: 43200,  # 12 hours
            CacheCategory.DHARMIC_CONTENT: 86400,  # 24 hours
            CacheCategory.ANALYTICS: 1800,  # 30 minutes
            CacheCategory.TEMPORARY: 300,  # 5 minutes
        }
        
    async def initialize(self) -> bool:
        """Initialize the cache service"""
        try:
            if self.redis_client:
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("Cache service initialized with Redis")
            else:
                logger.info("Cache service initialized with memory cache")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            return False
    
    def _generate_key(self, category: CacheCategory, key: str) -> str:
        """Generate a namespaced cache key"""
        return f"dharmamind:{category.value}:{key}"
    
    def _hash_key(self, data: Any) -> str:
        """Generate a hash key for complex data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get(
        self, 
        category: CacheCategory, 
        key: str,
        default: Any = None
    ) -> Any:
        """Get value from cache"""
        try:
            cache_key = self._generate_key(category, key)
            
            if self.redis_client:
                # Try Redis first
                value = await self.redis_client.get(cache_key)
                if value:
                    return json.loads(value.decode('utf-8'))
            else:
                # Use memory cache
                if cache_key in self.memory_cache:
                    cache_entry = self.memory_cache[cache_key]
                    if cache_entry['expires'] > datetime.now():
                        return cache_entry['value']
                    else:
                        # Remove expired entry
                        del self.memory_cache[cache_key]
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for {cache_key}: {e}")
            return default
    
    async def set(
        self,
        category: CacheCategory,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with TTL"""
        try:
            cache_key = self._generate_key(category, key)
            ttl = ttl or self.category_ttls.get(category, self.default_ttl)
            
            if self.redis_client:
                # Use Redis
                serialized_value = json.dumps(value, default=str)
                await self.redis_client.setex(cache_key, ttl, serialized_value)
            else:
                # Use memory cache
                self.memory_cache[cache_key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=ttl)
                }
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_key}: {e}")
            return False
    
    async def delete(self, category: CacheCategory, key: str) -> bool:
        """Delete value from cache"""
        try:
            cache_key = self._generate_key(category, key)
            
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            else:
                self.memory_cache.pop(cache_key, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for {cache_key}: {e}")
            return False
    
    async def exists(self, category: CacheCategory, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            cache_key = self._generate_key(category, key)
            
            if self.redis_client:
                return bool(await self.redis_client.exists(cache_key))
            else:
                if cache_key in self.memory_cache:
                    if self.memory_cache[cache_key]['expires'] > datetime.now():
                        return True
                    else:
                        del self.memory_cache[cache_key]
                        return False
                return False
                
        except Exception as e:
            logger.error(f"Cache exists error for {cache_key}: {e}")
            return False
    
    async def clear_category(self, category: CacheCategory) -> bool:
        """Clear all keys in a category"""
        try:
            pattern = f"dharmamind:{category.value}:*"
            
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            else:
                # Memory cache
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(f"dharmamind:{category.value}:")]
                for key in keys_to_delete:
                    del self.memory_cache[key]
            
            logger.info(f"Cleared cache category: {category.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache category {category.value}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                "cache_type": "redis" if self.redis_client else "memory",
                "categories": {},
                "total_keys": 0
            }
            
            if self.redis_client:
                # Redis stats
                info = await self.redis_client.info()
                stats["redis_info"] = {
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
                
                # Count keys by category
                for category in CacheCategory:
                    pattern = f"dharmamind:{category.value}:*"
                    keys = await self.redis_client.keys(pattern)
                    stats["categories"][category.value] = len(keys)
                    stats["total_keys"] += len(keys)
            else:
                # Memory cache stats
                now = datetime.now()
                valid_keys = 0
                
                for category in CacheCategory:
                    category_keys = 0
                    for key in self.memory_cache:
                        if key.startswith(f"dharmamind:{category.value}:"):
                            if self.memory_cache[key]['expires'] > now:
                                category_keys += 1
                                valid_keys += 1
                    stats["categories"][category.value] = category_keys
                
                stats["total_keys"] = valid_keys
                stats["memory_cache_size"] = len(self.memory_cache)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache service health"""
        try:
            if self.redis_client:
                # Test Redis connection
                await self.redis_client.ping()
                return {
                    "status": "healthy",
                    "cache_type": "redis",
                    "redis_connected": True
                }
            else:
                return {
                    "status": "healthy",
                    "cache_type": "memory",
                    "memory_entries": len(self.memory_cache)
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Global cache service instance
_cache_service: Optional[CacheService] = None

async def get_cache_service() -> CacheService:
    """Get the global cache service instance"""
    global _cache_service
    
    if _cache_service is None:
        # Try to get Redis client if available
        redis_client = None
        try:
            if HAS_REDIS:
                from ..config import settings
                redis_client = redis.from_url(settings.REDIS_URL)
                await redis_client.ping()
        except Exception:
            logger.warning("Redis not available, using memory cache")
            redis_client = None
        
        _cache_service = CacheService(redis_client)
        await _cache_service.initialize()
    
    return _cache_service

def get_cache_service_sync() -> CacheService:
    """Get cache service instance synchronously (may not be initialized)"""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
    
    return _cache_service

class AdvancedCacheService(CacheService):
    """Advanced cache service with additional features for security middleware"""
    
    def __init__(self, redis_client=None):
        super().__init__(redis_client)
        self.rate_limit_counters: Dict[str, Dict[str, int]] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = {}
    
    async def track_rate_limit(
        self,
        identifier: str,
        window_seconds: int = 60,
        max_requests: int = 100
    ) -> bool:
        """Track rate limiting for an identifier"""
        try:
            current_time = int(datetime.now().timestamp())
            window_key = f"rate_limit:{identifier}:{current_time // window_seconds}"
            
            # Get current count
            current_count = await self.get(CacheCategory.SECURITY, window_key, 0)
            
            if current_count >= max_requests:
                return False  # Rate limit exceeded
            
            # Increment counter
            await self.set(
                CacheCategory.SECURITY,
                window_key,
                current_count + 1,
                ttl=window_seconds
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking rate limit: {e}")
            return True  # Allow on error
    
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        try:
            blocked_key = f"blocked_ip:{ip_address}"
            is_blocked = await self.get(CacheCategory.SECURITY, blocked_key, False)
            return bool(is_blocked)
        except Exception as e:
            logger.error(f"Error checking blocked IP: {e}")
            return False
    
    async def block_ip(self, ip_address: str, duration_seconds: int = 3600) -> bool:
        """Block an IP address"""
        try:
            blocked_key = f"blocked_ip:{ip_address}"
            await self.set(
                CacheCategory.SECURITY,
                blocked_key,
                True,
                ttl=duration_seconds
            )
            logger.warning(f"Blocked IP {ip_address} for {duration_seconds} seconds")
            return True
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
            return False
    
    async def track_suspicious_activity(
        self,
        pattern: str,
        identifier: str,
        threshold: int = 5
    ) -> bool:
        """Track suspicious activity patterns"""
        try:
            pattern_key = f"suspicious:{pattern}:{identifier}"
            current_count = await self.get(CacheCategory.SECURITY, pattern_key, 0)
            
            # Increment counter
            new_count = current_count + 1
            await self.set(
                CacheCategory.SECURITY,
                pattern_key,
                new_count,
                ttl=3600  # 1 hour window
            )
            
            if new_count >= threshold:
                logger.warning(f"Suspicious activity detected: {pattern} for {identifier}")
                return True  # Threshold exceeded
            
            return False
            
        except Exception as e:
            logger.error(f"Error tracking suspicious activity: {e}")
            return False
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-related metrics"""
        try:
            metrics = {
                "blocked_ips_count": len(self.blocked_ips),
                "suspicious_patterns_count": len(self.suspicious_patterns),
                "rate_limit_windows_active": len(self.rate_limit_counters),
                "cache_health": await self.health_check()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return {}

# Global advanced cache service instance
_advanced_cache_service: Optional[AdvancedCacheService] = None

async def get_advanced_cache_service() -> AdvancedCacheService:
    """Get the global advanced cache service instance"""
    global _advanced_cache_service
    
    if _advanced_cache_service is None:
        redis_client = None
        try:
            if HAS_REDIS:
                from ..config import settings
                redis_client = redis.from_url(settings.REDIS_URL)
                await redis_client.ping()
        except Exception:
            logger.warning("Redis not available for advanced cache, using memory cache")
            redis_client = None
        
        _advanced_cache_service = AdvancedCacheService(redis_client)
        await _advanced_cache_service.initialize()
    
    return _advanced_cache_service
