"""
🔧 Redis Connection Manager
============================

Manages Redis connections with fallback to FakeRedis for development.
"""

import logging
import asyncio
from typing import Optional, Union

try:
    import redis.asyncio as redis
    import fakeredis.aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)

class RedisManager:
    """Redis connection manager with fallback support"""
    
    def __init__(self):
        self.connection: Optional[Union[redis.Redis, fakeredis.aioredis.FakeRedis]] = None
        self.is_fake = False
    
    async def connect(self, redis_url: str = "redis://localhost:6379/0", password: Optional[str] = None) -> bool:
        """Connect to Redis with fallback to FakeRedis"""
        if not HAS_REDIS:
            logger.warning("Redis not available - using in-memory fallback")
            return False
        
        try:
            # Try real Redis first
            self.connection = redis.from_url(redis_url, password=password, decode_responses=True)
            await self.connection.ping()
            logger.info("✅ Connected to Redis server")
            self.is_fake = False
            return True
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to FakeRedis")
            try:
                # Fallback to FakeRedis
                self.connection = fakeredis.aioredis.FakeRedis(decode_responses=True)
                await self.connection.ping()
                logger.info("✅ Using FakeRedis for development")
                self.is_fake = True
                return True
            except Exception as fake_error:
                logger.error(f"FakeRedis also failed: {fake_error}")
                return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if not self.connection:
            return None
        try:
            return await self.connection.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis"""
        if not self.connection:
            return False
        try:
            await self.connection.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.connection:
            return False
        try:
            result = await self.connection.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.connection:
            return False
        try:
            result = await self.connection.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.connection:
            try:
                await self.connection.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None

async def get_redis_manager() -> RedisManager:
    """Get global Redis manager instance"""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
        # Try to connect with environment settings
        from ..config import settings
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        redis_password = getattr(settings, 'REDIS_PASSWORD', None)
        await _redis_manager.connect(redis_url, redis_password)
    return _redis_manager

def create_redis_manager() -> RedisManager:
    """Create new Redis manager instance"""
    return RedisManager()

# Export classes and functions
__all__ = [
    'RedisManager',
    'get_redis_manager',
    'create_redis_manager'
]
