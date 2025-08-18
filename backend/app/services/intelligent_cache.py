"""
⚡ Intelligent API Response Caching System

Advanced caching with compression, invalidation strategies, and performance optimization
for high-throughput API responses.
"""

import asyncio
import time
import gzip
import pickle
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from pydantic import BaseModel
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import zlib

logger = logging.getLogger(__name__)

class CacheStrategy(str, Enum):
    """Cache strategy types"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used  
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

class CompressionType(str, Enum):
    """Compression types"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZMA = "lzma"

@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    size: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: int
    compressed: bool
    compression_type: CompressionType
    content_type: str
    etag: str
    tags: List[str] = None

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_size: int = 0
    entries_count: int = 0
    hit_rate: float = 0.0
    average_response_time: float = 0.0
    compression_ratio: float = 0.0
    evictions: int = 0

class CacheKeyGenerator:
    """Intelligent cache key generation"""
    
    @staticmethod
    def generate_key(
        endpoint: str,
        method: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        user_context: Dict[str, Any] = None
    ) -> str:
        """Generate cache key from request components"""
        
        key_components = {
            "endpoint": endpoint,
            "method": method,
            "params": params or {},
            "relevant_headers": CacheKeyGenerator._extract_relevant_headers(headers or {}),
            "user_context": user_context or {}
        }
        
        # Sort for consistency
        key_string = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    @staticmethod
    def _extract_relevant_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Extract only caching-relevant headers"""
        relevant = ["accept", "accept-language", "authorization"]
        return {k.lower(): v for k, v in headers.items() if k.lower() in relevant}
    
    @staticmethod
    def generate_tag_key(tag: str) -> str:
        """Generate tag-based key for cache invalidation"""
        return f"tag:{hashlib.sha256(tag.encode()).hexdigest()}"

class ResponseCompressor:
    """Advanced response compression"""
    
    @staticmethod
    async def compress(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm"""
        
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression_type == CompressionType.ZLIB:
            return zlib.compress(data)
        elif compression_type == CompressionType.LZMA:
            import lzma
            return lzma.compress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    @staticmethod
    async def decompress(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm"""
        
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression_type == CompressionType.LZMA:
            import lzma
            return lzma.decompress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    @staticmethod
    def choose_compression(data_size: int, content_type: str) -> CompressionType:
        """Choose optimal compression based on data characteristics"""
        
        # Don't compress small data or binary content
        if data_size < 1024:  # Less than 1KB
            return CompressionType.NONE
        
        # JSON and text compress well with gzip
        if any(ct in content_type.lower() for ct in ["json", "text", "xml", "html"]):
            if data_size > 10240:  # Greater than 10KB
                return CompressionType.GZIP
            else:
                return CompressionType.ZLIB
        
        # Default to no compression for unknown types
        return CompressionType.NONE

class CacheInvalidator:
    """Cache invalidation strategies"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.tag_prefix = "cache_tag:"
        self.entry_prefix = "cache_entry:"
    
    async def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with specific tag"""
        
        tag_key = f"{self.tag_prefix}{tag}"
        
        # Get all entries with this tag
        entry_keys = self.redis.smembers(tag_key)
        
        if entry_keys:
            # Delete all entries
            pipeline = self.redis.pipeline()
            for entry_key in entry_keys:
                pipeline.delete(f"{self.entry_prefix}{entry_key}")
                
            # Remove tag set
            pipeline.delete(tag_key)
            pipeline.execute()
            
            logger.info(f"Invalidated {len(entry_keys)} cache entries with tag: {tag}")
    
    async def invalidate_by_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        
        matching_keys = []
        for key in self.redis.scan_iter(match=f"{self.entry_prefix}{pattern}"):
            matching_keys.append(key)
        
        if matching_keys:
            self.redis.delete(*matching_keys)
            logger.info(f"Invalidated {len(matching_keys)} cache entries matching pattern: {pattern}")
    
    async def invalidate_expired(self):
        """Remove expired cache entries"""
        
        current_time = datetime.utcnow()
        expired_keys = []
        
        # Scan all cache entries
        for key in self.redis.scan_iter(match=f"{self.entry_prefix}*"):
            try:
                entry_data = self.redis.get(key)
                if entry_data:
                    entry = pickle.loads(entry_data)
                    if isinstance(entry, dict) and "expires_at" in entry:
                        expires_at = datetime.fromisoformat(entry["expires_at"])
                        if expires_at < current_time:
                            expired_keys.append(key)
            except:
                # Invalid entry, mark for deletion
                expired_keys.append(key)
        
        if expired_keys:
            self.redis.delete(*expired_keys)
            logger.info(f"Removed {len(expired_keys)} expired cache entries")

class IntelligentCache:
    """Intelligent API response caching system"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        default_ttl: int = 3600,
        max_size: int = 1000000000,  # 1GB
        compression_threshold: int = 1024,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.compression_threshold = compression_threshold
        self.strategy = strategy
        
        self.cache_prefix = "api_cache:"
        self.meta_prefix = "cache_meta:"
        self.metrics_key = "cache_metrics"
        
        self.compressor = ResponseCompressor()
        self.invalidator = CacheInvalidator(redis_client)
        self.key_generator = CacheKeyGenerator()
        
        # Performance tracking
        self.metrics = CacheMetrics()
        
    async def get(
        self,
        key: str,
        request_context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            cache_key = f"{self.cache_prefix}{key}"
            
            # Get cache entry
            cached_data = self.redis.get(cache_key)
            if not cached_data:
                self.metrics.cache_misses += 1
                return None
            
            # Deserialize entry
            entry = pickle.loads(cached_data)
            
            # Check expiration
            if entry.get("expires_at"):
                expires_at = datetime.fromisoformat(entry["expires_at"])
                if expires_at < datetime.utcnow():
                    await self._delete_entry(cache_key, key)
                    self.metrics.cache_misses += 1
                    return None
            
            # Update access metrics
            await self._update_access_metrics(key, entry)
            
            # Decompress if needed
            response_data = entry["data"]
            if entry.get("compressed", False):
                compression_type = CompressionType(entry.get("compression_type", "none"))
                response_data = await self.compressor.decompress(response_data, compression_type)
            
            # Deserialize response
            if entry.get("content_type", "").startswith("application/json"):
                response = json.loads(response_data.decode())
            else:
                response = response_data
            
            self.metrics.cache_hits += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + 
                 (time.time() - start_time)) / self.metrics.total_requests
            )
            
            logger.debug(f"Cache hit for key: {key[:16]}...")
            return {
                "data": response,
                "metadata": {
                    "cached": True,
                    "cache_key": key,
                    "created_at": entry.get("created_at"),
                    "etag": entry.get("etag")
                }
            }
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.metrics.cache_misses += 1
            return None
    
    async def set(
        self,
        key: str,
        data: Any,
        ttl: int = None,
        tags: List[str] = None,
        content_type: str = "application/json",
        compression: CompressionType = None
    ):
        """Cache response data"""
        
        try:
            # Serialize data
            if content_type.startswith("application/json"):
                serialized_data = json.dumps(data, default=str).encode()
            else:
                serialized_data = data if isinstance(data, bytes) else str(data).encode()
            
            # Choose compression
            if compression is None:
                compression = self.compressor.choose_compression(
                    len(serialized_data), content_type
                )
            
            # Compress if needed
            compressed_data = serialized_data
            if compression != CompressionType.NONE:
                compressed_data = await self.compressor.compress(serialized_data, compression)
            
            # Create cache entry
            cache_ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=cache_ttl)
            
            entry = {
                "data": compressed_data,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat(),
                "content_type": content_type,
                "compressed": compression != CompressionType.NONE,
                "compression_type": compression.value,
                "original_size": len(serialized_data),
                "compressed_size": len(compressed_data),
                "access_count": 0,
                "last_accessed": datetime.utcnow().isoformat(),
                "tags": tags or [],
                "etag": hashlib.md5(serialized_data).hexdigest()
            }
            
            # Store in Redis
            cache_key = f"{self.cache_prefix}{key}"
            self.redis.setex(cache_key, cache_ttl, pickle.dumps(entry))
            
            # Update tag associations
            if tags:
                for tag in tags:
                    tag_key = f"cache_tag:{tag}"
                    self.redis.sadd(tag_key, key)
                    self.redis.expire(tag_key, cache_ttl)
            
            # Update metrics
            self.metrics.entries_count += 1
            self.metrics.total_size += len(compressed_data)
            
            if len(serialized_data) > 0:
                compression_ratio = len(compressed_data) / len(serialized_data)
                self.metrics.compression_ratio = (
                    (self.metrics.compression_ratio * (self.metrics.entries_count - 1) + 
                     compression_ratio) / self.metrics.entries_count
                )
            
            # Check size limits
            await self._enforce_size_limits()
            
            logger.debug(f"Cached response for key: {key[:16]}... (compression: {compression.value})")
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    async def delete(self, key: str):
        """Delete cached entry"""
        
        cache_key = f"{self.cache_prefix}{key}"
        
        # Get entry for metrics update
        cached_data = self.redis.get(cache_key)
        if cached_data:
            try:
                entry = pickle.loads(cached_data)
                self.metrics.total_size -= entry.get("compressed_size", 0)
                self.metrics.entries_count -= 1
            except:
                pass
        
        await self._delete_entry(cache_key, key)
    
    async def _delete_entry(self, cache_key: str, original_key: str):
        """Internal method to delete cache entry"""
        
        # Delete from Redis
        self.redis.delete(cache_key)
        
        # Remove from tag associations
        for tag_key in self.redis.scan_iter(match="cache_tag:*"):
            self.redis.srem(tag_key, original_key)
    
    async def _update_access_metrics(self, key: str, entry: Dict[str, Any]):
        """Update access metrics for cache entry"""
        
        try:
            entry["access_count"] = entry.get("access_count", 0) + 1
            entry["last_accessed"] = datetime.utcnow().isoformat()
            
            cache_key = f"{self.cache_prefix}{key}"
            
            # Get current TTL and preserve it
            ttl = self.redis.ttl(cache_key)
            if ttl > 0:
                self.redis.setex(cache_key, ttl, pickle.dumps(entry))
            
        except Exception as e:
            logger.error(f"Error updating access metrics: {e}")
    
    async def _enforce_size_limits(self):
        """Enforce cache size limits using eviction strategy"""
        
        if self.metrics.total_size <= self.max_size:
            return
        
        # Get all cache entries for eviction analysis
        entries_for_eviction = []
        
        for key in self.redis.scan_iter(match=f"{self.cache_prefix}*"):
            try:
                entry_data = self.redis.get(key)
                if entry_data:
                    entry = pickle.loads(entry_data)
                    original_key = key.decode().replace(self.cache_prefix, "")
                    
                    entries_for_eviction.append({
                        "key": original_key,
                        "cache_key": key,
                        "size": entry.get("compressed_size", 0),
                        "last_accessed": datetime.fromisoformat(entry.get("last_accessed")),
                        "access_count": entry.get("access_count", 0),
                        "created_at": datetime.fromisoformat(entry.get("created_at"))
                    })
            except:
                continue
        
        # Sort by eviction strategy
        if self.strategy == CacheStrategy.LRU:
            entries_for_eviction.sort(key=lambda x: x["last_accessed"])
        elif self.strategy == CacheStrategy.LFU:
            entries_for_eviction.sort(key=lambda x: x["access_count"])
        elif self.strategy == CacheStrategy.TTL:
            entries_for_eviction.sort(key=lambda x: x["created_at"])
        else:  # ADAPTIVE
            # Combine multiple factors
            for entry in entries_for_eviction:
                age_score = (datetime.utcnow() - entry["created_at"]).total_seconds()
                access_score = 1.0 / (entry["access_count"] + 1)
                recency_score = (datetime.utcnow() - entry["last_accessed"]).total_seconds()
                
                entry["eviction_score"] = age_score * 0.3 + access_score * 0.4 + recency_score * 0.3
            
            entries_for_eviction.sort(key=lambda x: x["eviction_score"], reverse=True)
        
        # Evict entries until under size limit
        target_size = self.max_size * 0.8  # Evict to 80% capacity
        current_size = self.metrics.total_size
        
        for entry in entries_for_eviction:
            if current_size <= target_size:
                break
            
            await self._delete_entry(entry["cache_key"], entry["key"])
            current_size -= entry["size"]
            self.metrics.evictions += 1
        
        logger.info(f"Evicted {self.metrics.evictions} cache entries to enforce size limits")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        # Update hit rate
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate = self.metrics.cache_hits / self.metrics.total_requests
        
        return {
            "metrics": asdict(self.metrics),
            "configuration": {
                "default_ttl": self.default_ttl,
                "max_size": self.max_size,
                "compression_threshold": self.compression_threshold,
                "strategy": self.strategy.value
            },
            "redis_info": {
                "memory_usage": self.redis.memory_usage(self.cache_prefix + "*") if hasattr(self.redis, 'memory_usage') else 0,
                "key_count": len(list(self.redis.scan_iter(match=f"{self.cache_prefix}*")))
            }
        }
    
    async def clear_all(self):
        """Clear all cached entries"""
        
        keys_to_delete = list(self.redis.scan_iter(match=f"{self.cache_prefix}*"))
        if keys_to_delete:
            self.redis.delete(*keys_to_delete)
        
        # Clear tag associations
        tag_keys = list(self.redis.scan_iter(match="cache_tag:*"))
        if tag_keys:
            self.redis.delete(*tag_keys)
        
        # Reset metrics
        self.metrics = CacheMetrics()
        
        logger.info("All cache entries cleared")

# Cache middleware for FastAPI
class CacheMiddleware:
    """FastAPI middleware for automatic response caching"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.cacheable_methods = {"GET", "HEAD"}
        self.cacheable_status_codes = {200, 201, 202, 203, 204, 300, 301, 302, 303, 304, 307, 308}
    
    async def __call__(self, request: Request, call_next):
        """Process request through cache middleware"""
        
        # Skip caching for non-cacheable methods
        if request.method not in self.cacheable_methods:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self.cache.key_generator.generate_key(
            endpoint=str(request.url.path),
            method=request.method,
            params=dict(request.query_params),
            headers=dict(request.headers)
        )
        
        # Try to get from cache
        cached_response = await self.cache.get(cache_key)
        if cached_response:
            return JSONResponse(
                content=cached_response["data"],
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Key": cache_key[:16],
                    "ETag": cached_response["metadata"]["etag"]
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache response if cacheable
        if (response.status_code in self.cacheable_status_codes and
            hasattr(response, 'body')):
            
            try:
                # Extract response data
                response_body = response.body
                content_type = response.headers.get("content-type", "application/json")
                
                # Cache the response
                await self.cache.set(
                    key=cache_key,
                    data=response_body,
                    content_type=content_type,
                    tags=[f"endpoint:{request.url.path}", f"method:{request.method}"]
                )
                
                # Add cache headers
                response.headers["X-Cache"] = "MISS"
                response.headers["X-Cache-Key"] = cache_key[:16]
                
            except Exception as e:
                logger.error(f"Error caching response: {e}")
        
        return response

# Global cache instance
intelligent_cache: Optional[IntelligentCache] = None

def get_intelligent_cache() -> IntelligentCache:
    """Get the global intelligent cache instance"""
    if intelligent_cache is None:
        raise RuntimeError("Intelligent cache not initialized")
    return intelligent_cache

def init_intelligent_cache(
    redis_client: redis.Redis,
    **kwargs
) -> IntelligentCache:
    """Initialize the global intelligent cache"""
    global intelligent_cache
    intelligent_cache = IntelligentCache(redis_client, **kwargs)
    return intelligent_cache
