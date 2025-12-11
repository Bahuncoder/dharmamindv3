"""
<<<<<<< HEAD
Intelligent Cache Service for DharmaMind platform

Advanced caching with AI-powered cache optimization, predictive prefetching,
and spiritual content awareness.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .cache_service import CacheService, CacheCategory, get_cache_service
=======
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
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

logger = logging.getLogger(__name__)

class CacheStrategy(str, Enum):
<<<<<<< HEAD
    """Intelligent caching strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SPIRITUAL_RELEVANCE = "spiritual_relevance"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    TIME_AWARE = "time_aware"

class CachePriority(str, Enum):
    """Cache priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CacheEntry(object):
    """Intelligent cache entry with metadata"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        priority: CachePriority = CachePriority.MEDIUM,
        tags: Optional[List[str]] = None
    ):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.ttl = ttl
        self.priority = priority
        self.tags = tags or []
        self.spiritual_relevance_score = 0.0
        self.predicted_next_access: Optional[datetime] = None
        
    def access(self):
        """Mark entry as accessed"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def get_age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_last_access_seconds(self) -> float:
        """Get seconds since last access"""
        return (datetime.now() - self.last_accessed).total_seconds()

class IntelligentCache:
    """Advanced intelligent caching system"""
    
    def __init__(self, base_cache_service: Optional[CacheService] = None):
        self.base_cache = base_cache_service
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.strategy = CacheStrategy.ADAPTIVE
        self.max_memory_entries = 10000
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.spiritual_keywords = [
            "dharma", "meditation", "enlightenment", "consciousness", "soul",
            "spirit", "wisdom", "compassion", "mindfulness", "karma"
        ]
        
    async def initialize(self) -> bool:
        """Initialize the intelligent cache"""
        try:
            if not self.base_cache:
                self.base_cache = await get_cache_service()
            logger.info("Intelligent Cache initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Intelligent Cache: {e}")
            return False
    
    async def get(
        self,
        key: str,
        category: CacheCategory = CacheCategory.USER_DATA,
        default: Any = None
    ) -> Any:
        """Get value from intelligent cache"""
        try:
            cache_key = self._generate_intelligent_key(category, key)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not entry.is_expired():
                    entry.access()
                    self._record_access_pattern(cache_key)
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[cache_key]
            
            # Check base cache
            if self.base_cache:
                value = await self.base_cache.get(category, key, default)
                if value != default:
                    # Store in memory cache for faster access
                    await self._store_in_memory(cache_key, value, category)
                    self._record_access_pattern(cache_key)
                    return value
            
            return default
            
        except Exception as e:
            logger.error(f"Error getting from intelligent cache: {e}")
            return default
=======
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
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
    
    async def set(
        self,
        key: str,
<<<<<<< HEAD
        value: Any,
        category: CacheCategory = CacheCategory.USER_DATA,
        ttl: Optional[int] = None,
        priority: CachePriority = CachePriority.MEDIUM,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in intelligent cache"""
        try:
            cache_key = self._generate_intelligent_key(category, key)
            
            # Calculate spiritual relevance
            spiritual_score = self._calculate_spiritual_relevance(value)
            
            # Store in memory cache
            entry = CacheEntry(
                key=cache_key,
                value=value,
                ttl=ttl,
                priority=priority,
                tags=tags
            )
            entry.spiritual_relevance_score = spiritual_score
            
            # Apply cache eviction if needed
            await self._apply_eviction_strategy()
            
            self.memory_cache[cache_key] = entry
            
            # Store in base cache
            if self.base_cache:
                await self.base_cache.set(category, key, value, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting in intelligent cache: {e}")
            return False
    
    async def invalidate(
        self,
        key: Optional[str] = None,
        category: Optional[CacheCategory] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Intelligently invalidate cache entries"""
        try:
            invalidated_count = 0
            
            if key and category:
                # Invalidate specific key
                cache_key = self._generate_intelligent_key(category, key)
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    invalidated_count += 1
                
                if self.base_cache:
                    await self.base_cache.delete(category, key)
            
            elif tags:
                # Invalidate by tags
                keys_to_remove = []
                for cache_key, entry in self.memory_cache.items():
                    if any(tag in entry.tags for tag in tags):
                        keys_to_remove.append(cache_key)
                
                for cache_key in keys_to_remove:
                    del self.memory_cache[cache_key]
                    invalidated_count += 1
            
            logger.info(f"Invalidated {invalidated_count} cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    def _generate_intelligent_key(self, category: CacheCategory, key: str) -> str:
        """Generate intelligent cache key with context"""
        context_data = {
            "category": category.value,
            "key": key,
            "timestamp_hour": datetime.now().hour,  # Time-aware caching
        }
        context_str = json.dumps(context_data, sort_keys=True)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
        return f"intelligent:{category.value}:{key}:{context_hash}"
    
    def _calculate_spiritual_relevance(self, value: Any) -> float:
        """Calculate spiritual relevance score for content"""
        if not isinstance(value, (str, dict, list)):
            return 0.0
        
        content_str = str(value).lower()
        relevance_score = 0.0
        
        # Check for spiritual keywords
        for keyword in self.spiritual_keywords:
            if keyword in content_str:
                relevance_score += 0.1
        
        # Check for dharmic concepts
        dharmic_concepts = ["ahimsa", "satya", "asteya", "brahmacharya", "aparigraha"]
        for concept in dharmic_concepts:
            if concept in content_str:
                relevance_score += 0.15
        
        # Check for meditation/practice terms
        practice_terms = ["meditation", "mindfulness", "yoga", "contemplation", "practice"]
        for term in practice_terms:
            if term in content_str:
                relevance_score += 0.12
        
        return min(relevance_score, 1.0)
    
    async def _store_in_memory(
        self,
        cache_key: str,
        value: Any,
        category: CacheCategory,
        priority: CachePriority = CachePriority.MEDIUM
    ):
        """Store entry in memory cache with intelligent metadata"""
        entry = CacheEntry(
            key=cache_key,
            value=value,
            priority=priority
        )
        entry.spiritual_relevance_score = self._calculate_spiritual_relevance(value)
        
        await self._apply_eviction_strategy()
        self.memory_cache[cache_key] = entry
    
    async def _apply_eviction_strategy(self):
        """Apply intelligent cache eviction strategy"""
        if len(self.memory_cache) < self.max_memory_entries:
            return
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still over limit, apply strategy-based eviction
        if len(self.memory_cache) >= self.max_memory_entries:
            entries_to_remove = len(self.memory_cache) - self.max_memory_entries + 100
            
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used
                sorted_entries = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].last_accessed
                )
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                sorted_entries = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].access_count
                )
            elif self.strategy == CacheStrategy.SPIRITUAL_RELEVANCE:
                # Keep spiritually relevant content
                sorted_entries = sorted(
                    self.memory_cache.items(),
                    key=lambda x: (x[1].spiritual_relevance_score, x[1].access_count)
                )
            else:  # ADAPTIVE strategy
                # Combine multiple factors
                sorted_entries = sorted(
                    self.memory_cache.items(),
                    key=lambda x: (
                        x[1].priority.value,
                        x[1].spiritual_relevance_score,
                        x[1].access_count / max(x[1].get_age_seconds(), 1),
                        -x[1].get_last_access_seconds()
                    )
                )
            
            # Remove lowest scoring entries
            for key, _ in sorted_entries[:entries_to_remove]:
                del self.memory_cache[key]
    
    def _record_access_pattern(self, cache_key: str):
        """Record access pattern for predictive caching"""
        if cache_key not in self.access_patterns:
            self.access_patterns[cache_key] = []
        
        self.access_patterns[cache_key].append(datetime.now())
        
        # Keep only recent access history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.access_patterns[cache_key] = [
            access_time for access_time in self.access_patterns[cache_key]
            if access_time > cutoff_time
        ]
    
    async def predict_cache_needs(self) -> List[str]:
        """Predict what might be accessed soon"""
        predictions = []
        
        for cache_key, access_times in self.access_patterns.items():
            if len(access_times) < 2:
                continue
            
            # Simple pattern detection - if accessed regularly, predict next access
            time_diffs = []
            for i in range(1, len(access_times)):
                diff = (access_times[i] - access_times[i-1]).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                avg_interval = sum(time_diffs) / len(time_diffs)
                last_access = access_times[-1]
                predicted_next = last_access + timedelta(seconds=avg_interval)
                
                # If prediction is soon, add to list
                if predicted_next <= datetime.now() + timedelta(minutes=30):
                    predictions.append(cache_key)
        
        return predictions[:50]  # Limit predictions
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get intelligent cache statistics"""
        total_entries = len(self.memory_cache)
        
        # Priority distribution
        priority_dist = {}
        spiritual_scores = []
        access_counts = []
        
        for entry in self.memory_cache.values():
            priority_dist[entry.priority.value] = priority_dist.get(entry.priority.value, 0) + 1
            spiritual_scores.append(entry.spiritual_relevance_score)
            access_counts.append(entry.access_count)
        
        return {
            "total_memory_entries": total_entries,
            "max_memory_entries": self.max_memory_entries,
            "memory_utilization": total_entries / self.max_memory_entries,
            "priority_distribution": priority_dist,
            "average_spiritual_relevance": sum(spiritual_scores) / len(spiritual_scores) if spiritual_scores else 0,
            "average_access_count": sum(access_counts) / len(access_counts) if access_counts else 0,
            "strategy": self.strategy.value,
            "access_patterns_tracked": len(self.access_patterns)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check intelligent cache health"""
        stats = await self.get_cache_stats()
        
        return {
            "status": "healthy" if stats["memory_utilization"] < 0.95 else "degraded",
            "cache": "intelligent",
            "memory_entries": stats["total_memory_entries"],
            "utilization": f"{stats['memory_utilization']:.1%}",
            "strategy": stats["strategy"]
        }

class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for intelligent caching of HTTP responses"""
    
    def __init__(self, app, cache_service: Optional[IntelligentCache] = None):
        super().__init__(app)
        self.cache = cache_service
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip caching for non-GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip caching for authenticated endpoints that might have user-specific data
        if "/api/v1/auth/" in str(request.url) or "/api/v1/user/" in str(request.url):
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"http:{request.method}:{request.url.path}:{str(request.query_params)}"
        
        # Try to get cached response
        if self.cache:
            cached_response = await self.cache.get(
                cache_key,
                CacheCategory.API_RESPONSES
            )
            
            if cached_response:
                logger.debug(f"Cache hit for {cache_key}")
                return Response(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers=cached_response["headers"],
                    media_type=cached_response["media_type"]
                )
        
        # Call the actual endpoint
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200 and self.cache:
            # Read response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Cache the response
            cached_data = {
                "content": response_body.decode(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type
            }
            
            # Determine TTL based on endpoint
            ttl = 300  # 5 minutes default
            if "/api/v1/spiritual/" in str(request.url):
                ttl = 3600  # 1 hour for spiritual content
            elif "/api/v1/health" in str(request.url):
                ttl = 60   # 1 minute for health checks
            
            await self.cache.set(
                cache_key,
                cached_data,
                CacheCategory.API_RESPONSES,
                ttl=ttl,
                priority=CachePriority.MEDIUM
            )
            
            logger.debug(f"Cached response for {cache_key}")
            
            # Recreate response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
        
        return response

# Global intelligent cache instance
_intelligent_cache: Optional[IntelligentCache] = None

async def get_intelligent_cache() -> IntelligentCache:
    """Get the global intelligent cache instance"""
    global _intelligent_cache
    
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache()
        await _intelligent_cache.initialize()
    
    return _intelligent_cache

async def init_intelligent_cache() -> IntelligentCache:
    """Initialize and return the intelligent cache"""
    return await get_intelligent_cache()
=======
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
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
