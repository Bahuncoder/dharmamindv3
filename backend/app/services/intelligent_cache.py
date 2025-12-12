"""
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

logger = logging.getLogger(__name__)

class CacheStrategy(str, Enum):
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
    
    async def set(
        self,
        key: str,
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
