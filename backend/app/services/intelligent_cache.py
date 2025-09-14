"""
🧠 Intelligent Cache System
===========================

Advanced caching system with AI-powered cache optimization, predictive caching,
and intelligent eviction policies.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from collections import defaultdict
import warnings

try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as e:
    warnings.warn(f"FastAPI not available for middleware: {e}")
    BaseHTTPMiddleware = object

logger = logging.getLogger(__name__)

class CachePolicy(str, Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    INTELLIGENT = "intelligent"  # AI-powered eviction

class CachePriority(str, Enum):
    """Cache entry priorities"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IntelligentCacheEntry:
    """Enhanced cache entry with intelligence metrics"""
    
    def __init__(self, key: str, value: Any, ttl_seconds: Optional[int] = None, priority: CachePriority = CachePriority.MEDIUM):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        self.priority = priority
        
        # Intelligence metrics
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.access_pattern = []
        self.compute_cost = 1.0  # Relative cost to regenerate
        self.popularity_score = 0.0
        
    def access(self):
        """Record cache access"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.access_pattern.append(datetime.now())
        
        # Keep only last 100 accesses for pattern analysis
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]
        
        self._update_popularity_score()
    
    def _update_popularity_score(self):
        """Update popularity score based on access patterns"""
        now = datetime.now()
        
        # Recent access weight (last hour gets more weight)
        recent_accesses = sum(1 for access_time in self.access_pattern 
                            if (now - access_time).total_seconds() < 3600)
        
        # Frequency component
        frequency_score = min(self.access_count / 100.0, 1.0)
        
        # Recency component
        recency_score = max(0, 1.0 - (now - self.last_accessed).total_seconds() / 3600)
        
        # Priority component
        priority_weights = {
            CachePriority.CRITICAL: 1.0,
            CachePriority.HIGH: 0.8,
            CachePriority.MEDIUM: 0.5,
            CachePriority.LOW: 0.2
        }
        priority_score = priority_weights[self.priority]
        
        self.popularity_score = (frequency_score * 0.4 + 
                               recency_score * 0.4 + 
                               priority_score * 0.2 +
                               min(recent_accesses / 10.0, 0.3))
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def should_evict(self, policy: CachePolicy) -> float:
        """Calculate eviction score (higher = more likely to evict)"""
        if policy == CachePolicy.LRU:
            return (datetime.now() - self.last_accessed).total_seconds()
        elif policy == CachePolicy.LFU:
            return 1.0 / max(self.access_count, 1)
        elif policy == CachePolicy.TTL:
            if self.expires_at:
                return max(0, (datetime.now() - self.expires_at).total_seconds())
            return 0
        elif policy == CachePolicy.INTELLIGENT:
            return 1.0 - self.popularity_score
        else:
            return 0

class IntelligentCache:
    """🧠 Intelligent caching system with AI optimization"""
    
    def __init__(self, redis_client=None, max_size: int = 10000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.redis_client = redis_client
        self.max_size = max_size
        
        # Cache storage
        self.cache: Dict[str, IntelligentCacheEntry] = {}
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Configuration
        self.policy = CachePolicy.INTELLIGENT
        self.hit_threshold = 0.8  # Target hit rate
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "predictions_made": 0,
            "predictions_correct": 0
        }
        
        self.logger.info("🧠 Intelligent cache system initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligence tracking"""
        
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            entry.access()
            self.stats["hits"] += 1
            
            # Predictive caching based on access patterns
            await self._trigger_predictive_caching(key)
            
            return entry.value
        else:
            self.stats["misses"] += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        priority: CachePriority = CachePriority.MEDIUM,
        compute_cost: float = 1.0
    ) -> bool:
        """Set value in cache with intelligent management"""
        
        try:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size:
                await self._intelligent_eviction()
            
            # Create cache entry
            entry = IntelligentCacheEntry(key, value, ttl_seconds, priority)
            entry.compute_cost = compute_cost
            
            self.cache[key] = entry
            
            # Update access patterns
            self.access_patterns[key].append(datetime.now())
            if len(self.access_patterns[key]) > 1000:
                self.access_patterns[key] = self.access_patterns[key][-1000:]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_patterns:
                    del self.access_patterns[key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def _intelligent_eviction(self):
        """Intelligent cache eviction based on multiple factors"""
        
        if not self.cache:
            return
        
        # Calculate eviction scores for all entries
        eviction_candidates = []
        
        for key, entry in self.cache.items():
            eviction_score = entry.should_evict(self.policy)
            
            # Adjust score based on compute cost (harder to recreate = lower eviction score)
            adjusted_score = eviction_score / max(entry.compute_cost, 0.1)
            
            eviction_candidates.append((key, adjusted_score))
        
        # Sort by eviction score (highest first)
        eviction_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evict 10% of cache or at least 1 entry
        evict_count = max(1, int(len(self.cache) * 0.1))
        
        for i in range(min(evict_count, len(eviction_candidates))):
            key_to_evict = eviction_candidates[i][0]
            del self.cache[key_to_evict]
            if key_to_evict in self.access_patterns:
                del self.access_patterns[key_to_evict]
            
            self.stats["evictions"] += 1
        
        self.logger.debug(f"Evicted {evict_count} cache entries using intelligent policy")
    
    async def _trigger_predictive_caching(self, accessed_key: str):
        """Trigger predictive caching based on access patterns"""
        
        # Simple pattern: if someone accesses A, they often access B next
        # In a real implementation, this would use ML models
        
        patterns = self.access_patterns.get(accessed_key, [])
        if len(patterns) < 5:
            return
        
        # Example predictive logic (simplified)
        predicted_keys = await self._predict_next_access(accessed_key)
        
        for predicted_key in predicted_keys:
            if predicted_key not in self.cache:
                # In real implementation, this would trigger background cache warming
                self.stats["predictions_made"] += 1
                self.logger.debug(f"Predicted next access: {predicted_key}")
    
    async def _predict_next_access(self, current_key: str) -> List[str]:
        """Predict next likely cache accesses (simplified implementation)"""
        
        # This is a simplified implementation
        # Real implementation would use ML models trained on access patterns
        
        predicted = []
        
        # Pattern-based prediction
        if "user:" in current_key:
            user_id = current_key.split(":")[1]
            predicted.extend([
                f"user:{user_id}:preferences",
                f"user:{user_id}:spiritual_profile",
                f"user:{user_id}:recent_sessions"
            ])
        elif "rishi:" in current_key:
            predicted.extend([
                "spiritual:wisdom:daily",
                "meditation:guidance",
                "emotional:healing"
            ])
        
        return predicted[:3]  # Limit predictions
    
    async def warm_cache(self, keys_and_values: List[Tuple[str, Any]]):
        """Warm cache with predicted entries"""
        
        for key, value in keys_and_values:
            await self.set(key, value, priority=CachePriority.LOW)
        
        self.logger.info(f"Cache warmed with {len(keys_and_values)} entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        prediction_accuracy = 0
        if self.stats["predictions_made"] > 0:
            prediction_accuracy = (self.stats["predictions_correct"] / 
                                 self.stats["predictions_made"] * 100)
        
        # Cache distribution by priority
        priority_distribution = defaultdict(int)
        for entry in self.cache.values():
            priority_distribution[entry.priority.value] += 1
        
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "evictions": self.stats["evictions"],
            "predictions_made": self.stats["predictions_made"],
            "prediction_accuracy_percent": round(prediction_accuracy, 2),
            "priority_distribution": dict(priority_distribution),
            **self.stats
        }
    
    async def optimize_performance(self):
        """Optimize cache performance based on metrics"""
        
        hit_rate = self.get_cache_stats()["hit_rate_percent"]
        
        if hit_rate < self.hit_threshold * 100:
            # Increase cache size if hit rate is low
            if self.max_size < 50000:
                self.max_size = int(self.max_size * 1.2)
                self.logger.info(f"Increased cache size to {self.max_size}")
        
        # Clean expired entries
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
        for key in expired_keys:
            await self.delete(key)
        
        if expired_keys:
            self.logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

class CacheMiddleware(BaseHTTPMiddleware):
    """HTTP cache middleware for automatic response caching"""
    
    def __init__(self, app, cache: IntelligentCache):
        super().__init__(app)
        self.cache = cache
        self.cacheable_methods = {"GET"}
        self.cacheable_paths = {"/api/v1/spiritual", "/api/v1/rishi", "/api/v1/knowledge"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip caching for non-cacheable methods
        if request.method not in self.cacheable_methods:
            return await call_next(request)
        
        # Skip caching for non-cacheable paths
        if not any(request.url.path.startswith(path) for path in self.cacheable_paths):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        cached_response = await self.cache.get(cache_key)
        if cached_response:
            return Response(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"]
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            cached_data = {
                "content": response_body,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
            await self.cache.set(
                cache_key, 
                cached_data, 
                ttl_seconds=300,  # 5 minutes
                priority=CachePriority.MEDIUM
            )
            
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers
            )
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        key_parts = [
            request.method,
            str(request.url),
            str(sorted(request.query_params.items()))
        ]
        
        key_string = "|".join(key_parts)
        return f"http_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

# Global cache instance
_intelligent_cache: Optional[IntelligentCache] = None

def init_intelligent_cache(redis_client=None, max_size: int = 10000) -> IntelligentCache:
    """Initialize intelligent cache"""
    global _intelligent_cache
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache(redis_client, max_size)
    return _intelligent_cache

def get_intelligent_cache() -> IntelligentCache:
    """Get intelligent cache instance"""
    global _intelligent_cache
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache()
    return _intelligent_cache

# Export commonly used classes and functions
__all__ = [
    'IntelligentCache',
    'IntelligentCacheEntry',
    'CachePolicy',
    'CachePriority',
    'CacheMiddleware',
    'init_intelligent_cache',
    'get_intelligent_cache'
]
