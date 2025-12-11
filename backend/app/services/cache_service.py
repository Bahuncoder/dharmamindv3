<<<<<<< HEAD
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
=======
"""
🕉️ DharmaMind Advanced Caching Service

Intelligent caching system for optimal performance and wisdom retention:

Core Features:
- Multi-level caching (Memory, Redis, Disk)
- Intelligent cache warming and eviction
- Dharmic response caching with wisdom scoring
- Vector embedding caching for semantic search
- Session and conversation caching
- Performance monitoring and optimization
- Cache coherency and invalidation strategies

Caching Philosophy:
- Wisdom once gained should be easily accessible
- Compassionate responses deserve to be remembered
- Knowledge should flow efficiently to those in need
- Cache with mindfulness and purpose

May this service accelerate the path to wisdom for all beings 🚀
"""

import asyncio
import logging
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Redis for distributed caching
try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logging.warning("Redis not available - using memory-only caching")

from ..config import settings

logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Cache storage levels"""
    MEMORY = "memory"           # In-memory (fastest)
    REDIS = "redis"            # Redis (distributed)
    DISK = "disk"              # Disk storage (persistent)
    HYBRID = "hybrid"          # Multi-level hybrid


class CacheStrategy(str, Enum):
    """Cache eviction strategies"""
    LRU = "lru"               # Least Recently Used
    LFU = "lfu"               # Least Frequently Used
    TTL = "ttl"               # Time To Live
    WISDOM_SCORE = "wisdom"   # Based on dharmic wisdom score
    ADAPTIVE = "adaptive"     # Adaptive based on usage patterns


class CacheCategory(str, Enum):
    """Cache content categories"""
    RESPONSES = "responses"             # AI responses
    CONVERSATIONS = "conversations"     # Conversation history
    EMBEDDINGS = "embeddings"          # Vector embeddings
    USER_SESSIONS = "sessions"         # User session data
    SYSTEM_CONFIG = "config"           # System configuration
    DHARMIC_WISDOM = "wisdom"          # Dharmic wisdom content
    ANALYTICS = "analytics"            # Analytics data
    TEMP_DATA = "temp"                 # Temporary data


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    category: CacheCategory
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    wisdom_score: float
    size_bytes: int
    tags: List[str]


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_response_time: float
    memory_usage_mb: float
    redis_usage_mb: float
    evictions_count: int
    wisdom_cache_effectiveness: float


class AdvancedCacheService:
    """🚀 Advanced Multi-Level Caching Service with Dharmic Intelligence"""
    
    def __init__(self):
        """Initialize advanced caching service"""
        self.name = "AdvancedCacheService"
        
        # Cache storage layers
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.redis_client = None
        
        # Cache configuration
        self.config = {
            'memory_max_size_mb': 256,
            'memory_max_entries': 10000,
            'redis_max_size_mb': 1024,
            'default_ttl_seconds': 3600,
            'wisdom_cache_ttl': 86400,  # 24 hours for dharmic wisdom
            'temp_cache_ttl': 300,      # 5 minutes for temporary data
            'cleanup_interval': 300     # 5 minutes
        }
        
        # Performance tracking
        self.metrics = CacheMetrics(
            total_requests=0,
            cache_hits=0,
            cache_misses=0,
            hit_rate=0.0,
            avg_response_time=0.0,
            memory_usage_mb=0.0,
            redis_usage_mb=0.0,
            evictions_count=0,
            wisdom_cache_effectiveness=0.0
        )
        
        # Cache strategy settings
        self.category_strategies = {
            CacheCategory.RESPONSES: CacheStrategy.WISDOM_SCORE,
            CacheCategory.CONVERSATIONS: CacheStrategy.LRU,
            CacheCategory.EMBEDDINGS: CacheStrategy.LFU,
            CacheCategory.USER_SESSIONS: CacheStrategy.TTL,
            CacheCategory.SYSTEM_CONFIG: CacheStrategy.TTL,
            CacheCategory.DHARMIC_WISDOM: CacheStrategy.WISDOM_SCORE,
            CacheCategory.ANALYTICS: CacheStrategy.TTL,
            CacheCategory.TEMP_DATA: CacheStrategy.TTL
        }
        
        # Category TTL settings
        self.category_ttls = {
            CacheCategory.RESPONSES: 3600,        # 1 hour
            CacheCategory.CONVERSATIONS: 7200,    # 2 hours
            CacheCategory.EMBEDDINGS: 86400,      # 24 hours
            CacheCategory.USER_SESSIONS: 3600,    # 1 hour
            CacheCategory.SYSTEM_CONFIG: 1800,    # 30 minutes
            CacheCategory.DHARMIC_WISDOM: 86400,  # 24 hours (wisdom is timeless)
            CacheCategory.ANALYTICS: 900,         # 15 minutes
            CacheCategory.TEMP_DATA: 300          # 5 minutes
        }
        
        # Wisdom scoring weights
        self.wisdom_weights = {
            'dharmic_alignment': 0.4,
            'wisdom_content': 0.3,
            'user_benefit': 0.2,
            'timeless_value': 0.1
        }
        
        logger.info("🚀 Advanced Cache Service initialized")
    
    async def initialize(self):
        """Initialize caching service with all backends"""
        try:
            logger.info("Initializing Advanced Cache Service...")
            
            # Initialize Redis connection
            if HAS_REDIS and settings.REDIS_URL:
                await self._init_redis()
                logger.info("✅ Redis cache backend ready")
            
            # Start background cleanup task
            asyncio.create_task(self._background_cleanup())
            
            # Warm up critical caches
            await self._warm_up_caches()
            
            logger.info("🚀 Advanced Cache Service ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                retry_on_timeout=True,
                decode_responses=False  # We'll handle encoding manually
            )
            
            # Test connection
            await self.redis_client.ping()
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    async def _warm_up_caches(self):
        """Warm up critical caches with frequently needed data"""
        try:
            # Pre-cache system configuration
            await self.set(
                "system_config_main",
                {"version": "2.0.0", "dharmic_mode": True},
                category=CacheCategory.SYSTEM_CONFIG,
                ttl=1800
            )
            
            # Pre-cache common dharmic wisdom
            dharmic_wisdom = {
                "core_principles": ["ahimsa", "satya", "asteya", "brahmacharya", "aparigraha"],
                "meditation_basics": "Focus on breath awareness and mindful presence",
                "compassion_practice": "May all beings be happy and free from suffering"
            }
            
            await self.set(
                "dharmic_wisdom_core",
                dharmic_wisdom,
                category=CacheCategory.DHARMIC_WISDOM,
                wisdom_score=0.95,
                tags=["core", "wisdom", "dharma"]
            )
            
            logger.info("Cache warm-up completed")
            
        except Exception as e:
            logger.error(f"Cache warm-up error: {e}")
    
    def _generate_cache_key(self, key: str, category: CacheCategory) -> str:
        """Generate standardized cache key"""
        return f"dharma_cache:{category.value}:{key}"
    
    def _calculate_wisdom_score(self, content: Any, metadata: Dict[str, Any] = None) -> float:
        """Calculate wisdom score for content caching priority"""
        try:
            if metadata is None:
                metadata = {}
            
            # Base score
            score = 0.5
            
            # Dharmic alignment bonus
            dharmic_score = metadata.get('dharmic_alignment', 0.5)
            score += dharmic_score * self.wisdom_weights['dharmic_alignment']
            
            # Wisdom content bonus
            wisdom_score = metadata.get('wisdom_content', 0.5)
            score += wisdom_score * self.wisdom_weights['wisdom_content']
            
            # User benefit factor
            user_benefit = metadata.get('user_benefit', 0.5)
            score += user_benefit * self.wisdom_weights['user_benefit']
            
            # Timeless value (spiritual content is timeless)
            timeless_value = metadata.get('timeless_value', 0.5)
            score += timeless_value * self.wisdom_weights['timeless_value']
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Wisdom score calculation error: {e}")
            return 0.5
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value).encode()
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return b""
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first (more efficient)
            try:
                return json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    async def get(self, key: str, category: CacheCategory = CacheCategory.TEMP_DATA) -> Optional[Any]:
        """Get value from cache with intelligent retrieval"""
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            cache_key = self._generate_cache_key(key, category)
            
            # Try memory cache first (fastest)
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                
                # Check TTL
                if self._is_expired(entry):
                    del self.memory_cache[cache_key]
                else:
                    # Update access metadata
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    self.metrics.cache_hits += 1
                    self._update_response_time(start_time)
                    
                    return entry.value
            
            # Try Redis cache
            if self.redis_client:
                try:
                    data = await self.redis_client.get(cache_key)
                    if data:
                        value = self._deserialize_value(data)
                        
                        # Store in memory cache for faster access
                        await self._store_in_memory(cache_key, value, category)
                        
                        self.metrics.cache_hits += 1
                        self._update_response_time(start_time)
                        
                        return value
                except Exception as e:
                    logger.warning(f"Redis get error: {e}")
            
            # Cache miss
            self.metrics.cache_misses += 1
            self._update_response_time(start_time)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.metrics.cache_misses += 1
            return None
    
    async def set(self, key: str, value: Any, 
                 category: CacheCategory = CacheCategory.TEMP_DATA,
                 ttl: Optional[int] = None,
                 wisdom_score: Optional[float] = None,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set value in cache with intelligent storage"""
        try:
            cache_key = self._generate_cache_key(key, category)
            
            # Determine TTL
            if ttl is None:
                ttl = self.category_ttls.get(category, self.config['default_ttl_seconds'])
            
            # Calculate wisdom score
            if wisdom_score is None:
                wisdom_score = self._calculate_wisdom_score(value, metadata)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                category=category,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl,
                wisdom_score=wisdom_score,
                size_bytes=len(str(value)),
                tags=tags or []
            )
            
            # Store in memory cache
            await self._store_in_memory_with_eviction(entry)
            
            # Store in Redis cache
            if self.redis_client:
                try:
                    serialized = self._serialize_value(value)
                    await self.redis_client.setex(cache_key, ttl, serialized)
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def _store_in_memory(self, cache_key: str, value: Any, category: CacheCategory):
        """Store entry in memory cache"""
        entry = CacheEntry(
            key=cache_key,
            value=value,
            category=category,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            ttl_seconds=self.category_ttls.get(category, 3600),
            wisdom_score=0.5,
            size_bytes=len(str(value)),
            tags=[]
        )
        
        self.memory_cache[cache_key] = entry
    
    async def _store_in_memory_with_eviction(self, entry: CacheEntry):
        """Store in memory cache with intelligent eviction"""
        # Check if we need to evict entries
        if len(self.memory_cache) >= self.config['memory_max_entries']:
            await self._evict_entries()
        
        self.memory_cache[entry.key] = entry
    
    async def _evict_entries(self):
        """Evict entries based on strategy"""
        try:
            if not self.memory_cache:
                return
            
            # Group entries by category and apply appropriate strategy
            categories = {}
            for key, entry in self.memory_cache.items():
                if entry.category not in categories:
                    categories[entry.category] = []
                categories[entry.category].append((key, entry))
            
            entries_to_evict = []
            
            for category, entries in categories.items():
                strategy = self.category_strategies.get(category, CacheStrategy.LRU)
                
                if strategy == CacheStrategy.LRU:
                    # Evict least recently used
                    entries.sort(key=lambda x: x[1].last_accessed)
                    entries_to_evict.extend(entries[:len(entries)//4])  # Evict 25%
                
                elif strategy == CacheStrategy.LFU:
                    # Evict least frequently used
                    entries.sort(key=lambda x: x[1].access_count)
                    entries_to_evict.extend(entries[:len(entries)//4])
                
                elif strategy == CacheStrategy.WISDOM_SCORE:
                    # Evict lowest wisdom score entries
                    entries.sort(key=lambda x: x[1].wisdom_score)
                    entries_to_evict.extend(entries[:len(entries)//4])
                
                elif strategy == CacheStrategy.TTL:
                    # Evict expired entries first
                    expired = [e for e in entries if self._is_expired(e[1])]
                    entries_to_evict.extend(expired)
            
            # Remove selected entries
            for key, _ in entries_to_evict:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    self.metrics.evictions_count += 1
            
            logger.debug(f"Evicted {len(entries_to_evict)} cache entries")
            
        except Exception as e:
            logger.error(f"Cache eviction error: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl_seconds is None:
            return False
        
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds
    
    def _update_response_time(self, start_time: float):
        """Update average response time metric"""
        response_time = time.time() - start_time
        
        # Simple moving average
        if self.metrics.total_requests == 1:
            self.metrics.avg_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.metrics.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.avg_response_time
            )
    
    async def invalidate(self, key: str, category: CacheCategory = CacheCategory.TEMP_DATA) -> bool:
        """Invalidate specific cache entry"""
        try:
            cache_key = self._generate_cache_key(key, category)
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # Remove from Redis cache
            if self.redis_client:
                try:
                    await self.redis_client.delete(cache_key)
                except Exception as e:
                    logger.warning(f"Redis delete error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        try:
            invalidated = 0
            
            # Memory cache
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated += 1
            
            # For Redis, we'd need to implement tag indexing
            # This is a simplified version
            
            return invalidated
            
        except Exception as e:
            logger.error(f"Tag-based invalidation error: {e}")
            return 0
    
    async def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        try:
            # Update hit rate
            if self.metrics.total_requests > 0:
                self.metrics.hit_rate = self.metrics.cache_hits / self.metrics.total_requests
            
            # Update memory usage
            memory_usage = sum(entry.size_bytes for entry in self.memory_cache.values())
            self.metrics.memory_usage_mb = memory_usage / (1024 * 1024)
            
            # Calculate wisdom cache effectiveness
            wisdom_entries = [
                entry for entry in self.memory_cache.values()
                if entry.category == CacheCategory.DHARMIC_WISDOM
            ]
            
            if wisdom_entries:
                wisdom_hits = sum(entry.access_count for entry in wisdom_entries)
                total_hits = sum(entry.access_count for entry in self.memory_cache.values())
                self.metrics.wisdom_cache_effectiveness = wisdom_hits / total_hits if total_hits > 0 else 0
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return self.metrics
    
    async def _background_cleanup(self):
        """Background task for cache maintenance"""
        while True:
            try:
                await asyncio.sleep(self.config['cleanup_interval'])
                
                # Remove expired entries
                expired_keys = []
                for key, entry in self.memory_cache.items():
                    if self._is_expired(entry):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def flush_all(self):
        """Flush all cache data"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear Redis cache
            if self.redis_client:
                await self.redis_client.flushdb()
            
            # Reset metrics
            self.metrics = CacheMetrics(
                total_requests=0,
                cache_hits=0,
                cache_misses=0,
                hit_rate=0.0,
                avg_response_time=0.0,
                memory_usage_mb=0.0,
                redis_usage_mb=0.0,
                evictions_count=0,
                wisdom_cache_effectiveness=0.0
            )
            
            logger.info("All cache data flushed")
            
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
    
    async def cleanup(self):
        """Cleanup cache service resources"""
        try:
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Clear memory cache
            self.memory_cache.clear()
            
            logger.info("Cache service cleanup completed")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")


# Factory function
def create_cache_service() -> AdvancedCacheService:
    """Create cache service instance"""
    return AdvancedCacheService()


# Global instance
_cache_service = None

def get_cache_service() -> AdvancedCacheService:
    """Get global cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = create_cache_service()
    return _cache_service
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
