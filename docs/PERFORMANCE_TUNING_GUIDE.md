# 🕉️ DharmaMind Production Performance Tuning Guide

## Table of Contents
1. [Database Optimization](#database-optimization)
2. [Redis Caching Strategy](#redis-caching-strategy)
3. [Application Performance](#application-performance)
4. [System Resource Optimization](#system-resource-optimization)
5. [AI Model Optimization](#ai-model-optimization)
6. [Network and Load Balancing](#network-and-load-balancing)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Performance Testing](#performance-testing)

---

## Database Optimization

### PostgreSQL Configuration

#### Memory Settings
```postgresql
# postgresql.conf optimizations for production

# Memory
shared_buffers = 256MB                    # 25% of total RAM
effective_cache_size = 1GB               # 75% of total RAM
work_mem = 4MB                           # Per connection working memory
maintenance_work_mem = 64MB              # For maintenance operations
max_wal_size = 1GB                       # WAL size before checkpoint
min_wal_size = 80MB                      # Minimum WAL size

# Connection settings
max_connections = 200                     # Adjust based on concurrent users
shared_preload_libraries = 'pg_stat_statements'

# Checkpoints
checkpoint_completion_target = 0.7        # Spread checkpoint I/O
checkpoint_timeout = 10min                # Maximum time between checkpoints
wal_buffers = 16MB                       # WAL buffer size

# Query planner
default_statistics_target = 100          # Histogram buckets for ANALYZE
random_page_cost = 1.1                   # SSD optimization
effective_io_concurrency = 200           # Concurrent I/O operations
```

#### Index Optimization
```sql
-- Monitor index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_tup_read / NULLIF(idx_tup_fetch, 0) as ratio
FROM pg_stat_user_indexes 
ORDER BY idx_tup_read DESC;

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes 
WHERE idx_scan = 0 
    AND indexrelid NOT IN (
        SELECT indexrelid 
        FROM pg_index 
        WHERE indisunique OR indisprimary
    )
ORDER BY pg_relation_size(indexrelid) DESC;

-- Critical indexes for DharmaMind
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_messages_user_created 
    ON chat_messages(user_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_application_logs_timestamp_level 
    ON application_logs(timestamp DESC, level);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dharmic_insights_wisdom_level 
    ON dharmic_insights(wisdom_level DESC, created_at DESC);

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_users 
    ON users(last_active_at) 
    WHERE is_active = true;

-- Expression indexes for search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_messages_search 
    ON chat_messages USING GIN(to_tsvector('english', message_content));
```

#### Query Optimization
```sql
-- Enable query statistics
SELECT pg_stat_statements_reset();

-- Monitor slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
WHERE calls > 10 
ORDER BY mean_time DESC 
LIMIT 20;

-- Vacuum and analyze automation
-- Add to crontab: 0 2 * * * psql -d dharmamind -c "VACUUM ANALYZE;"

-- Connection pooling with PgBouncer
-- pgbouncer.ini configuration:
[databases]
dharmamind = host=postgres_primary port=5432 dbname=dharmamind
dharmamind_replica = host=postgres_replica port=5432 dbname=dharmamind

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
server_idle_timeout = 600
```

---

## Redis Caching Strategy

### Redis Configuration
```redis
# redis.conf optimizations

# Memory management
maxmemory 1gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes

# Network
tcp-keepalive 300
timeout 0
tcp-backlog 511

# Advanced features
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
```

### Caching Patterns

#### Multi-Level Caching
```python
# Example implementation in cache_service.py

class CacheHierarchy:
    """Multi-level caching with intelligent eviction"""
    
    def __init__(self):
        self.l1_cache = {}  # In-memory (fastest)
        self.l2_cache = redis_client  # Redis (fast)
        self.l3_cache = database  # Database (persistent)
    
    async def get_dharmic_response(self, query_hash: str):
        # L1: Memory cache (sub-millisecond)
        if query_hash in self.l1_cache:
            await self.track_cache_hit("L1", query_hash)
            return self.l1_cache[query_hash]
        
        # L2: Redis cache (1-5ms)
        cached = await self.l2_cache.get(f"dharmic:{query_hash}")
        if cached:
            self.l1_cache[query_hash] = cached  # Promote to L1
            await self.track_cache_hit("L2", query_hash)
            return cached
        
        # L3: Database (10-50ms)
        result = await self.l3_cache.get_cached_response(query_hash)
        if result:
            await self.l2_cache.setex(f"dharmic:{query_hash}", 3600, result)
            self.l1_cache[query_hash] = result
            await self.track_cache_hit("L3", query_hash)
            return result
        
        # Cache miss - need to generate new response
        await self.track_cache_miss(query_hash)
        return None
```

#### Cache Warming Strategies
```python
# Predictive cache warming based on user patterns
async def warm_cache_for_user(user_id: str):
    """Pre-load likely queries for user"""
    
    # Get user's common query patterns
    common_queries = await get_user_query_patterns(user_id)
    
    # Pre-compute responses for high-probability queries
    for query_pattern in common_queries:
        if query_pattern.probability > 0.7:
            await pre_compute_dharmic_response(query_pattern.template)
    
    # Warm wisdom content cache
    user_wisdom_preferences = await get_user_wisdom_preferences(user_id)
    await pre_load_wisdom_content(user_wisdom_preferences)

# Time-based cache warming
async def scheduled_cache_warming():
    """Warm cache during low-traffic periods"""
    
    # Popular dharmic content
    popular_topics = await get_trending_dharmic_topics()
    for topic in popular_topics:
        await pre_generate_topic_responses(topic)
    
    # Seasonal wisdom content
    current_season = get_dharmic_season()
    await pre_load_seasonal_wisdom(current_season)
```

---

## Application Performance

### FastAPI Optimization

#### Production Settings
```python
# main.py optimizations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

app = FastAPI(
    title="DharmaMind API",
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
    openapi_url="/openapi.json" if settings.ENV != "production" else None,
)

# Performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Optimized CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,
    max_age=86400,  # Cache preflight for 24 hours
)

# Connection pooling
@app.on_event("startup")
async def startup_event():
    await database_manager.initialize_pool(
        min_connections=10,
        max_connections=100,
        max_idle_time=300
    )
    await redis_manager.initialize_pool(
        min_connections=5,
        max_connections=50
    )

# Gunicorn configuration (gunicorn.conf.py)
bind = "0.0.0.0:8000"
workers = 4  # CPU cores * 2
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 300
keepalive = 5
preload_app = True

# Worker memory monitoring
max_worker_memory = 512 * 1024 * 1024  # 512MB
worker_tmp_dir = "/dev/shm"  # Use RAM for temp files
```

#### Response Time Optimization
```python
# Async processing optimization
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DharmicResponseOptimizer:
    """Optimize dharmic response generation"""
    
    def __init__(self):
        self.cpu_executor = ThreadPoolExecutor(max_workers=4)
        self.io_executor = ThreadPoolExecutor(max_workers=8)
    
    async def generate_optimized_response(self, query: str, user_context: dict):
        """Parallel processing for faster responses"""
        
        # Parallel tasks
        tasks = [
            self.get_cached_response(query),
            self.analyze_user_context(user_context),
            self.get_relevant_wisdom(query),
            self.check_dharmic_principles(query)
        ]
        
        # Execute in parallel
        cached_response, user_analysis, wisdom_content, principles = \
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Fast path: return cached response if available and valid
        if cached_response and self.is_response_valid(cached_response, user_analysis):
            return cached_response
        
        # Generate new response with parallel AI calls
        ai_tasks = []
        
        # Primary AI response
        ai_tasks.append(self.generate_primary_response(query, user_analysis))
        
        # Parallel wisdom enrichment
        if wisdom_content:
            ai_tasks.append(self.enrich_with_wisdom(query, wisdom_content))
        
        # Parallel emotional intelligence
        ai_tasks.append(self.add_emotional_intelligence(query, user_analysis))
        
        # Wait for all AI tasks
        ai_results = await asyncio.gather(*ai_tasks)
        
        # Combine results
        final_response = self.combine_ai_responses(ai_results)
        
        # Cache asynchronously (don't block response)
        asyncio.create_task(self.cache_response(query, final_response))
        
        return final_response
```

### Database Query Optimization
```python
# Efficient database queries
class OptimizedQueries:
    """Database query optimizations"""
    
    async def get_user_conversation_history(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0
    ):
        """Optimized conversation history with pagination"""
        
        query = """
            SELECT 
                cm.id,
                cm.message_content,
                cm.response_content,
                cm.created_at,
                cm.wisdom_score,
                u.username
            FROM chat_messages cm
            JOIN users u ON cm.user_id = u.id
            WHERE cm.user_id = $1
                AND cm.created_at >= NOW() - INTERVAL '30 days'  -- Recent only
            ORDER BY cm.created_at DESC
            LIMIT $2 OFFSET $3
        """
        
        # Use prepared statement for better performance
        return await self.db.fetch_all(query, user_id, limit, offset)
    
    async def get_dharmic_insights_batch(self, insight_ids: List[str]):
        """Batch loading to reduce database roundtrips"""
        
        query = """
            SELECT 
                id,
                insight_type,
                wisdom_level,
                compassion_score,
                content,
                created_at
            FROM dharmic_insights
            WHERE id = ANY($1)
            ORDER BY wisdom_level DESC
        """
        
        return await self.db.fetch_all(query, insight_ids)
    
    async def get_trending_wisdom_topics(self, hours: int = 24):
        """Optimized trending analysis with materialized view"""
        
        # Use materialized view for complex analytics
        query = """
            SELECT 
                topic,
                query_count,
                avg_wisdom_score,
                unique_users
            FROM trending_wisdom_mv
            WHERE updated_at >= NOW() - INTERVAL '%s hours'
            ORDER BY query_count DESC, avg_wisdom_score DESC
            LIMIT 20
        """ % hours
        
        return await self.db.fetch_all(query)
```

---

## System Resource Optimization

### CPU Optimization
```bash
# CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# CPU affinity for containers
# docker-compose.prod.yml additions:
services:
  dharmamind_app:
    cpuset: "0,1"  # Bind to specific CPU cores
    cpu_rt_runtime: 950000  # Real-time scheduling
    
  postgres_primary:
    cpuset: "2,3"  # Dedicated cores for database
    
  redis_master:
    cpuset: "1"    # Single core for Redis
```

### Memory Optimization
```bash
# Kernel memory settings
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf
echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf

# Huge pages for database
echo 'vm.nr_hugepages=1024' >> /etc/sysctl.conf

# Apply settings
sysctl -p
```

### Storage Optimization
```bash
# SSD optimizations
echo noop > /sys/block/sda/queue/scheduler
echo 4096 > /sys/block/sda/queue/read_ahead_kb

# Mount options for database storage
# /etc/fstab:
/dev/sdb1 /var/lib/postgresql ext4 noatime,nodiratime,nobarrier 0 2
/dev/sdc1 /var/lib/redis ext4 noatime,nodiratime 0 2

# Docker volume optimization
docker volume create --driver local \
    --opt type=ext4 \
    --opt device=/dev/sdb1 \
    --opt o=noatime,nodiratime \
    postgres_data_optimized
```

---

## AI Model Optimization

### Model Loading and Caching
```python
# Optimized AI model management
class AIModelOptimizer:
    """Optimize AI model loading and inference"""
    
    def __init__(self):
        self.model_cache = {}
        self.model_lock = asyncio.Lock()
        self.inference_queue = asyncio.Queue(maxsize=100)
    
    async def load_model_optimized(self, model_name: str):
        """Load model with memory optimization"""
        
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        async with self.model_lock:
            # Double-check after acquiring lock
            if model_name in self.model_cache:
                return self.model_cache[model_name]
            
            # Load with CPU optimization
            import torch
            torch.set_num_threads(2)  # Limit CPU threads
            
            # Load model with half precision for inference
            model = await self.load_model(model_name)
            if hasattr(model, 'half'):
                model = model.half()  # Reduce memory usage
            
            self.model_cache[model_name] = model
            return model
    
    async def batch_inference(self, requests: List[dict]):
        """Batch multiple requests for efficiency"""
        
        if len(requests) == 1:
            return await self.single_inference(requests[0])
        
        # Group requests by model type
        grouped_requests = {}
        for req in requests:
            model_type = req.get('model_type', 'default')
            if model_type not in grouped_requests:
                grouped_requests[model_type] = []
            grouped_requests[model_type].append(req)
        
        # Process each group in parallel
        tasks = []
        for model_type, group in grouped_requests.items():
            task = self.process_batch_group(model_type, group)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Flatten and return results in original order
        return self.flatten_batch_results(results, requests)
```

### Embedding Optimization
```python
# Efficient embedding computation and storage
class EmbeddingOptimizer:
    """Optimize vector embeddings for dharmic content"""
    
    def __init__(self):
        self.embedding_cache = LRUCache(maxsize=10000)
        self.batch_size = 32
        self.embedding_model = None
    
    async def get_optimized_embeddings(self, texts: List[str]):
        """Batch embedding computation with caching"""
        
        # Check cache first
        cached_embeddings = {}
        uncached_texts = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[text_hash]
            else:
                uncached_texts.append((i, text, text_hash))
        
        # Compute embeddings for uncached texts in batches
        if uncached_texts:
            embeddings = await self.compute_batch_embeddings(
                [text for _, text, _ in uncached_texts]
            )
            
            # Cache new embeddings
            for (i, text, text_hash), embedding in zip(uncached_texts, embeddings):
                self.embedding_cache[text_hash] = embedding
                cached_embeddings[i] = embedding
        
        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]
    
    async def optimize_vector_search(self, query_embedding, top_k=10):
        """Optimized vector similarity search"""
        
        # Use approximate nearest neighbor for speed
        import faiss
        
        # Build FAISS index if not exists
        if not hasattr(self, 'faiss_index'):
            await self.build_faiss_index()
        
        # Search with IVF (Inverted File) for speed
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), 
            top_k * 2  # Get more results for reranking
        )
        
        # Rerank with more sophisticated similarity
        reranked_results = await self.dharmic_rerank(
            query_embedding, 
            indices[0][:top_k]
        )
        
        return reranked_results
```

---

## Network and Load Balancing

### Nginx Configuration
```nginx
# nginx.conf optimizations
worker_processes auto;
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # Basic optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    client_max_body_size 10M;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Caching
    open_file_cache max=10000 inactive=60s;
    open_file_cache_valid 30s;
    open_file_cache_min_uses 2;
    open_file_cache_errors on;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=chat:10m rate=5r/s;
    
    # Upstream configuration
    upstream dharmamind_backend {
        least_conn;
        server dharmamind_app_primary:8000 weight=3 max_fails=3 fail_timeout=30s;
        server dharmamind_app_replica:8000 weight=2 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    # Main server block
    server {
        listen 80;
        listen 443 ssl http2;
        server_name dharmamind.com;
        
        # SSL configuration
        ssl_certificate /etc/ssl/certs/dharmamind.crt;
        ssl_certificate_key /etc/ssl/private/dharmamind.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://dharmamind_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
        }
        
        # Chat endpoints with stricter rate limiting
        location /api/chat/ {
            limit_req zone=chat burst=10 nodelay;
            proxy_pass http://dharmamind_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_connect_timeout 60s;
            proxy_read_timeout 600s;  # Longer timeout for AI responses
        }
        
        # Static file caching
        location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            add_header Vary "Accept-Encoding";
        }
        
        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://dharmamind_backend/health;
            proxy_connect_timeout 1s;
            proxy_read_timeout 3s;
        }
    }
}
```

### Load Balancing Strategy
```yaml
# docker-compose.prod.yml - Advanced load balancing
version: '3.8'

services:
  # HAProxy for advanced load balancing
  haproxy:
    image: haproxy:2.4-alpine
    container_name: dharmamind_haproxy
    ports:
      - "80:80"
      - "443:443"
      - "8404:8404"  # Stats page
    volumes:
      - ./haproxy/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
      - ssl_certs:/etc/ssl/certs:ro
    depends_on:
      - dharmamind_app
      - dharmamind_app_replica
    networks:
      - dharmamind_network

# haproxy.cfg
global
    daemon
    stats socket /var/run/haproxy.sock mode 600 level admin
    stats timeout 2m

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option redispatch
    retries 3

# Statistics page
stats uri /stats
stats refresh 30s
stats realm HAProxy\ Statistics
stats auth admin:dharmapass

# Frontend
frontend dharmamind_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/dharmamind.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    # Route to backend
    default_backend dharmamind_backend

# Backend with health checks
backend dharmamind_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server app1 dharmamind_app:8000 check inter 30s rise 2 fall 3
    server app2 dharmamind_app_replica:8000 check inter 30s rise 2 fall 3 backup
```

---

## Monitoring and Alerting

### Performance Metrics Dashboard
```yaml
# prometheus.yml - Comprehensive metrics collection
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  # Application metrics
  - job_name: 'dharmamind-app'
    static_configs:
      - targets: ['dharmamind_app:8000', 'dharmamind_app_replica:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Database metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']
    
  # Redis metrics  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis_exporter:9121']
      
  # System metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node_exporter:9100']
      
  # Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

alertmanager:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Alert Rules
```yaml
# alert_rules.yml
groups:
  - name: dharmamind_alerts
    rules:
      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
      
      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      # Database connection issues
      - alert: DatabaseConnections
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connection usage"
          description: "Database connections at {{ $value | humanizePercentage }}"
      
      # Redis memory usage
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis memory usage critical"
          description: "Redis memory usage at {{ $value | humanizePercentage }}"
      
      # AI model inference time
      - alert: SlowAIInference
        expr: histogram_quantile(0.95, rate(ai_inference_duration_seconds_bucket[5m])) > 10
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "AI inference is slow"
          description: "95th percentile AI inference time is {{ $value }}s"
```

---

## Performance Testing

### Load Testing Script
```python
# load_test.py - Comprehensive load testing
import asyncio
import aiohttp
import time
import json
from dataclasses import dataclass
from typing import List
import random

@dataclass
class TestConfig:
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 100
    test_duration: int = 300  # 5 minutes
    ramp_up_time: int = 60    # 1 minute
    dharmic_queries: List[str] = None

class DharmaMindLoadTest:
    """Comprehensive load testing for DharmaMind"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': [],
            'dharmic_quality_scores': []
        }
        
        # Sample dharmic queries for testing
        self.dharmic_queries = config.dharmic_queries or [
            "How can I find inner peace?",
            "What is the meaning of compassion?",
            "How do I overcome suffering?",
            "What is mindfulness?",
            "How can I be more present?",
            "What is the path to enlightenment?",
            "How do I practice loving-kindness?",
            "What is non-attachment?",
            "How do I find balance in life?",
            "What is the nature of impermanence?"
        ]
    
    async def single_user_session(self, session: aiohttp.ClientSession, user_id: int):
        """Simulate single user behavior"""
        
        user_requests = 0
        session_start = time.time()
        
        while time.time() - session_start < self.config.test_duration:
            try:
                # Random delay between requests (1-10 seconds)
                await asyncio.sleep(random.uniform(1, 10))
                
                # Select random dharmic query
                query = random.choice(self.dharmic_queries)
                
                # Make request
                start_time = time.time()
                async with session.post(
                    f"{self.config.base_url}/api/chat/dharmic",
                    json={
                        "query": query,
                        "user_id": f"test_user_{user_id}",
                        "session_id": f"load_test_session_{user_id}"
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_time = time.time() - start_time
                    response_data = await response.json()
                    
                    # Record results
                    self.results['total_requests'] += 1
                    user_requests += 1
                    
                    if response.status == 200:
                        self.results['successful_requests'] += 1
                        self.results['response_times'].append(response_time)
                        
                        # Analyze dharmic quality
                        if 'wisdom_score' in response_data:
                            self.results['dharmic_quality_scores'].append(
                                response_data['wisdom_score']
                            )
                    else:
                        self.results['failed_requests'] += 1
                        self.results['errors'].append({
                            'status': response.status,
                            'user_id': user_id,
                            'query': query,
                            'response_time': response_time
                        })
                        
            except Exception as e:
                self.results['failed_requests'] += 1
                self.results['errors'].append({
                    'error': str(e),
                    'user_id': user_id,
                    'type': 'exception'
                })
        
        print(f"User {user_id} completed {user_requests} requests")
    
    async def run_load_test(self):
        """Execute load test with gradual ramp-up"""
        
        print(f"Starting load test with {self.config.concurrent_users} users")
        print(f"Test duration: {self.config.test_duration}s")
        print(f"Ramp-up time: {self.config.ramp_up_time}s")
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Gradual ramp-up
            tasks = []
            users_per_second = self.config.concurrent_users / self.config.ramp_up_time
            
            for user_id in range(self.config.concurrent_users):
                # Stagger user start times
                delay = user_id / users_per_second
                task = asyncio.create_task(
                    self.delayed_user_session(session, user_id, delay)
                )
                tasks.append(task)
            
            # Wait for all users to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def delayed_user_session(self, session, user_id, delay):
        """Start user session after delay"""
        await asyncio.sleep(delay)
        await self.single_user_session(session, user_id)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        
        if not self.results['response_times']:
            print("No successful responses to analyze")
            return
        
        response_times = sorted(self.results['response_times'])
        total_requests = self.results['total_requests']
        successful_requests = self.results['successful_requests']
        
        # Calculate percentiles
        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f == len(data) - 1:
                return data[f]
            return data[f] * (1 - c) + data[f + 1] * c
        
        report = f"""
🕉️ DharmaMind Load Test Report
================================

Test Configuration:
- Concurrent Users: {self.config.concurrent_users}
- Test Duration: {self.config.test_duration}s
- Base URL: {self.config.base_url}

Request Statistics:
- Total Requests: {total_requests}
- Successful Requests: {successful_requests}
- Failed Requests: {self.results['failed_requests']}
- Success Rate: {(successful_requests/total_requests)*100:.2f}%

Response Time Analysis:
- Average: {sum(response_times)/len(response_times):.3f}s
- Median (50th percentile): {percentile(response_times, 50):.3f}s
- 90th percentile: {percentile(response_times, 90):.3f}s
- 95th percentile: {percentile(response_times, 95):.3f}s
- 99th percentile: {percentile(response_times, 99):.3f}s
- Min: {min(response_times):.3f}s
- Max: {max(response_times):.3f}s

Throughput:
- Requests per second: {total_requests/self.config.test_duration:.2f}
- Successful RPS: {successful_requests/self.config.test_duration:.2f}
"""
        
        # Dharmic quality analysis
        if self.results['dharmic_quality_scores']:
            quality_scores = self.results['dharmic_quality_scores']
            report += f"""
Dharmic Quality Analysis:
- Average Wisdom Score: {sum(quality_scores)/len(quality_scores):.3f}
- Min Wisdom Score: {min(quality_scores):.3f}
- Max Wisdom Score: {max(quality_scores):.3f}
- Responses with High Wisdom (>0.8): {len([s for s in quality_scores if s > 0.8])}
"""
        
        # Error analysis
        if self.results['errors']:
            report += f"""
Error Analysis:
- Total Errors: {len(self.results['errors'])}
- Error Types: {len(set(e.get('type', 'http_error') for e in self.results['errors']))}
"""
        
        print(report)
        
        # Save detailed report
        with open(f"load_test_report_{int(time.time())}.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

# Usage
async def main():
    config = TestConfig(
        concurrent_users=50,
        test_duration=300,
        ramp_up_time=30
    )
    
    test = DharmaMindLoadTest(config)
    await test.run_load_test()
    test.generate_report()

if __name__ == "__main__":
    asyncio.run(main())
```

### Benchmarking Script
```bash
#!/bin/bash
# benchmark_production.sh - Production performance benchmarking

echo "🕉️ DharmaMind Production Benchmarking"
echo "====================================="

# API Performance Test
echo "📊 Testing API Performance..."
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p post_data.json http://localhost:8000/api/chat/dharmic

# Database Performance Test  
echo "🗄️ Testing Database Performance..."
pgbench -h localhost -p 5432 -U dharmamind -d dharmamind -c 10 -j 2 -T 60

# Redis Performance Test
echo "📈 Testing Redis Performance..."
redis-benchmark -h localhost -p 6379 -a $REDIS_PASSWORD -t set,get -n 10000 -c 10

# System Resource Monitoring
echo "💻 System Resource Usage:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.2f%%", $3/$2 * 100.0)}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"

# Container Performance
echo "🐳 Container Performance:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo "✅ Benchmarking completed!"
```

---

This comprehensive performance tuning guide provides production-ready optimizations for all components of DharmaMind. Implement these optimizations gradually and monitor their impact on your specific workload and infrastructure.

**Key Performance Targets:**
- API Response Time: < 200ms (95th percentile)
- Database Query Time: < 50ms (average)
- Cache Hit Rate: > 85%
- AI Inference Time: < 3s (95th percentile)
- System CPU Usage: < 70% (average)
- Memory Usage: < 80% (average)

Regular performance testing and monitoring will help maintain optimal performance as your user base grows.
