# 🛡️ Advanced Rate Limiting Middleware for DharmaMind
# Production-grade rate limiting with Redis backend and intelligent protection

import time
import json
import logging
from typing import Dict, Optional, Tuple, Set, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import hashlib
from datetime import datetime, timedelta

import redis.asyncio as redis
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
import structlog

from ..config import settings

# Configure structured logging
logger = structlog.get_logger(__name__)

class RateLimitType(str, Enum):
    """Types of rate limiting"""
    GLOBAL = "global"
    IP = "ip"
    USER = "user"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"
    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    AUTHENTICATION = "authentication"

@dataclass
class RateLimitRule:
    """Rate limiting rule definition"""
    limit: int                    # Maximum requests
    window: int                   # Time window in seconds
    burst_limit: Optional[int] = None    # Burst limit (higher temporary limit)
    burst_window: int = 60       # Burst window in seconds
    block_duration: int = 0      # Block duration after limit exceeded (0 = no block)
    message: str = "Rate limit exceeded"
    priority: int = 1            # Rule priority (lower = higher priority)

class RateLimitConfig:
    """Advanced rate limiting configuration"""
    
    # Global rate limits
    GLOBAL_LIMITS = {
        "requests_per_second": RateLimitRule(limit=100, window=1, burst_limit=150),
        "requests_per_minute": RateLimitRule(limit=1000, window=60, burst_limit=1500),
        "requests_per_hour": RateLimitRule(limit=10000, window=3600, burst_limit=15000),
    }
    
    # IP-based rate limits
    IP_LIMITS = {
        "requests_per_minute": RateLimitRule(
            limit=60, window=60, burst_limit=100, 
            message="Too many requests from your IP address"
        ),
        "requests_per_hour": RateLimitRule(
            limit=1000, window=3600, burst_limit=1500,
            message="Hourly request limit exceeded for your IP"
        ),
    }
    
    # User-based rate limits (authenticated users)
    USER_LIMITS = {
        "free_plan": {
            "spiritual_questions_per_day": RateLimitRule(
                limit=50, window=86400, burst_limit=60,
                message="Daily spiritual guidance limit reached. Consider upgrading to Pro."
            ),
            "api_calls_per_hour": RateLimitRule(
                limit=100, window=3600, burst_limit=120,
                message="Hourly API limit reached for free plan"
            ),
        },
        "pro_plan": {
            "spiritual_questions_per_day": RateLimitRule(
                limit=500, window=86400, burst_limit=600,
                message="Daily spiritual guidance limit reached"
            ),
            "api_calls_per_hour": RateLimitRule(
                limit=1000, window=3600, burst_limit=1200,
                message="Hourly API limit reached for Pro plan"
            ),
        },
        "enterprise_plan": {
            "spiritual_questions_per_day": RateLimitRule(
                limit=10000, window=86400, burst_limit=12000,
                message="Daily spiritual guidance limit reached"
            ),
            "api_calls_per_hour": RateLimitRule(
                limit=10000, window=3600, burst_limit=12000,
                message="Hourly API limit reached for Enterprise plan"
            ),
        }
    }
    
    # Endpoint-specific rate limits
    ENDPOINT_LIMITS = {
        # Authentication endpoints (more restrictive)
        "/auth/login": RateLimitRule(
            limit=5, window=300, block_duration=900,  # 5 attempts per 5 min, block for 15 min
            message="Too many login attempts. Please try again later."
        ),
        "/auth/register": RateLimitRule(
            limit=3, window=3600, block_duration=3600,  # 3 registrations per hour
            message="Registration limit exceeded. Please try again later."
        ),
        "/auth/forgot-password": RateLimitRule(
            limit=3, window=3600, 
            message="Password reset limit exceeded. Please try again later."
        ),
        
        # Spiritual guidance endpoints
        "/api/v1/chat/spiritual-guidance": RateLimitRule(
            limit=30, window=3600, burst_limit=40,
            message="Spiritual guidance request limit exceeded. Please take time to reflect."
        ),
        "/api/v1/chat/practice-recommendation": RateLimitRule(
            limit=20, window=3600, burst_limit=25,
            message="Practice recommendation limit exceeded."
        ),
        
        # API endpoints
        "/api/v1/knowledge/search": RateLimitRule(
            limit=100, window=3600, burst_limit=120,
            message="Knowledge search limit exceeded."
        ),
        
        # Heavy computation endpoints
        "/api/v1/analysis/deep": RateLimitRule(
            limit=10, window=3600, burst_limit=15,
            message="Deep analysis limit exceeded. This feature requires intensive computation."
        ),
    }
    
    # Security-focused limits
    SECURITY_LIMITS = {
        "failed_auth_per_ip": RateLimitRule(
            limit=10, window=3600, block_duration=7200,  # Block for 2 hours
            message="Too many failed authentication attempts from your IP"
        ),
        "suspicious_activity_per_ip": RateLimitRule(
            limit=5, window=300, block_duration=1800,  # Block for 30 minutes
            message="Suspicious activity detected from your IP"
        ),
    }

    # Whitelist for unlimited access (internal services, etc.)
    WHITELIST_IPS: Set[str] = {
        "127.0.0.1",
        "::1",
        "localhost"
    }
    
    # Premium API keys with higher limits
    PREMIUM_API_KEYS: Dict[str, str] = {}  # api_key -> plan_type

class AdvancedRateLimiter:
    """Production-grade rate limiter with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.config = RateLimitConfig()
        self.logger = structlog.get_logger(__name__)
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "rate_limited_ips": set(),
            "rate_limited_users": set(),
        }
    
    async def check_rate_limit(
        self, 
        request: Request,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        plan_type: str = "free_plan"
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Comprehensive rate limit checking
        Returns: (is_allowed, limit_info)
        """
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        endpoint = self._normalize_endpoint(request.url.path)
        method = request.method
        
        # Check whitelist first
        if client_ip in self.config.WHITELIST_IPS:
            return True, None
        
        # Check if IP is currently blocked
        if await self._is_blocked(client_ip, "ip"):
            await self._log_rate_limit_event("ip_blocked", {
                "ip": client_ip,
                "endpoint": endpoint,
                "method": method
            })
            return False, {
                "error": "IP address is temporarily blocked",
                "retry_after": await self._get_block_remaining_time(client_ip, "ip")
            }
        
        # Check if user is blocked
        if user_id and await self._is_blocked(user_id, "user"):
            return False, {
                "error": "User is temporarily blocked",
                "retry_after": await self._get_block_remaining_time(user_id, "user")
            }
        
        # Apply rate limiting rules in priority order
        checks = [
            ("endpoint", self._check_endpoint_limits, endpoint),
            ("ip", self._check_ip_limits, client_ip),
            ("user", self._check_user_limits, user_id, plan_type),
            ("global", self._check_global_limits, None),
        ]
        
        for check_type, check_func, *args in checks:
            try:
                is_allowed, limit_info = await check_func(*args)
                if not is_allowed:
                    # Track metrics
                    self.metrics["blocked_requests"] += 1
                    if check_type == "ip":
                        self.metrics["rate_limited_ips"].add(client_ip)
                    elif check_type == "user" and user_id:
                        self.metrics["rate_limited_users"].add(user_id)
                    
                    # Log rate limit violation
                    await self._log_rate_limit_event(f"{check_type}_limit_exceeded", {
                        "ip": client_ip,
                        "user_id": user_id,
                        "endpoint": endpoint,
                        "method": method,
                        "plan_type": plan_type,
                        "limit_info": limit_info
                    })
                    
                    return False, limit_info
                    
            except Exception as e:
                self.logger.error(f"Rate limit check failed for {check_type}", error=str(e))
                # On error, allow request but log the issue
                continue
        
        # All checks passed
        self.metrics["total_requests"] += 1
        return True, None
    
    async def _check_endpoint_limits(self, endpoint: str) -> Tuple[bool, Optional[Dict]]:
        """Check endpoint-specific rate limits"""
        
        if endpoint not in self.config.ENDPOINT_LIMITS:
            return True, None
        
        rule = self.config.ENDPOINT_LIMITS[endpoint]
        key = f"rate_limit:endpoint:{endpoint}"
        
        return await self._apply_rate_limit_rule(key, rule)
    
    async def _check_ip_limits(self, client_ip: str) -> Tuple[bool, Optional[Dict]]:
        """Check IP-based rate limits"""
        
        for limit_name, rule in self.config.IP_LIMITS.items():
            key = f"rate_limit:ip:{client_ip}:{limit_name}"
            is_allowed, limit_info = await self._apply_rate_limit_rule(key, rule)
            
            if not is_allowed:
                return False, limit_info
        
        return True, None
    
    async def _check_user_limits(self, user_id: Optional[str], plan_type: str) -> Tuple[bool, Optional[Dict]]:
        """Check user-based rate limits"""
        
        if not user_id or plan_type not in self.config.USER_LIMITS:
            return True, None
        
        user_limits = self.config.USER_LIMITS[plan_type]
        
        for limit_name, rule in user_limits.items():
            key = f"rate_limit:user:{user_id}:{limit_name}"
            is_allowed, limit_info = await self._apply_rate_limit_rule(key, rule)
            
            if not is_allowed:
                return False, limit_info
        
        return True, None
    
    async def _check_global_limits(self, _) -> Tuple[bool, Optional[Dict]]:
        """Check global rate limits"""
        
        for limit_name, rule in self.config.GLOBAL_LIMITS.items():
            key = f"rate_limit:global:{limit_name}"
            is_allowed, limit_info = await self._apply_rate_limit_rule(key, rule)
            
            if not is_allowed:
                return False, limit_info
        
        return True, None
    
    async def _apply_rate_limit_rule(self, key: str, rule: RateLimitRule) -> Tuple[bool, Optional[Dict]]:
        """Apply a specific rate limiting rule"""
        
        try:
            current_time = int(time.time())
            window_start = current_time - rule.window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request timestamp
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, rule.window)
            
            results = await pipe.execute()
            current_count = results[1]  # Result from zcard
            
            # Check if limit exceeded
            effective_limit = rule.limit
            
            # Check burst limit if configured
            if rule.burst_limit:
                burst_key = f"{key}:burst"
                burst_window_start = current_time - rule.burst_window
                
                burst_pipe = self.redis.pipeline()
                burst_pipe.zremrangebyscore(burst_key, 0, burst_window_start)
                burst_pipe.zcard(burst_key)
                burst_pipe.zadd(burst_key, {str(current_time): current_time})
                burst_pipe.expire(burst_key, rule.burst_window)
                
                burst_results = await burst_pipe.execute()
                burst_count = burst_results[1]
                
                if burst_count <= rule.burst_limit:
                    effective_limit = rule.burst_limit
            
            if current_count > effective_limit:
                # Apply blocking if configured
                if rule.block_duration > 0:
                    await self._apply_block(key, rule.block_duration)
                
                # Calculate retry after
                retry_after = await self._calculate_retry_after(key, rule)
                
                return False, {
                    "error": rule.message,
                    "limit": effective_limit,
                    "current": current_count,
                    "window": rule.window,
                    "retry_after": retry_after,
                    "reset_time": current_time + retry_after
                }
            
            return True, {
                "limit": effective_limit,
                "current": current_count,
                "remaining": max(0, effective_limit - current_count),
                "window": rule.window,
                "reset_time": current_time + rule.window
            }
            
        except Exception as e:
            self.logger.error(f"Rate limit rule application failed", key=key, error=str(e))
            # On Redis error, allow the request
            return True, None
    
    async def _apply_block(self, identifier: str, duration: int):
        """Apply temporary block to an identifier"""
        block_key = f"blocked:{identifier}"
        await self.redis.setex(block_key, duration, int(time.time()))
    
    async def _is_blocked(self, identifier: str, block_type: str) -> bool:
        """Check if an identifier is currently blocked"""
        block_key = f"blocked:{block_type}:{identifier}"
        return await self.redis.exists(block_key)
    
    async def _get_block_remaining_time(self, identifier: str, block_type: str) -> int:
        """Get remaining block time for an identifier"""
        block_key = f"blocked:{block_type}:{identifier}"
        ttl = await self.redis.ttl(block_key)
        return max(0, ttl)
    
    async def _calculate_retry_after(self, key: str, rule: RateLimitRule) -> int:
        """Calculate retry-after time"""
        # Get the oldest request in current window
        oldest_requests = await self.redis.zrange(key, 0, 0, withscores=True)
        
        if oldest_requests:
            oldest_time = int(oldest_requests[0][1])
            return oldest_time + rule.window - int(time.time())
        
        return rule.window
    
    async def _log_rate_limit_event(self, event_type: str, data: Dict):
        """Log rate limiting events for monitoring"""
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        
        # Log to structured logger
        self.logger.warning("Rate limit event", **event_data)
        
        # Store in Redis for analytics (optional)
        try:
            event_key = f"rate_limit_events:{datetime.utcnow().strftime('%Y%m%d')}"
            await self.redis.lpush(event_key, json.dumps(event_data))
            await self.redis.expire(event_key, 86400 * 7)  # Keep for 7 days
        except Exception:
            pass  # Don't fail on logging errors
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request with proxy support"""
        # Check for forwarded headers (from load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for rate limiting"""
        # Remove query parameters
        path = path.split("?")[0]
        
        # Replace path parameters with placeholders
        import re
        path = re.sub(r'/\d+', '/{id}', path)  # Replace numeric IDs
        path = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', path)  # Replace UUIDs
        
        return path
    
    async def get_rate_limit_status(self, identifier: str, limit_type: str) -> Dict:
        """Get current rate limit status for an identifier"""
        try:
            status = {}
            
            if limit_type == "ip":
                limits = self.config.IP_LIMITS
            elif limit_type == "user":
                limits = self.config.USER_LIMITS.get("free_plan", {})
            elif limit_type == "global":
                limits = self.config.GLOBAL_LIMITS
            else:
                return {"error": "Invalid limit type"}
            
            for limit_name, rule in limits.items():
                key = f"rate_limit:{limit_type}:{identifier}:{limit_name}"
                current_count = await self.redis.zcard(key)
                
                status[limit_name] = {
                    "limit": rule.limit,
                    "current": current_count,
                    "remaining": max(0, rule.limit - current_count),
                    "window": rule.window,
                    "reset_time": int(time.time()) + rule.window
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get rate limit status", error=str(e))
            return {"error": "Failed to retrieve status"}
    
    async def reset_rate_limit(self, identifier: str, limit_type: str, limit_name: Optional[str] = None):
        """Reset rate limits for an identifier (admin function)"""
        try:
            if limit_name:
                key = f"rate_limit:{limit_type}:{identifier}:{limit_name}"
                await self.redis.delete(key)
            else:
                # Reset all limits for the identifier
                pattern = f"rate_limit:{limit_type}:{identifier}:*"
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
            
            self.logger.info(f"Rate limit reset", identifier=identifier, limit_type=limit_type, limit_name=limit_name)
            
        except Exception as e:
            self.logger.error(f"Failed to reset rate limit", error=str(e))
            raise
    
    async def get_metrics(self) -> Dict:
        """Get rate limiting metrics"""
        return {
            "total_requests": self.metrics["total_requests"],
            "blocked_requests": self.metrics["blocked_requests"],
            "block_rate": self.metrics["blocked_requests"] / max(1, self.metrics["total_requests"]),
            "rate_limited_ips_count": len(self.metrics["rate_limited_ips"]),
            "rate_limited_users_count": len(self.metrics["rate_limited_users"]),
            "active_blocks": await self._count_active_blocks()
        }
    
    async def _count_active_blocks(self) -> int:
        """Count currently active blocks"""
        try:
            block_keys = await self.redis.keys("blocked:*")
            return len(block_keys)
        except Exception:
            return 0

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app: ASGIApp, redis_client: redis.Redis):
        super().__init__(app)
        self.rate_limiter = AdvancedRateLimiter(redis_client)
        self.logger = structlog.get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting"""
        
        # Skip rate limiting for certain paths
        skip_paths = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Extract user information if available
        user_id = None
        api_key = None
        plan_type = "free_plan"
        
        # Try to extract user info from headers or auth token
        auth_header = request.headers.get("Authorization")
        if auth_header:
            # This would integrate with your auth system
            # For now, we'll extract from a custom header
            user_id = request.headers.get("X-User-ID")
            plan_type = request.headers.get("X-Plan-Type", "free_plan")
        
        api_key = request.headers.get("X-API-Key")
        
        # Check rate limits
        is_allowed, limit_info = await self.rate_limiter.check_rate_limit(
            request, user_id, api_key, plan_type
        )
        
        if not is_allowed:
            # Return rate limit exceeded response
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": limit_info.get("error", "Too many requests"),
                    "retry_after": limit_info.get("retry_after", 60),
                    "reset_time": limit_info.get("reset_time"),
                    "limit": limit_info.get("limit"),
                    "current": limit_info.get("current")
                },
                headers={
                    "X-RateLimit-Limit": str(limit_info.get("limit", 0)),
                    "X-RateLimit-Remaining": str(limit_info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(limit_info.get("reset_time", 0)),
                    "Retry-After": str(limit_info.get("retry_after", 60))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        if limit_info:
            response.headers["X-RateLimit-Limit"] = str(limit_info.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(limit_info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(limit_info.get("reset_time", 0))
        
        return response

# Utility functions for integration
async def create_rate_limiter() -> AdvancedRateLimiter:
    """Create and return a rate limiter instance"""
    redis_client = redis.from_url(settings.REDIS_URL)
    return AdvancedRateLimiter(redis_client)

async def get_rate_limit_middleware(app: ASGIApp) -> RateLimitMiddleware:
    """Create rate limiting middleware"""
    redis_client = redis.from_url(settings.REDIS_URL)
    return RateLimitMiddleware(app, redis_client)
