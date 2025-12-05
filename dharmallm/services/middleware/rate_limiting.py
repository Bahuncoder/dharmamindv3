"""Rate limiting middleware for API endpoints - SECURE IMPLEMENTATION"""

import logging
from typing import Callable, Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement rate limiting with automatic blocking
    
    Features:
    - Per-IP rate limiting
    - Configurable limits and burst capacity
    - Automatic temporary blocking for abuse
    - Request tracking and monitoring
    """
    
    def __init__(
        self,
        app,
        calls_per_minute: int = 60,
        burst: int = 100,
        block_duration_minutes: int = 10
    ):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.burst = burst
        self.block_duration = timedelta(minutes=block_duration_minutes)
        
        # Track requests per IP
        self.request_counts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Track blocked IPs
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Track total requests (for monitoring)
        self.total_requests = 0
        self.total_blocked = 0
        
        logger.info(
            f"✓ Rate limiting enabled: {calls_per_minute} calls/min, "
            f"burst: {burst}"
        )
    
    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request
        
        Handles forwarded headers from reverse proxies
        """
        # Check X-Forwarded-For header (from reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (client IP)
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"
    
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted (e.g., localhost)"""
        # Add your whitelist logic here
        whitelist = ["127.0.0.1", "::1", "localhost"]
        return ip in whitelist
    
    def clean_old_requests(self, ip: str, cutoff: datetime):
        """Remove request timestamps older than cutoff time"""
        self.request_counts[ip] = [
            ts for ts in self.request_counts[ip] if ts > cutoff
        ]
    
    def check_rate_limit(self, ip: str, now: datetime) -> tuple[bool, int]:
        """
        Check if IP has exceeded rate limit
        
        Returns:
            (is_allowed, remaining_requests)
        """
        # Clean old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.clean_old_requests(ip, cutoff)
        
        # Count requests in last minute
        request_count = len(self.request_counts[ip])
        
        # Check against limit
        if request_count >= self.calls_per_minute:
            return False, 0
        
        remaining = self.calls_per_minute - request_count
        return True, remaining
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with rate limiting"""
        self.total_requests += 1
        client_ip = self.get_client_ip(request)
        now = datetime.now()
        
        # Skip rate limiting for whitelisted IPs
        if self.is_whitelisted(client_ip):
            return await call_next(request)
        
        # Check if IP is currently blocked
        if client_ip in self.blocked_ips:
            block_until = self.blocked_ips[client_ip]
            if now < block_until:
                # Still blocked
                remaining_time = int((block_until - now).total_seconds())
                logger.warning(
                    f"Blocked request from {client_ip} "
                    f"({remaining_time}s remaining)"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Too many requests. "
                        f"Please try again in {remaining_time} seconds."
                    ),
                    headers={
                        "Retry-After": str(remaining_time),
                        "X-RateLimit-Limit": str(self.calls_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(block_until.timestamp()))
                    }
                )
            else:
                # Block expired, remove from list
                del self.blocked_ips[client_ip]
                self.request_counts[client_ip] = []
                logger.info(f"Block expired for IP: {client_ip}")
        
        # Check rate limit
        is_allowed, remaining = self.check_rate_limit(client_ip, now)
        
        if not is_allowed:
            # Rate limit exceeded - block the IP
            block_until = now + self.block_duration
            self.blocked_ips[client_ip] = block_until
            self.total_blocked += 1
            
            logger.warning(
                f"Rate limit exceeded for {client_ip}. "
                f"Blocked until {block_until}"
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. "
                    f"Blocked for {self.block_duration.seconds // 60} minutes."
                ),
                headers={
                    "Retry-After": str(self.block_duration.seconds),
                    "X-RateLimit-Limit": str(self.calls_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(block_until.timestamp()))
                }
            )
        
        # Add current request to tracking
        self.request_counts[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        response.headers["X-RateLimit-Reset"] = str(
            int((now + timedelta(minutes=1)).timestamp())
        )
        
        return response
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiting statistics"""
        return {
            "total_requests": self.total_requests,
            "total_blocked": self.total_blocked,
            "currently_blocked_ips": len(self.blocked_ips),
            "tracked_ips": len(self.request_counts),
            "calls_per_minute": self.calls_per_minute,
            "burst": self.burst
        }
    
    def clear_ip(self, ip: str) -> bool:
        """Clear rate limit data for an IP (admin function)"""
        if ip in self.blocked_ips:
            del self.blocked_ips[ip]
        if ip in self.request_counts:
            del self.request_counts[ip]
        logger.info(f"Rate limit data cleared for IP: {ip}")
        return True
    
    def reset_all(self):
        """Reset all rate limiting data (admin function)"""
        self.request_counts.clear()
        self.blocked_ips.clear()
        self.total_requests = 0
        self.total_blocked = 0
        logger.info("✓ All rate limiting data reset")

