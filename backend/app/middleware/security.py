"""
🕉️ DharmaMind Security Middleware - Enterprise-Grade Protection

Implements comprehensive security measures including:
- HTTPS enforcement
- Security headers
- CORS protection
- Rate limiting
- Request validation
- Security event logging
"""

import time
import json
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis
from ..config import settings, is_production
from ..services.cache_service import AdvancedCacheService

# Security logger
security_logger = logging.getLogger("dharmamind.security")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add comprehensive security headers to all responses
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            # HTTPS Security
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            
            # Content Security
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # Privacy and Security
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            
            # DharmaMind Branding
            "X-Powered-By": "DharmaMind-Spiritual-AI",
            "X-Dharma-Version": "1.0.0-beta",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' http://localhost:8003; "  # LLM Gateway only
                "frame-ancestors 'none';"
            )
        }
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add security context
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting with Redis backend and intelligent throttling
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.cache_service = AdvancedCacheService()
        self.rate_limits = {
            "default": {
                "requests": settings.RATE_LIMIT_REQUESTS,
                "window": settings.RATE_LIMIT_WINDOW
            },
            "auth": {
                "requests": 10,
                "window": 300  # 5 minutes
            },
            "chat": {
                "requests": settings.RATE_LIMIT_PER_MINUTE,
                "window": 60
            },
            "api": {
                "requests": 1000,
                "window": 3600  # Premium users
            }
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Determine rate limit based on endpoint
        limit_type = self._get_limit_type(request.url.path)
        rate_limit = self.rate_limits.get(limit_type, self.rate_limits["default"])
        
        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(
            client_id, limit_type, rate_limit
        )
        
        if not is_allowed:
            # Log security event
            security_logger.warning(
                f"Rate limit exceeded for {client_id} on {request.url.path}",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "limit_type": limit_type,
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please slow down and try again later.",
                    "retry_after": reset_time,
                    "dharmic_wisdom": "धैर्यं सर्व दुःख क्षयकरं - Patience destroys all suffering"
                },
                headers={
                    "Retry-After": str(reset_time),
                    "X-RateLimit-Limit": str(rate_limit["requests"]),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(reset_time)
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Try to get authenticated user ID
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    def _get_limit_type(self, path: str) -> str:
        """Determine rate limit type based on endpoint"""
        if path.startswith("/auth/"):
            return "auth"
        elif path.startswith("/chat/") or path.startswith("/api/chat"):
            return "chat"
        elif path.startswith("/api/"):
            return "api"
        else:
            return "default"
    
    async def _check_rate_limit(
        self, 
        client_id: str, 
        limit_type: str, 
        rate_limit: Dict[str, int]
    ) -> tuple[bool, int, int]:
        """Check if request is within rate limits"""
        key = f"rate_limit:{limit_type}:{client_id}"
        window = rate_limit["window"]
        limit = rate_limit["requests"]
        
        try:
            # Get current count
            current_count = await self.cache_service.get(key) or 0
            current_count = int(current_count)
            
            # Calculate reset time
            reset_time = int(time.time()) + window
            
            if current_count >= limit:
                return False, 0, reset_time
            
            # Increment counter
            await self.cache_service.set(key, current_count + 1, expire=window)
            
            remaining = limit - current_count - 1
            return True, remaining, reset_time
            
        except Exception as e:
            # Log error but allow request (fail open)
            security_logger.error(f"Rate limiting error: {e}")
            return True, limit, int(time.time()) + window


class BruteForceProtectionMiddleware(BaseHTTPMiddleware):
    """
    Protect against brute force attacks with exponential backoff
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.cache_service = AdvancedCacheService()
        self.max_attempts = 5
        self.base_lockout_time = 300  # 5 minutes
        self.max_lockout_time = 3600  # 1 hour
    
    async def dispatch(self, request: Request, call_next):
        # Only protect authentication endpoints
        if not self._is_auth_endpoint(request.url.path):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        # Check if client is locked out
        lockout_info = await self._get_lockout_info(client_id)
        if lockout_info and lockout_info["locked_until"] > time.time():
            remaining_time = int(lockout_info["locked_until"] - time.time())
            
            security_logger.warning(
                f"Brute force lockout active for {client_id}",
                extra={
                    "client_id": client_id,
                    "attempts": lockout_info["attempts"],
                    "remaining_time": remaining_time
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Account temporarily locked",
                    "message": f"Too many failed attempts. Try again in {remaining_time} seconds.",
                    "retry_after": remaining_time,
                    "dharmic_wisdom": "धैर्यं धर्मः - Patience is righteousness"
                },
                headers={"Retry-After": str(remaining_time)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Handle failed authentication
        if response.status_code == 401 and request.method == "POST":
            await self._record_failed_attempt(client_id)
        elif response.status_code == 200 and request.method == "POST":
            # Clear failed attempts on successful login
            await self._clear_failed_attempts(client_id)
        
        return response
    
    def _is_auth_endpoint(self, path: str) -> bool:
        """Check if path is an authentication endpoint"""
        auth_paths = ["/auth/login", "/auth/token", "/auth/refresh"]
        return any(path.startswith(auth_path) for auth_path in auth_paths)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for brute force tracking"""
        # Use IP address for brute force protection
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _get_lockout_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get lockout information for client"""
        key = f"brute_force:{client_id}"
        data = await self.cache_service.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None
    
    async def _record_failed_attempt(self, client_id: str):
        """Record a failed authentication attempt"""
        key = f"brute_force:{client_id}"
        lockout_info = await self._get_lockout_info(client_id) or {
            "attempts": 0,
            "first_attempt": time.time(),
            "locked_until": 0
        }
        
        lockout_info["attempts"] += 1
        lockout_info["last_attempt"] = time.time()
        
        # Calculate lockout time with exponential backoff
        if lockout_info["attempts"] >= self.max_attempts:
            backoff_multiplier = min(2 ** (lockout_info["attempts"] - self.max_attempts), 8)
            lockout_time = min(
                self.base_lockout_time * backoff_multiplier,
                self.max_lockout_time
            )
            lockout_info["locked_until"] = time.time() + lockout_time
            
            security_logger.error(
                f"Client {client_id} locked out after {lockout_info['attempts']} failed attempts",
                extra={
                    "client_id": client_id,
                    "attempts": lockout_info["attempts"],
                    "lockout_time": lockout_time
                }
            )
        
        # Store for 24 hours
        await self.cache_service.set(key, json.dumps(lockout_info), expire=86400)
    
    async def _clear_failed_attempts(self, client_id: str):
        """Clear failed attempts after successful login"""
        key = f"brute_force:{client_id}"
        await self.cache_service.delete(key)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate and sanitize incoming requests
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.suspicious_patterns = [
            # SQL Injection patterns
            r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
            r"(\bor\b.*=.*\bor\b)|(\band\b.*=.*\band\b)",
            
            # XSS patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            
            # Command injection
            r"[;&|`]",
            r"\$\(.*\)",
            
            # Path traversal
            r"\.\./",
            r"\.\.\\",
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            security_logger.warning(
                f"Request too large: {content_length} bytes",
                extra={"request_id": request_id, "size": content_length}
            )
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"}
            )
        
        # Validate headers
        user_agent = request.headers.get("user-agent", "")
        if not user_agent or len(user_agent) > 1000:
            security_logger.warning(
                "Suspicious or missing User-Agent",
                extra={"request_id": request_id, "user_agent": user_agent[:100]}
            )
        
        # Check for suspicious patterns in URL and query parameters
        full_url = str(request.url)
        if self._contains_suspicious_patterns(full_url):
            security_logger.error(
                f"Suspicious request pattern detected: {request.url.path}",
                extra={
                    "request_id": request_id,
                    "url": full_url,
                    "client_ip": request.client.host if request.client else "unknown"
                }
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid request",
                    "message": "Request contains suspicious patterns"
                }
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log slow requests
        if process_time > 5.0:  # Log requests taking more than 5 seconds
            security_logger.warning(
                f"Slow request: {process_time:.2f}s for {request.url.path}",
                extra={
                    "request_id": request_id,
                    "process_time": process_time,
                    "path": request.url.path
                }
            )
        
        return response
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        import re
        text_lower = text.lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False


def setup_security_middleware(app):
    """
    Configure all security middleware for the FastAPI app
    """
    
    # 1. HTTPS Redirect (only in production)
    if is_production():
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # 2. CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time", "X-RateLimit-*"]
    )
    
    # 3. Security Headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 4. Request Validation
    app.add_middleware(RequestValidationMiddleware)
    
    # 5. Brute Force Protection
    app.add_middleware(BruteForceProtectionMiddleware)
    
    # 6. Rate Limiting
    app.add_middleware(RateLimitMiddleware)
    
    security_logger.info("🔒 Security middleware configured successfully")
    
    # Log security configuration
    security_logger.info(
        "Security Configuration",
        extra={
            "https_redirect": is_production(),
            "cors_origins": settings.CORS_ORIGINS,
            "rate_limit_requests": settings.RATE_LIMIT_REQUESTS,
            "rate_limit_window": settings.RATE_LIMIT_WINDOW,
            "environment": settings.ENVIRONMENT
        }
    )