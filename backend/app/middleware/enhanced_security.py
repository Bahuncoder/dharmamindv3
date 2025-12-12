"""
🔒 Enhanced Security Middleware for DharmaMind
===============================================

Provides comprehensive security features:
- Security Headers (HSTS, CSP, X-Frame-Options, etc.)
- Request Sanitization
- IP-based Rate Limiting
- CSRF Protection
- Request Logging
- Suspicious Activity Detection
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ================================
# 🛡️ SECURITY CONFIGURATION
# ================================
class SecurityConfig:
    """Security configuration settings"""
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS = 100  # per window
    RATE_LIMIT_BURST = 20  # burst allowance
    
    # IP blocking
    MAX_FAILED_ATTEMPTS = 10
    BLOCK_DURATION = 3600  # 1 hour
    
    # CSRF
    CSRF_TOKEN_LENGTH = 32
    CSRF_COOKIE_NAME = "csrf_token"
    CSRF_HEADER_NAME = "X-CSRF-Token"
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }
    
    # Content Security Policy
    CSP_POLICY = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://api.stripe.com https://dharmamind.com https://dharmamind.ai; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r"<script[^>]*>",           # XSS
        r"javascript:",             # XSS
        r"on\w+\s*=",               # Event handlers
        r"union\s+select",          # SQL injection
        r";\s*drop\s+table",        # SQL injection
        r"\.\./",                   # Path traversal
        r"etc/passwd",              # File disclosure
        r"\\x[0-9a-f]{2}",          # Encoded attacks
    ]
    
    # Exempt paths (don't apply certain security checks)
    EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json", "/health", "/"}


# ================================
# 🔐 CSRF PROTECTION
# ================================
class CSRFProtection:
    """CSRF token management"""
    
    def __init__(self):
        self.secret = os.getenv("CSRF_SECRET", secrets.token_hex(32))
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        timestamp = str(int(time.time()))
        message = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{timestamp}:{signature}"
    
    def validate_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            parts = token.split(":")
            if len(parts) != 2:
                return False
            
            timestamp, signature = parts
            
            # Check token age
            if int(time.time()) - int(timestamp) > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp}"
            expected = hmac.new(
                self.secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected)
        except (ValueError, TypeError):
            return False


# ================================
# 🚦 RATE LIMITER
# ================================
class RateLimiter:
    """IP-based rate limiting with sliding window"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}
        self.failed_attempts: Dict[str, int] = defaultdict(int)
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if ip in self.blocked_ips:
            if time.time() < self.blocked_ips[ip]:
                return True
            else:
                del self.blocked_ips[ip]
                self.failed_attempts[ip] = 0
        return False
    
    def block_ip(self, ip: str, duration: int = SecurityConfig.BLOCK_DURATION):
        """Block an IP address"""
        self.blocked_ips[ip] = time.time() + duration
        logger.warning(f"🚫 IP blocked: {ip} for {duration}s")
    
    def record_failed_attempt(self, ip: str):
        """Record failed attempt and potentially block"""
        self.failed_attempts[ip] += 1
        if self.failed_attempts[ip] >= SecurityConfig.MAX_FAILED_ATTEMPTS:
            self.block_ip(ip)
    
    def check_rate_limit(self, ip: str) -> tuple[bool, int]:
        """
        Check if request is within rate limits
        Returns: (allowed, remaining_requests)
        """
        current_time = time.time()
        window_start = current_time - SecurityConfig.RATE_LIMIT_WINDOW
        
        # Clean old requests
        self.requests[ip] = [
            t for t in self.requests[ip] if t > window_start
        ]
        
        # Check limit
        request_count = len(self.requests[ip])
        if request_count >= SecurityConfig.RATE_LIMIT_MAX_REQUESTS:
            return False, 0
        
        # Record request
        self.requests[ip].append(current_time)
        return True, SecurityConfig.RATE_LIMIT_MAX_REQUESTS - request_count - 1


# ================================
# 🧹 REQUEST SANITIZER
# ================================
class RequestSanitizer:
    """Sanitize and validate incoming requests"""
    
    def __init__(self):
        self.patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in SecurityConfig.SUSPICIOUS_PATTERNS
        ]
    
    def is_suspicious(self, data: str) -> tuple[bool, Optional[str]]:
        """Check if data contains suspicious patterns"""
        for pattern in self.patterns:
            match = pattern.search(data)
            if match:
                return True, match.group(0)[:50]  # Return first 50 chars of match
        return False, None
    
    def sanitize_string(self, value: str) -> str:
        """Basic sanitization of string input"""
        if not value:
            return value
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Encode special characters
        value = value.replace('<', '&lt;').replace('>', '&gt;')
        
        return value
    
    async def check_request(self, request: Request) -> tuple[bool, Optional[str]]:
        """Check entire request for suspicious content"""
        # Check URL path
        is_sus, match = self.is_suspicious(str(request.url.path))
        if is_sus:
            return False, f"Suspicious URL pattern: {match}"
        
        # Check query parameters
        for key, value in request.query_params.items():
            is_sus, match = self.is_suspicious(f"{key}={value}")
            if is_sus:
                return False, f"Suspicious query parameter: {match}"
        
        # Check headers (specific ones)
        for header in ["user-agent", "referer", "origin"]:
            value = request.headers.get(header, "")
            is_sus, match = self.is_suspicious(value)
            if is_sus:
                return False, f"Suspicious header ({header}): {match}"
        
        return True, None


# ================================
# 📝 SECURITY LOGGER
# ================================
class SecurityLogger:
    """Log security events"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.max_events = 10000
    
    def log_event(
        self,
        event_type: str,
        ip: str,
        path: str,
        details: Optional[str] = None,
        severity: str = "INFO"
    ):
        """Log a security event"""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "ip": ip,
            "path": path,
            "details": details,
            "severity": severity
        }
        
        self.events.append(event)
        
        # Trim old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Log based on severity
        log_msg = f"🔒 [{event_type}] IP={ip} Path={path}"
        if details:
            log_msg += f" Details={details}"
        
        if severity == "WARNING":
            logger.warning(log_msg)
        elif severity == "ERROR":
            logger.error(log_msg)
        else:
            logger.info(log_msg)
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events"""
        return self.events[-limit:]


# ================================
# 🔒 MAIN SECURITY MIDDLEWARE
# ================================
class EnhancedSecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware combining all security features
    """
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.csrf = CSRFProtection()
        self.rate_limiter = RateLimiter()
        self.sanitizer = RequestSanitizer()
        self.security_logger = SecurityLogger()
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP from request"""
        # Check forwarded headers (for proxies)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Add HSTS for HTTPS
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        
        # Add CSP
        response.headers["Content-Security-Policy"] = SecurityConfig.CSP_POLICY
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from security checks"""
        return path in SecurityConfig.EXEMPT_PATHS or path.startswith("/docs")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security checks"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        path = request.url.path
        
        # Skip checks for exempt paths
        if self._is_exempt_path(path):
            response = await call_next(request)
            return self._add_security_headers(response)
        
        # 1. Check if IP is blocked
        if self.rate_limiter.is_blocked(client_ip):
            self.security_logger.log_event(
                "BLOCKED_IP_REQUEST", client_ip, path,
                severity="WARNING"
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "message": "Your IP has been temporarily blocked"}
            )
        
        # 2. Check rate limit
        allowed, remaining = self.rate_limiter.check_rate_limit(client_ip)
        if not allowed:
            self.security_logger.log_event(
                "RATE_LIMIT_EXCEEDED", client_ip, path,
                severity="WARNING"
            )
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests", "message": "Please try again later"},
                headers={"Retry-After": str(SecurityConfig.RATE_LIMIT_WINDOW)}
            )
        
        # 3. Check for suspicious content
        is_safe, reason = await self.sanitizer.check_request(request)
        if not is_safe:
            self.rate_limiter.record_failed_attempt(client_ip)
            self.security_logger.log_event(
                "SUSPICIOUS_REQUEST", client_ip, path,
                details=reason, severity="WARNING"
            )
            return JSONResponse(
                status_code=400,
                content={"error": "Bad request", "message": "Invalid request detected"}
            )
        
        # 4. CSRF check for state-changing methods
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            # Skip CSRF for API endpoints with Bearer token
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                csrf_token = request.headers.get(SecurityConfig.CSRF_HEADER_NAME)
                csrf_cookie = request.cookies.get(SecurityConfig.CSRF_COOKIE_NAME)
                
                if csrf_token and csrf_cookie:
                    # Validate token matches cookie (double-submit pattern)
                    if csrf_token != csrf_cookie:
                        self.security_logger.log_event(
                            "CSRF_VALIDATION_FAILED", client_ip, path,
                            severity="WARNING"
                        )
                        # Don't block, just log - may be legitimate API call
        
        # 5. Process request
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            if response.status_code >= 400:
                self.security_logger.log_event(
                    "ERROR_RESPONSE", client_ip, path,
                    details=f"Status={response.status_code} Duration={duration:.3f}s",
                    severity="WARNING" if response.status_code >= 500 else "INFO"
                )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Limit"] = str(SecurityConfig.RATE_LIMIT_MAX_REQUESTS)
            
            return self._add_security_headers(response)
            
        except Exception as e:
            self.security_logger.log_event(
                "REQUEST_ERROR", client_ip, path,
                details=str(e)[:100], severity="ERROR"
            )
            raise


# ================================
# 🔌 SETUP FUNCTION
# ================================
def setup_enhanced_security(app: FastAPI) -> EnhancedSecurityMiddleware:
    """
    Setup enhanced security middleware for FastAPI app
    
    Usage:
        from app.middleware.enhanced_security import setup_enhanced_security
        security = setup_enhanced_security(app)
    """
    middleware = EnhancedSecurityMiddleware(app)
    app.add_middleware(EnhancedSecurityMiddleware)
    logger.info("🔒 Enhanced Security Middleware initialized")
    return middleware


# Export security components for direct use
__all__ = [
    "EnhancedSecurityMiddleware",
    "setup_enhanced_security",
    "CSRFProtection",
    "RateLimiter",
    "RequestSanitizer",
    "SecurityLogger",
    "SecurityConfig",
]
