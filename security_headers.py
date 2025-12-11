"""
🔒 DharmaMind Security Headers Configuration
Enhanced security middleware for spiritual AI protection
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import time

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    🛡️ Security headers middleware for DharmaMind
    Protects spiritual AI endpoints with proper security headers
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers for spiritual data protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Custom header for spiritual AI protection
        response.headers["X-Dharma-Protection"] = "spiritual-data-secured"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    🕉️ Spiritual-aware rate limiting
    Protects against abuse while allowing genuine spiritual seeking
    """
    
    def __init__(self, app, calls_per_minute: int = 30):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if any(t > current_time - 60 for t in times)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            # Filter requests from last minute
            recent_requests = [t for t in self.requests[client_ip] if t > current_time - 60]
            if len(recent_requests) >= self.calls_per_minute:
                return Response(
                    content="🕉️ Rate limit exceeded. Please seek wisdom mindfully.",
                    status_code=429,
                    headers={"X-Dharma-Message": "patience-is-a-virtue"}
                )
            self.requests[client_ip] = recent_requests + [current_time]
        else:
            self.requests[client_ip] = [current_time]
        
        return await call_next(request)

def configure_security(app: FastAPI):
    """
    🔒 Configure comprehensive security for DharmaMind
    """
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add rate limiting (gentle for spiritual seekers)
    app.add_middleware(RateLimitMiddleware, calls_per_minute=30)
    
    # Configure CORS properly
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],  # Update with your domain
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    return app

# Example usage in your FastAPI app:
"""
from security_headers import configure_security

app = FastAPI(title="DharmaMind Spiritual AI")
app = configure_security(app)
"""
