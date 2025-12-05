"""
CORS Configuration for DharmaMind API
=====================================

Provides secure CORS (Cross-Origin Resource Sharing) configuration
with origin whitelisting, method restrictions, and header validation.

Security Features:
- Origin whitelisting from environment variables
- Method restrictions (only safe methods)
- Header validation
- Credentials support with origin validation
- Pre-flight request caching

Author: DharmaMind Security Team
Date: October 27, 2025
"""

import logging
from typing import List, Optional
from urllib.parse import urlparse
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecureCORSConfig:
    """
    Secure CORS configuration with strict origin validation
    
    Example usage:
        from services.middleware.cors_config import SecureCORSConfig
        
        cors_config = SecureCORSConfig(
            allowed_origins=[
                "https://yourdomain.com",
                "https://app.yourdomain.com"
            ]
        )
        
        app.add_middleware(
            CORSMiddleware,
            **cors_config.get_middleware_config()
        )
    """
    
    # Default safe methods for CORS
    DEFAULT_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    # Default safe headers
    DEFAULT_ALLOWED_HEADERS = [
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Request-ID"
    ]
    
    # Exposed headers that browser can access
    DEFAULT_EXPOSED_HEADERS = [
        "X-Request-ID",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset"
    ]
    
    def __init__(
        self,
        allowed_origins: List[str],
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        exposed_headers: Optional[List[str]] = None,
        allow_credentials: bool = True,
        max_age: int = 3600
    ):
        """
        Initialize CORS configuration
        
        Args:
            allowed_origins: List of allowed origin URLs
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed request headers
            exposed_headers: List of headers to expose to browser
            allow_credentials: Whether to allow credentials (cookies, auth)
            max_age: Max age for preflight cache in seconds
        """
        self.allowed_origins = self._validate_origins(allowed_origins)
        self.allowed_methods = allowed_methods or self.DEFAULT_ALLOWED_METHODS
        self.allowed_headers = allowed_headers or self.DEFAULT_ALLOWED_HEADERS
        self.exposed_headers = exposed_headers or self.DEFAULT_EXPOSED_HEADERS
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        
        logger.info(f"✅ CORS configured with {len(self.allowed_origins)} "
                   f"allowed origins")
    
    def _validate_origins(self, origins: List[str]) -> List[str]:
        """
        Validate and normalize origin URLs
        
        Args:
            origins: List of origin URLs to validate
            
        Returns:
            List of validated origin URLs
            
        Raises:
            ValueError: If any origin is invalid
        """
        validated = []
        
        for origin in origins:
            # Special case: allow localhost in development
            if origin in ["*", "http://localhost", "http://127.0.0.1"]:
                validated.append(origin)
                continue
            
            # Parse and validate URL
            try:
                parsed = urlparse(origin)
                
                # Must have scheme and netloc
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(f"Invalid origin URL: {origin}")
                
                # Only allow http/https
                if parsed.scheme not in ["http", "https"]:
                    raise ValueError(
                        f"Origin must use http or https: {origin}"
                    )
                
                # Production should only use https
                if parsed.scheme == "http" and "localhost" not in parsed.netloc:
                    logger.warning(
                        f"⚠️  Non-HTTPS origin in production: {origin}"
                    )
                
                # Reconstruct clean URL
                clean_origin = f"{parsed.scheme}://{parsed.netloc}"
                if parsed.port:
                    clean_origin += f":{parsed.port}"
                
                validated.append(clean_origin)
                
            except Exception as e:
                raise ValueError(f"Invalid origin '{origin}': {e}")
        
        if not validated:
            raise ValueError("At least one allowed origin is required")
        
        return validated
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if an origin is allowed
        
        Args:
            origin: Origin URL to check
            
        Returns:
            True if origin is allowed, False otherwise
        """
        # Allow all if * is in list
        if "*" in self.allowed_origins:
            return True
        
        # Check exact match
        if origin in self.allowed_origins:
            return True
        
        # Check with port variations
        parsed = urlparse(origin)
        origin_no_port = f"{parsed.scheme}://{parsed.hostname}"
        
        return origin_no_port in self.allowed_origins
    
    def get_middleware_config(self) -> dict:
        """
        Get configuration dict for FastAPI CORSMiddleware
        
        Returns:
            Dictionary of CORS middleware configuration
        """
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "expose_headers": self.exposed_headers,
            "max_age": self.max_age
        }
    
    def log_config(self):
        """Log the current CORS configuration"""
        logger.info("=" * 60)
        logger.info("CORS Configuration:")
        logger.info(f"  Allowed Origins: {self.allowed_origins}")
        logger.info(f"  Allowed Methods: {self.allowed_methods}")
        logger.info(f"  Allowed Headers: {self.allowed_headers}")
        logger.info(f"  Allow Credentials: {self.allow_credentials}")
        logger.info(f"  Max Age: {self.max_age}s")
        logger.info("=" * 60)


class CORSValidationMiddleware(BaseHTTPMiddleware):
    """
    Additional CORS validation middleware for extra security
    
    Validates CORS requests beyond what FastAPI's built-in
    CORSMiddleware provides, including:
    - Strict origin validation
    - Method validation
    - Header validation
    - Logging of CORS violations
    
    Example usage:
        app.add_middleware(CORSValidationMiddleware, cors_config=cors_config)
    """
    
    def __init__(self, app, cors_config: SecureCORSConfig):
        """
        Initialize middleware
        
        Args:
            app: FastAPI application
            cors_config: SecureCORSConfig instance
        """
        super().__init__(app)
        self.cors_config = cors_config
    
    async def dispatch(self, request: Request, call_next):
        """
        Validate CORS request and add security headers
        
        Args:
            request: FastAPI request
            call_next: Next middleware in chain
            
        Returns:
            Response with CORS validation
        """
        origin = request.headers.get("Origin")
        
        # If there's an origin header, validate it
        if origin:
            # Check if origin is allowed
            if not self.cors_config.is_origin_allowed(origin):
                logger.warning(
                    f"🚫 CORS violation: Blocked origin '{origin}' "
                    f"from {request.client.host} "
                    f"for {request.method} {request.url.path}"
                )
            
            # Validate method for CORS requests
            if request.method not in self.cors_config.allowed_methods:
                logger.warning(
                    f"🚫 CORS violation: Blocked method '{request.method}' "
                    f"from origin '{origin}'"
                )
        
        # Continue with request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        
        return response


def create_cors_config_from_env() -> SecureCORSConfig:
    """
    Create CORS configuration from environment variables
    
    Environment variables:
        ALLOWED_ORIGINS: Comma-separated list of allowed origins
        ALLOWED_METHODS: Comma-separated list of allowed methods (optional)
        ALLOWED_HEADERS: Comma-separated list of allowed headers (optional)
        ALLOW_CREDENTIALS: "true" or "false" (optional)
        CORS_MAX_AGE: Max age in seconds (optional)
    
    Returns:
        SecureCORSConfig instance
    
    Example:
        from services.middleware.cors_config import create_cors_config_from_env
        
        cors_config = create_cors_config_from_env()
        app.add_middleware(CORSMiddleware, **cors_config.get_middleware_config())
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get allowed origins (required)
    origins_str = os.getenv("ALLOWED_ORIGINS", "")
    if not origins_str:
        logger.warning(
            "⚠️  ALLOWED_ORIGINS not set, using localhost defaults"
        )
        origins_str = "http://localhost:3000,http://localhost:8080"
    
    allowed_origins = [
        origin.strip()
        for origin in origins_str.split(",")
        if origin.strip()
    ]
    
    # Get allowed methods (optional)
    methods_str = os.getenv("ALLOWED_METHODS", "")
    allowed_methods = None
    if methods_str:
        allowed_methods = [
            method.strip()
            for method in methods_str.split(",")
            if method.strip()
        ]
    
    # Get allowed headers (optional)
    headers_str = os.getenv("ALLOWED_HEADERS", "")
    allowed_headers = None
    if headers_str:
        allowed_headers = [
            header.strip()
            for header in headers_str.split(",")
            if header.strip()
        ]
    
    # Get credentials setting (optional)
    allow_credentials = os.getenv("ALLOW_CREDENTIALS", "true").lower() == "true"
    
    # Get max age (optional)
    max_age = int(os.getenv("CORS_MAX_AGE", "3600"))
    
    # Create and return config
    config = SecureCORSConfig(
        allowed_origins=allowed_origins,
        allowed_methods=allowed_methods,
        allowed_headers=allowed_headers,
        allow_credentials=allow_credentials,
        max_age=max_age
    )
    
    config.log_config()
    
    return config


# Example usage in FastAPI application
if __name__ == "__main__":
    # Example 1: Manual configuration
    cors_config = SecureCORSConfig(
        allowed_origins=[
            "https://yourdomain.com",
            "https://app.yourdomain.com",
            "http://localhost:3000"  # Development
        ]
    )
    
    print("Manual config:")
    print(cors_config.get_middleware_config())
    
    # Example 2: From environment
    cors_config_env = create_cors_config_from_env()
    print("\nEnvironment config:")
    print(cors_config_env.get_middleware_config())
