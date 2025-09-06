"""
🛡️ Advanced Security Middleware
Next-level security enhancements for DharmaMind

Features:
- Automatic MFA enforcement for admin routes
- Advanced rate limiting with user context
- IP reputation checking
- Suspicious activity detection
- Security headers enhancement
"""

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Dict, Set, Optional, Tuple
import time
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
import ipaddress
import re
from pathlib import Path

class AdvancedSecurityMiddleware(BaseHTTPMiddleware):
    """Advanced security middleware with intelligent threat detection"""
    
    def __init__(self, app, config: Optional[Dict] = None):
        super().__init__(app)
        
        # Configuration
        self.config = config or {}
        self.max_requests_per_minute = self.config.get('max_requests_per_minute', 60)
        self.max_login_attempts = self.config.get('max_login_attempts', 5)
        self.lockout_duration = self.config.get('lockout_duration', 900)  # 15 minutes
        self.mfa_required_routes = self.config.get('mfa_required_routes', [
            '/api/v1/admin',
            '/api/v1/security',
            '/api/v1/users/delete',
            '/api/v1/system/config'
        ])
        
        # Threat tracking
        self.request_counts = defaultdict(deque)
        self.failed_logins = defaultdict(list)
        self.blocked_ips = {}
        self.suspicious_patterns = []
        
        # Load IP reputation data
        self._load_threat_intelligence()
        
        # Initialize monitoring
        self.security_events = deque(maxlen=10000)
        self.active_sessions = {}
    
    def _load_threat_intelligence(self):
        """Load known threat IP ranges and patterns"""
        try:
            threat_file = Path("security/threat_intel.json")
            if threat_file.exists():
                with open(threat_file, 'r') as f:
                    threat_data = json.load(f)
                    self.known_bad_ips = set(threat_data.get('bad_ips', []))
                    self.suspicious_patterns = threat_data.get('patterns', [])
            else:
                self.known_bad_ips = set()
                self.suspicious_patterns = [
                    r'\.\./',  # Directory traversal
                    r'<script',  # XSS attempts
                    r'UNION.*SELECT',  # SQL injection
                    r'\\x[0-9a-f]{2}',  # Encoded payloads
                    r'cmd\.exe|powershell',  # Command injection
                ]
        except Exception:
            self.known_bad_ips = set()
            self.suspicious_patterns = []
    
    async def dispatch(self, request: Request, call_next):
        """Main security processing pipeline"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            # 1. IP reputation check
            if await self._check_ip_reputation(client_ip):
                return self._create_security_response(
                    "Access denied: IP reputation", 
                    status.HTTP_403_FORBIDDEN
                )
            
            # 2. Rate limiting check
            if await self._check_rate_limits(client_ip, request):
                return self._create_security_response(
                    "Rate limit exceeded", 
                    status.HTTP_429_TOO_MANY_REQUESTS
                )
            
            # 3. Pattern-based threat detection
            if await self._detect_malicious_patterns(request):
                await self._log_security_event(
                    "malicious_pattern_detected",
                    client_ip,
                    request.url.path,
                    "Suspicious pattern in request"
                )
                return self._create_security_response(
                    "Request blocked by security filter",
                    status.HTTP_400_BAD_REQUEST
                )
            
            # 4. MFA enforcement for sensitive routes
            if await self._requires_mfa_check(request):
                mfa_valid = await self._verify_mfa_status(request)
                if not mfa_valid:
                    return self._create_security_response(
                        "MFA required for this action",
                        status.HTTP_403_FORBIDDEN,
                        {"mfa_required": True}
                    )
            
            # 5. Process request
            response = await call_next(request)
            
            # 6. Post-process security
            await self._post_process_security(request, response, client_ip)
            
            # 7. Add security headers
            self._add_security_headers(response)
            
            # 8. Log successful request
            processing_time = time.time() - start_time
            if processing_time > 5.0:  # Log slow requests
                await self._log_security_event(
                    "slow_request",
                    client_ip,
                    request.url.path,
                    f"Request took {processing_time:.2f}s"
                )
            
            return response
            
        except Exception as e:
            # Log security middleware errors
            await self._log_security_event(
                "middleware_error",
                client_ip,
                request.url.path,
                f"Security middleware error: {str(e)}"
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxies"""
        # Check X-Forwarded-For header (from load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header (from nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"
    
    async def _check_ip_reputation(self, client_ip: str) -> bool:
        """Check if IP is in known bad actor list"""
        try:
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                block_info = self.blocked_ips[client_ip]
                if datetime.now() < block_info['expires']:
                    return True  # Still blocked
                else:
                    # Block expired, remove
                    del self.blocked_ips[client_ip]
            
            # Check against known bad IPs
            if client_ip in self.known_bad_ips:
                await self._block_ip(client_ip, "Known malicious IP", duration=3600)
                return True
            
            # Check for private/local IPs (allow in development)
            try:
                ip_obj = ipaddress.ip_address(client_ip)
                if ip_obj.is_private or ip_obj.is_loopback:
                    return False  # Allow local IPs
            except ValueError:
                pass  # Not a valid IP, continue checks
            
            return False
            
        except Exception:
            return False  # Don't block on errors
    
    async def _check_rate_limits(self, client_ip: str, request: Request) -> bool:
        """Advanced rate limiting with context awareness"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        while self.request_counts[client_ip] and self.request_counts[client_ip][0] < minute_ago:
            self.request_counts[client_ip].popleft()
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        
        # Check rate limits
        request_count = len(self.request_counts[client_ip])
        
        # Stricter limits for authentication endpoints
        if "/auth/" in request.url.path:
            max_auth_requests = 10
            if request_count > max_auth_requests:
                await self._log_security_event(
                    "auth_rate_limit_exceeded",
                    client_ip,
                    request.url.path,
                    f"Auth requests: {request_count}/min"
                )
                return True
        
        # General rate limiting
        if request_count > self.max_requests_per_minute:
            # Temporary block for excessive requests
            await self._block_ip(client_ip, "Rate limit exceeded", duration=300)
            return True
        
        return False
    
    async def _detect_malicious_patterns(self, request: Request) -> bool:
        """Detect malicious patterns in request"""
        try:
            # Check URL path
            url_path = str(request.url.path)
            query_params = str(request.url.query) if request.url.query else ""
            
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, url_path, re.IGNORECASE):
                    return True
                if re.search(pattern, query_params, re.IGNORECASE):
                    return True
            
            # Check request body for POST requests
            if request.method == "POST":
                # Note: This is a simplified check
                # In production, you'd want to read body more carefully
                pass
            
            # Check user agent for known bad patterns
            user_agent = request.headers.get("user-agent", "").lower()
            suspicious_agents = [
                "sqlmap", "nikto", "dirb", "dirbuster", "nmap",
                "masscan", "zap", "burp", "crawler", "bot"
            ]
            
            for agent in suspicious_agents:
                if agent in user_agent:
                    return True
            
            return False
            
        except Exception:
            return False  # Don't block on detection errors
    
    async def _requires_mfa_check(self, request: Request) -> bool:
        """Check if route requires MFA verification"""
        path = request.url.path
        
        # Check if path matches MFA required routes
        for required_route in self.mfa_required_routes:
            if path.startswith(required_route):
                return True
        
        return False
    
    async def _verify_mfa_status(self, request: Request) -> bool:
        """Verify MFA status for sensitive operations"""
        try:
            # Check for MFA bypass token in headers
            mfa_token = request.headers.get("X-MFA-Token")
            if mfa_token:
                # Verify the MFA token (simplified)
                from ..security.mfa_manager import get_mfa_manager
                mfa_manager = get_mfa_manager()
                
                # Check if it's a trusted device token
                valid, user_id = mfa_manager.verify_trusted_device(mfa_token)
                return valid
            
            # If no MFA token, require MFA verification
            return False
            
        except Exception:
            return False  # Deny on errors
    
    async def _block_ip(self, ip: str, reason: str, duration: int = 900):
        """Block IP address for specified duration"""
        self.blocked_ips[ip] = {
            'reason': reason,
            'blocked_at': datetime.now(),
            'expires': datetime.now() + timedelta(seconds=duration)
        }
        
        await self._log_security_event(
            "ip_blocked",
            ip,
            "",
            f"Blocked for {duration}s: {reason}"
        )
    
    async def _post_process_security(self, request: Request, response: Response, client_ip: str):
        """Post-process security checks"""
        # Monitor failed authentication attempts
        if request.url.path.endswith('/login') and response.status_code == 401:
            self.failed_logins[client_ip].append(datetime.now())
            
            # Clean old attempts (older than 1 hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self.failed_logins[client_ip] = [
                attempt for attempt in self.failed_logins[client_ip] 
                if attempt > cutoff
            ]
            
            # Check for brute force
            if len(self.failed_logins[client_ip]) >= self.max_login_attempts:
                await self._block_ip(
                    client_ip, 
                    f"Brute force detected ({len(self.failed_logins[client_ip])} failed attempts)",
                    self.lockout_duration
                )
    
    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "X-DharmaMind-Security": "enabled"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
    
    async def _log_security_event(self, event_type: str, ip: str, path: str, details: str):
        """Log security events for monitoring"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'ip': ip,
            'path': path,
            'details': details
        }
        
        self.security_events.append(event)
        
        # Write to security log file
        try:
            log_file = Path("logs/security.log")
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception:
            pass  # Don't fail on logging errors
    
    def _create_security_response(self, message: str, status_code: int, extra_data: Dict = None):
        """Create standardized security response"""
        data = {
            "error": message,
            "security_block": True,
            "timestamp": datetime.now().isoformat()
        }
        
        if extra_data:
            data.update(extra_data)
        
        return Response(
            content=json.dumps(data),
            status_code=status_code,
            headers={"Content-Type": "application/json"}
        )
    
    def get_security_stats(self) -> Dict:
        """Get current security statistics"""
        return {
            "blocked_ips": len(self.blocked_ips),
            "active_rate_limits": len(self.request_counts),
            "security_events_today": len([
                e for e in self.security_events 
                if datetime.fromisoformat(e['timestamp']).date() == datetime.now().date()
            ]),
            "total_security_events": len(self.security_events)
        }
