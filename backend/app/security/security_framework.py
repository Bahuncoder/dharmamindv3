# 🔐 Security Framework Implementation
# Advanced security hardening for DharmaMind production

import os
import hashlib
import secrets
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
import asyncio
import redis
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography.fernet import Fernet
import re
from email_validator import validate_email, EmailNotValidError
import ipaddress
from collections import defaultdict
import time

# ================================
# 🛡️ SECURITY CONFIGURATION
# ================================
class SecurityConfig:
    """Advanced security configuration"""
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    REFRESH_TOKEN_EXPIRATION_DAYS = 30
    
    # Password Security
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_HASH_ROUNDS = 12
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = 60
    RATE_LIMIT_REQUESTS_PER_HOUR = 1000
    RATE_LIMIT_BURST_SIZE = 10
    
    # Session Security
    SESSION_TIMEOUT_MINUTES = 30
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    
    # Encryption
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }

# ================================
# 🔒 ENCRYPTION SERVICE
# ================================
class EncryptionService:
    """Advanced encryption and decryption service"""
    
    def __init__(self):
        self.fernet = Fernet(SecurityConfig.ENCRYPTION_KEY)
        self.logger = logging.getLogger(__name__)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise HTTPException(status_code=500, detail="Encryption failed")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise HTTPException(status_code=500, detail="Decryption failed")
    
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt"""
        salt = bcrypt.gensalt(rounds=SecurityConfig.PASSWORD_HASH_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)

# ================================
# 🔑 JWT TOKEN SERVICE
# ================================
class JWTService:
    """Advanced JWT token management"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    def create_access_token(self, user_id: str, email: str, roles: List[str] = None) -> str:
        """Create JWT access token"""
        payload = {
            'user_id': user_id,
            'email': email,
            'roles': roles or ['user'],
            'exp': datetime.utcnow() + timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        token = jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        # Store token in Redis for revocation capability
        if self.redis_client:
            self.redis_client.setex(
                f"token:{user_id}:{payload['iat']}", 
                SecurityConfig.JWT_EXPIRATION_HOURS * 3600,
                token
            )
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRATION_DAYS),
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }
        
        token = jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        # Store refresh token in Redis
        if self.redis_client:
            self.redis_client.setex(
                f"refresh:{user_id}:{payload['iat']}", 
                SecurityConfig.REFRESH_TOKEN_EXPIRATION_DAYS * 24 * 3600,
                token
            )
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET_KEY, algorithms=[SecurityConfig.JWT_ALGORITHM])
            
            # Check if token is revoked
            if self.redis_client and payload.get('type') == 'access':
                stored_token = self.redis_client.get(f"token:{payload['user_id']}:{payload['iat']}")
                if not stored_token or stored_token.decode() != token:
                    raise HTTPException(status_code=401, detail="Token revoked")
            
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def revoke_token(self, user_id: str, iat: int):
        """Revoke a specific token"""
        if self.redis_client:
            self.redis_client.delete(f"token:{user_id}:{iat}")
    
    def revoke_all_tokens(self, user_id: str):
        """Revoke all tokens for a user"""
        if self.redis_client:
            pattern = f"token:{user_id}:*"
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)

# ================================
# 🛡️ RATE LIMITING SERVICE
# ================================
class RateLimitService:
    """Advanced rate limiting with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int = SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE,
        window: int = 60
    ) -> bool:
        """Check if request is within rate limit"""
        try:
            key = f"rate_limit:{identifier}:{window}"
            current = self.redis_client.get(key)
            
            if current is None:
                # First request in window
                self.redis_client.setex(key, window, 1)
                return True
            
            current_count = int(current)
            if current_count >= limit:
                return False
            
            # Increment counter
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limiting error: {e}")
            # Fail open for availability
            return True
    
    async def get_rate_limit_info(self, identifier: str, window: int = 60) -> Dict[str, int]:
        """Get current rate limit status"""
        key = f"rate_limit:{identifier}:{window}"
        current = self.redis_client.get(key)
        ttl = self.redis_client.ttl(key)
        
        return {
            'requests_made': int(current) if current else 0,
            'requests_remaining': max(0, SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE - (int(current) if current else 0)),
            'reset_time': ttl if ttl > 0 else window
        }

# ================================
# 🔍 INPUT VALIDATION SERVICE
# ================================
class InputValidationService:
    """Advanced input validation and sanitization"""
    
    # Regex patterns for validation
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,20}$')
    PHONE_PATTERN = re.compile(r'^\+?1?[0-9]{10,15}$')
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\s|^)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
        r"(\s|^)(or|and)\s+\d+\s*=\s*\d+",
        r"'(\s|;|--|\*|\/)",
        r"(\s|^)(xp_|sp_|sys\.)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[\s\S]*?>[\s\S]*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format"""
        try:
            validate_email(email)
            return cls.EMAIL_PATTERN.match(email) is not None
        except EmailNotValidError:
            return False
    
    @classmethod
    def validate_password(cls, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters")
        
        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if SecurityConfig.PASSWORD_REQUIRE_DIGITS and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'strength_score': cls._calculate_password_strength(password)
        }
    
    @classmethod
    def _calculate_password_strength(cls, password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length bonus
        score += min(25, len(password) * 2)
        
        # Character variety bonus
        if re.search(r'[a-z]', password):
            score += 15
        if re.search(r'[A-Z]', password):
            score += 15
        if re.search(r'\d', password):
            score += 15
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 15
        
        # Pattern penalties
        if re.search(r'(.)\1{2,}', password):  # Repeated characters
            score -= 10
        if re.search(r'(012|123|234|345|456|567|678|789|890|abc|bcd|cde)', password.lower()):
            score -= 10
        
        return min(100, max(0, score))
    
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """Sanitize user input to prevent XSS and injection"""
        if not text:
            return ""
        
        # Remove potential XSS
        for pattern in cls.XSS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # HTML encode special characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text.strip()
    
    @classmethod
    def check_sql_injection(cls, text: str) -> bool:
        """Check for SQL injection patterns"""
        if not text:
            return False
        
        text_lower = text.lower()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

# ================================
# 🚨 SECURITY MONITORING SERVICE
# ================================
class SecurityMonitoringService:
    """Security event monitoring and alerting"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.suspicious_patterns = defaultdict(int)
    
    async def log_security_event(
        self, 
        event_type: str, 
        user_id: Optional[str], 
        ip_address: str, 
        details: Dict[str, Any]
    ):
        """Log security events for monitoring"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details
        }
        
        # Store in Redis for real-time analysis
        self.redis_client.lpush('security_events', str(event))
        self.redis_client.ltrim('security_events', 0, 9999)  # Keep last 10k events
        
        # Log for permanent storage
        self.logger.warning(f"Security Event: {event}")
        
        # Check for suspicious patterns
        await self._check_suspicious_activity(event_type, ip_address, user_id)
    
    async def _check_suspicious_activity(self, event_type: str, ip_address: str, user_id: Optional[str]):
        """Check for suspicious activity patterns"""
        now = time.time()
        
        # Track failed login attempts per IP
        if event_type == 'failed_login':
            key = f"failed_logins:{ip_address}"
            count = self.redis_client.incr(key)
            if count == 1:
                self.redis_client.expire(key, 300)  # 5 minutes window
            
            if count > SecurityConfig.MAX_FAILED_ATTEMPTS:
                await self._trigger_security_alert(
                    'multiple_failed_logins',
                    f"Multiple failed login attempts from IP: {ip_address}"
                )
        
        # Track suspicious requests per user
        if user_id and event_type in ['sql_injection_attempt', 'xss_attempt']:
            key = f"suspicious_user:{user_id}"
            count = self.redis_client.incr(key)
            if count == 1:
                self.redis_client.expire(key, 3600)  # 1 hour window
            
            if count > 3:
                await self._trigger_security_alert(
                    'suspicious_user_activity',
                    f"Multiple security violations from user: {user_id}"
                )
    
    async def _trigger_security_alert(self, alert_type: str, message: str):
        """Trigger security alert"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'alert_type': alert_type,
            'message': message,
            'severity': 'high'
        }
        
        # Store alert
        self.redis_client.lpush('security_alerts', str(alert))
        
        # Log critical alert
        self.logger.critical(f"SECURITY ALERT: {alert}")
        
        # TODO: Send to alerting system (Slack, email, etc.)
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data"""
        events = self.redis_client.lrange('security_events', 0, 99)
        alerts = self.redis_client.lrange('security_alerts', 0, 9)
        
        return {
            'recent_events': [eval(event.decode()) for event in events],
            'recent_alerts': [eval(alert.decode()) for alert in alerts],
            'total_events': self.redis_client.llen('security_events'),
            'total_alerts': self.redis_client.llen('security_alerts')
        }

# ================================
# 🔐 SECURITY MIDDLEWARE
# ================================
class SecurityMiddleware:
    """Advanced security middleware for FastAPI"""
    
    def __init__(self, app, redis_client: redis.Redis):
        self.app = app
        self.rate_limit_service = RateLimitService(redis_client)
        self.security_monitor = SecurityMonitoringService(redis_client)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Add security headers
        response = await call_next(request)
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Log request for monitoring
        await self._log_request(request, client_ip)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else 'unknown'
    
    async def _log_request(self, request: Request, client_ip: str):
        """Log request for security monitoring"""
        # Check for suspicious patterns in URL
        if InputValidationService.check_sql_injection(str(request.url)):
            await self.security_monitor.log_security_event(
                'sql_injection_attempt',
                None,  # No user ID at this point
                client_ip,
                {'url': str(request.url), 'method': request.method}
            )

# ================================
# 🛡️ AUTHENTICATION DECORATORS
# ================================
def require_auth(required_roles: List[str] = None):
    """Decorator to require authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs or args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(status_code=500, detail="Request not found")
            
            # Check authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise HTTPException(status_code=401, detail="Missing authorization token")
            
            token = auth_header.split(' ')[1]
            jwt_service = JWTService()  # Should be injected in real implementation
            
            try:
                payload = jwt_service.verify_token(token)
                
                # Check required roles
                if required_roles:
                    user_roles = payload.get('roles', [])
                    if not any(role in user_roles for role in required_roles):
                        raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Add user info to request
                request.state.user = payload
                
                return await func(*args, **kwargs)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=401, detail="Invalid token")
        
        return wrapper
    return decorator

# ================================
# 🧪 SECURITY TESTING UTILITIES
# ================================
class SecurityTester:
    """Security testing and validation utilities"""
    
    @staticmethod
    def test_password_policy():
        """Test password policy enforcement"""
        test_cases = [
            ("weak", False),
            ("StrongPass123!", True),
            ("nouppercase123!", False),
            ("NOLOWERCASE123!", False),
            ("NoNumbers!", False),
            ("NoSpecialChars123", False),
            ("Aa1!", False),  # Too short
        ]
        
        for password, should_be_valid in test_cases:
            result = InputValidationService.validate_password(password)
            assert result['valid'] == should_be_valid, f"Password '{password}' validation failed"
    
    @staticmethod
    def test_input_sanitization():
        """Test input sanitization"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "<iframe src='javascript:alert(1)'></iframe>",
            "javascript:alert(1)",
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = InputValidationService.sanitize_input(malicious_input)
            assert '<script>' not in sanitized.lower()
            assert 'javascript:' not in sanitized.lower()
    
    @staticmethod
    def test_sql_injection_detection():
        """Test SQL injection detection"""
        injection_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--",
        ]
        
        for attempt in injection_attempts:
            assert InputValidationService.check_sql_injection(attempt) == True

# ================================
# 📊 SECURITY METRICS COLLECTOR
# ================================
class SecurityMetrics:
    """Collect and export security metrics"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        now = datetime.utcnow()
        today = now.strftime('%Y-%m-%d')
        
        return {
            'failed_logins_today': self._count_events_today('failed_login'),
            'security_alerts_today': self._count_events_today('security_alert'),
            'blocked_ips_count': len(self._get_blocked_ips()),
            'active_sessions': self._count_active_sessions(),
            'password_strength_distribution': self._get_password_strength_stats(),
            'top_attack_types': self._get_top_attack_types(),
            'security_score': self._calculate_security_score()
        }
    
    def _count_events_today(self, event_type: str) -> int:
        """Count security events for today"""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        key = f"security_events:{event_type}:{today}"
        return self.redis_client.get(key) or 0
    
    def _get_blocked_ips(self) -> List[str]:
        """Get list of currently blocked IPs"""
        blocked_ips = []
        for key in self.redis_client.scan_iter(match="blocked_ip:*"):
            ip = key.decode().split(':')[1]
            blocked_ips.append(ip)
        return blocked_ips
    
    def _count_active_sessions(self) -> int:
        """Count active user sessions"""
        return len(list(self.redis_client.scan_iter(match="token:*")))
    
    def _get_password_strength_stats(self) -> Dict[str, int]:
        """Get password strength distribution"""
        # This would need to be implemented based on user registration data
        return {
            'weak': 0,
            'medium': 0,
            'strong': 0
        }
    
    def _get_top_attack_types(self) -> Dict[str, int]:
        """Get top attack types from recent events"""
        events = self.redis_client.lrange('security_events', 0, 999)
        attack_counts = defaultdict(int)
        
        for event in events:
            try:
                event_data = eval(event.decode())
                event_type = event_data.get('event_type', 'unknown')
                if 'attack' in event_type or 'injection' in event_type:
                    attack_counts[event_type] += 1
            except:
                continue
        
        return dict(sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security health score (0-100)"""
        score = 100
        
        # Deduct points for recent security events
        recent_alerts = self.redis_client.llen('security_alerts')
        score -= min(50, recent_alerts * 5)
        
        # Deduct points for failed logins
        failed_logins = self._count_events_today('failed_login')
        score -= min(20, failed_logins * 2)
        
        return max(0, score)

# ================================
# 🔧 SECURITY INITIALIZATION
# ================================
def initialize_security_services(redis_client: redis.Redis) -> Dict[str, Any]:
    """Initialize all security services"""
    
    services = {
        'encryption': EncryptionService(),
        'jwt': JWTService(redis_client),
        'rate_limit': RateLimitService(redis_client),
        'monitoring': SecurityMonitoringService(redis_client),
        'metrics': SecurityMetrics(redis_client)
    }
    
    # Run security tests
    SecurityTester.test_password_policy()
    SecurityTester.test_input_sanitization()
    SecurityTester.test_sql_injection_detection()
    
    logging.info("🔐 Security services initialized successfully")
    
    return services

# ================================
# 📋 SECURITY HEALTH CHECK
# ================================
async def security_health_check(redis_client: redis.Redis) -> Dict[str, Any]:
    """Comprehensive security health check"""
    
    health_status = {
        'status': 'healthy',
        'checks': {},
        'recommendations': []
    }
    
    try:
        # Check Redis connection
        redis_client.ping()
        health_status['checks']['redis'] = 'healthy'
    except:
        health_status['checks']['redis'] = 'unhealthy'
        health_status['status'] = 'unhealthy'
        health_status['recommendations'].append('Check Redis connection')
    
    # Check JWT configuration
    if len(SecurityConfig.JWT_SECRET_KEY) < 32:
        health_status['checks']['jwt_secret_strength'] = 'weak'
        health_status['recommendations'].append('Use stronger JWT secret key (32+ characters)')
    else:
        health_status['checks']['jwt_secret_strength'] = 'strong'
    
    # Check encryption key
    try:
        Fernet(SecurityConfig.ENCRYPTION_KEY)
        health_status['checks']['encryption_key'] = 'valid'
    except:
        health_status['checks']['encryption_key'] = 'invalid'
        health_status['status'] = 'unhealthy'
        health_status['recommendations'].append('Fix encryption key configuration')
    
    # Get security metrics
    metrics = SecurityMetrics(redis_client)
    security_score = metrics._calculate_security_score()
    health_status['security_score'] = security_score
    
    if security_score < 70:
        health_status['status'] = 'warning'
        health_status['recommendations'].append('Review recent security events')
    
    return health_status
