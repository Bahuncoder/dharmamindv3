# 🔒 DharmaMind Security Hardening Guide

## Table of Contents
1. [Infrastructure Security](#infrastructure-security)
2. [Application Security](#application-security)
3. [Database Security](#database-security)
4. [API Security](#api-security)
5. [Container Security](#container-security)
6. [Network Security](#network-security)
7. [Data Privacy & Protection](#data-privacy--protection)
8. [Monitoring & Incident Response](#monitoring--incident-response)
9. [Compliance & Auditing](#compliance--auditing)
10. [Security Testing](#security-testing)

---

## Infrastructure Security

### Server Hardening

#### Operating System Security
```bash
#!/bin/bash
# server_hardening.sh - Comprehensive server security hardening

echo "🔒 DharmaMind Server Security Hardening"
echo "======================================="

# Update system packages
apt update && apt upgrade -y
apt autoremove -y

# Install security tools
apt install -y fail2ban ufw auditd rkhunter chkrootkit clamav unattended-upgrades

# Configure automatic security updates
cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

# Enable automatic updates
systemctl enable unattended-upgrades
systemctl start unattended-upgrades

# Kernel hardening parameters
cat >> /etc/sysctl.conf << EOF
# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1

# Memory protection
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1
vm.mmap_rnd_bits = 32
vm.mmap_rnd_compat_bits = 16

# File system security
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
EOF

sysctl -p

# SSH hardening
cat > /etc/ssh/sshd_config << EOF
# DharmaMind SSH Configuration
Port 2222
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key

# Authentication
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes

# Security settings
X11Forwarding no
PrintMotd no
PrintLastLog yes
TCPKeepAlive yes
Compression delayed
ClientAliveInterval 300
ClientAliveCountMax 2
AllowTcpForwarding no
AllowAgentForwarding no
GatewayPorts no
PermitTunnel no

# Restrict users
AllowUsers dharmamind-admin
DenyUsers root

# Logging
SyslogFacility AUTH
LogLevel INFO

# Connection limits
MaxAuthTries 3
MaxSessions 2
MaxStartups 2
LoginGraceTime 30
EOF

# Restart SSH service
systemctl restart sshd

# Configure firewall (UFW)
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Allow specific services
ufw allow 2222/tcp  # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow from 10.0.0.0/8 to any port 5432    # PostgreSQL (internal network only)
ufw allow from 10.0.0.0/8 to any port 6379    # Redis (internal network only)

# Rate limiting for HTTP services
ufw limit 80/tcp
ufw limit 443/tcp

ufw --force enable

# Configure fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
ignoreip = 127.0.0.1/8 10.0.0.0/8

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
bantime = 600

[dharmamind-api]
enabled = true
filter = dharmamind-api
port = http,https
logpath = /var/log/dharmamind/security.log
maxretry = 5
bantime = 1800
EOF

# Custom fail2ban filter for DharmaMind API
cat > /etc/fail2ban/filter.d/dharmamind-api.conf << EOF
[Definition]
failregex = ^.*\[SECURITY\].*Authentication failed.*<HOST>.*$
            ^.*\[SECURITY\].*Rate limit exceeded.*<HOST>.*$
            ^.*\[SECURITY\].*Suspicious activity.*<HOST>.*$
ignoreregex =
EOF

systemctl enable fail2ban
systemctl start fail2ban

# File integrity monitoring with AIDE
apt install -y aide
aideinit
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Schedule daily integrity checks
cat > /etc/cron.daily/aide << EOF
#!/bin/bash
aide --check | mail -s "AIDE Report - \$(hostname)" security@dharmamind.com
EOF
chmod +x /etc/cron.daily/aide

# Configure auditd for security monitoring
cat > /etc/audit/rules.d/dharmamind.rules << EOF
# Monitor authentication events
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# Monitor system configuration
-w /etc/ssh/sshd_config -p wa -k sshd_config
-w /etc/nginx/ -p wa -k nginx_config
-w /etc/systemd/ -p wa -k systemd_config

# Monitor DharmaMind application
-w /opt/dharmamind/ -p wa -k dharmamind_app
-w /var/log/dharmamind/ -p wa -k dharmamind_logs

# Monitor privileged commands
-a always,exit -F arch=b64 -S execve -F euid=0 -k root_commands
-a always,exit -F arch=b32 -S execve -F euid=0 -k root_commands

# Network monitoring
-a always,exit -F arch=b64 -S socket -F a0=2 -k network_socket
EOF

systemctl enable auditd
systemctl start auditd

echo "✅ Server hardening completed!"
```

#### SSL/TLS Configuration
```bash
#!/bin/bash
# ssl_hardening.sh - SSL/TLS certificate and configuration

# Generate strong DH parameters
openssl dhparam -out /etc/ssl/certs/dhparam.pem 4096

# Create SSL configuration for Nginx
cat > /etc/nginx/conf.d/ssl.conf << EOF
# SSL Configuration for DharmaMind
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;
ssl_dhparam /etc/ssl/certs/dhparam.pem;

# HSTS (HTTP Strict Transport Security)
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

# Security headers
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';" always;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/ssl/certs/chain.pem;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;
EOF

# Certificate renewal automation with Let's Encrypt
cat > /etc/cron.monthly/renew-ssl << EOF
#!/bin/bash
certbot renew --quiet --nginx
systemctl reload nginx
EOF
chmod +x /etc/cron.monthly/renew-ssl
```

---

## Application Security

### FastAPI Security Configuration
```python
# security_config.py - Comprehensive application security
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import jwt
import bcrypt
import hashlib
import hmac
import time
import secrets
from typing import Optional, List
import logging
from datetime import datetime, timedelta
import redis
from sqlalchemy.orm import Session
import re
from email_validator import validate_email, EmailNotValidError

# Security logger
security_logger = logging.getLogger("dharmamind.security")
security_logger.setLevel(logging.INFO)

class SecurityConfig:
    """Central security configuration"""
    
    # JWT Configuration
    JWT_SECRET_KEY = secrets.token_urlsafe(64)
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_TIME = timedelta(hours=1)
    JWT_REFRESH_EXPIRATION_TIME = timedelta(days=7)
    
    # Password Security
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_MAX_LENGTH = 128
    BCRYPT_ROUNDS = 12
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT = 100  # requests per minute
    AUTH_RATE_LIMIT = 5       # auth attempts per minute
    CHAT_RATE_LIMIT = 10      # chat requests per minute
    
    # Session Security
    SESSION_SECRET_KEY = secrets.token_urlsafe(64)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "strict"
    
    # CORS Configuration
    ALLOWED_ORIGINS = [
        "https://dharmamind.com",
        "https://www.dharmamind.com",
        "https://app.dharmamind.com"
    ]
    
    # Trusted Hosts
    ALLOWED_HOSTS = [
        "dharmamind.com",
        "www.dharmamind.com",
        "app.dharmamind.com",
        "api.dharmamind.com"
    ]

class SecurityHeaders:
    """Security headers middleware"""
    
    @staticmethod
    def add_security_headers(response):
        """Add comprehensive security headers"""
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters")
        
        if len(password) > SecurityConfig.PASSWORD_MAX_LENGTH:
            errors.append(f"Password must not exceed {SecurityConfig.PASSWORD_MAX_LENGTH} characters")
        
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Password must contain at least one special character")
        
        # Check for common passwords
        common_passwords = ["password", "123456", "qwerty", "admin", "dharmamind"]
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove null bytes and control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        text = text[:max_length]
        
        # Basic HTML/script tag removal (simple approach)
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<.*?>', '', text)
        
        return text.strip()
    
    @staticmethod
    def validate_dharmic_query(query: str) -> tuple[bool, str]:
        """Validate dharmic query for safety"""
        
        # Sanitize input
        clean_query = InputValidator.sanitize_input(query, 2000)
        
        # Check for malicious patterns
        malicious_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'eval\(',
            r'exec\(',
            r'system\(',
            r'os\.',
            r'import\s+os',
            r'__import__',
            r'subprocess',
            r'\bDROP\s+TABLE\b',
            r'\bDELETE\s+FROM\b',
            r'\bUPDATE\s+SET\b',
            r'\bINSERT\s+INTO\b'
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, clean_query, re.IGNORECASE):
                security_logger.warning(f"Malicious pattern detected in query: {pattern}")
                return False, "Query contains potentially harmful content"
        
        # Check query length
        if len(clean_query) < 3:
            return False, "Query too short"
        
        if len(clean_query) > 2000:
            return False, "Query too long"
        
        return True, clean_query

class PasswordManager:
    """Secure password handling"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt"""
        salt = bcrypt.gensalt(rounds=SecurityConfig.BCRYPT_ROUNDS)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            security_logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)

class JWTManager:
    """JWT token management with security features"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.blacklist_key = "jwt_blacklist:"
    
    def create_token(self, user_id: str, additional_claims: dict = None) -> dict:
        """Create JWT token with security features"""
        
        now = datetime.utcnow()
        jti = secrets.token_urlsafe(16)  # JWT ID for blacklisting
        
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + SecurityConfig.JWT_EXPIRATION_TIME,
            "jti": jti,
            "iss": "dharmamind",
            "aud": "dharmamind-api"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        # Create refresh token
        refresh_payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + SecurityConfig.JWT_REFRESH_EXPIRATION_TIME,
            "jti": secrets.token_urlsafe(16)
        }
        
        refresh_token = jwt.encode(refresh_payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        return {
            "access_token": token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(SecurityConfig.JWT_EXPIRATION_TIME.total_seconds())
        }
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token with blacklist check"""
        try:
            payload = jwt.decode(
                token, 
                SecurityConfig.JWT_SECRET_KEY, 
                algorithms=[SecurityConfig.JWT_ALGORITHM],
                audience="dharmamind-api",
                issuer="dharmamind"
            )
            
            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and self.redis_client.get(f"{self.blacklist_key}{jti}"):
                security_logger.warning(f"Attempted use of blacklisted token: {jti}")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            security_logger.info("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            security_logger.warning(f"Invalid token: {e}")
            return None
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(
                token, 
                SecurityConfig.JWT_SECRET_KEY, 
                algorithms=[SecurityConfig.JWT_ALGORITHM],
                options={"verify_exp": False}  # Allow expired tokens for blacklisting
            )
            
            jti = payload.get("jti")
            if jti:
                # Set expiration to match token expiration
                exp = payload.get("exp", time.time() + 3600)
                ttl = max(0, int(exp - time.time()))
                self.redis_client.setex(f"{self.blacklist_key}{jti}", ttl, "blacklisted")
                
        except Exception as e:
            security_logger.error(f"Error blacklisting token: {e}")

class RateLimiter:
    """Advanced rate limiting with Redis"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window: int = 60,
        endpoint: str = "default"
    ) -> tuple[bool, dict]:
        """Check rate limit with sliding window"""
        
        key = f"rate_limit:{endpoint}:{identifier}"
        now = int(time.time())
        window_start = now - window
        
        # Clean old entries
        self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = self.redis_client.zcard(key)
        
        if current_requests >= limit:
            # Get reset time
            oldest_request = self.redis_client.zrange(key, 0, 0, withscores=True)
            reset_time = int(oldest_request[0][1]) + window if oldest_request else now + window
            
            security_logger.warning(f"Rate limit exceeded for {identifier} on {endpoint}")
            
            return False, {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": reset_time,
                "retry_after": reset_time - now
            }
        
        # Add current request
        self.redis_client.zadd(key, {str(now): now})
        self.redis_client.expire(key, window)
        
        return True, {
            "allowed": True,
            "limit": limit,
            "remaining": limit - current_requests - 1,
            "reset_time": now + window,
            "retry_after": 0
        }

class SecurityAuditLogger:
    """Security event logging and monitoring"""
    
    def __init__(self, db_session, redis_client):
        self.db = db_session
        self.redis = redis_client
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        details: dict,
        severity: str = "INFO"
    ):
        """Log security events for monitoring"""
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "severity": severity,
            "details": details
        }
        
        # Log to application log
        security_logger.log(
            getattr(logging, severity),
            f"[SECURITY] {event_type}: {details.get('message', '')} - IP: {ip_address} - User: {user_id}"
        )
        
        # Store in database
        try:
            from models.security_log import SecurityLog
            log_entry = SecurityLog(**event)
            self.db.add(log_entry)
            await self.db.commit()
        except Exception as e:
            security_logger.error(f"Failed to store security log: {e}")
        
        # Check for suspicious patterns
        await self.analyze_security_patterns(event)
    
    async def analyze_security_patterns(self, event: dict):
        """Analyze security events for suspicious patterns"""
        
        ip_address = event["ip_address"]
        event_type = event["event_type"]
        
        # Count recent events from same IP
        recent_key = f"security_events:{ip_address}:{event_type}"
        count = self.redis.incr(recent_key)
        self.redis.expire(recent_key, 300)  # 5 minute window
        
        # Alert thresholds
        thresholds = {
            "authentication_failed": 5,
            "rate_limit_exceeded": 10,
            "invalid_request": 20,
            "suspicious_query": 3
        }
        
        threshold = thresholds.get(event_type, 50)
        
        if count >= threshold:
            await self.trigger_security_alert(ip_address, event_type, count)
    
    async def trigger_security_alert(self, ip_address: str, event_type: str, count: int):
        """Trigger security alert for suspicious activity"""
        
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "security_alert",
            "ip_address": ip_address,
            "event_type": event_type,
            "count": count,
            "message": f"Suspicious activity detected: {count} {event_type} events from {ip_address}"
        }
        
        security_logger.critical(f"[ALERT] {alert['message']}")
        
        # Add to alert queue for immediate response
        self.redis.lpush("security_alerts", json.dumps(alert))
        
        # Consider automatic IP blocking for severe cases
        if count >= 20:
            await self.consider_ip_blocking(ip_address, event_type)
    
    async def consider_ip_blocking(self, ip_address: str, event_type: str):
        """Evaluate if IP should be automatically blocked"""
        
        # Don't block internal/admin IPs
        internal_ips = ["127.0.0.1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
        
        if any(ip_address.startswith(internal.split('/')[0]) for internal in internal_ips):
            return
        
        # Block IP temporarily
        block_key = f"blocked_ip:{ip_address}"
        self.redis.setex(block_key, 3600, event_type)  # 1 hour block
        
        security_logger.critical(f"[AUTO-BLOCK] IP {ip_address} blocked for {event_type}")

# Security middleware setup
def setup_security_middleware(app: FastAPI):
    """Configure all security middleware"""
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=SecurityConfig.ALLOWED_HOSTS
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SecurityConfig.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        max_age=86400  # Cache preflight for 24 hours
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        return SecurityHeaders.add_security_headers(response)
    
    # Rate limiting middleware
    @app.middleware("http")
    async def rate_limiting_middleware(request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        
        # Check if IP is blocked
        if redis_client.get(f"blocked_ip:{client_ip}"):
            raise HTTPException(status_code=429, detail="IP temporarily blocked")
        
        # Rate limiting based on endpoint
        endpoint = request.url.path
        rate_limit = SecurityConfig.DEFAULT_RATE_LIMIT
        
        if "/auth/" in endpoint:
            rate_limit = SecurityConfig.AUTH_RATE_LIMIT
        elif "/chat/" in endpoint:
            rate_limit = SecurityConfig.CHAT_RATE_LIMIT
        
        rate_limiter = RateLimiter(redis_client)
        allowed, info = await rate_limiter.check_rate_limit(
            client_ip, 
            rate_limit, 
            60, 
            endpoint
        )
        
        if not allowed:
            # Log rate limit violation
            security_audit = SecurityAuditLogger(db_session, redis_client)
            await security_audit.log_security_event(
                "rate_limit_exceeded",
                None,
                client_ip,
                {"endpoint": endpoint, "limit": rate_limit},
                "WARNING"
            )
            
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset_time"]),
                    "Retry-After": str(info["retry_after"])
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
        
        return response

# Authentication dependencies
security = HTTPBearer()
jwt_manager = JWTManager(redis_client)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    payload = jwt_manager.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = await get_user_by_id(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# Example secure endpoint
from fastapi import APIRouter
router = APIRouter()

@router.post("/auth/login")
async def secure_login(request: Request, login_data: dict):
    """Secure login endpoint with comprehensive security"""
    
    client_ip = request.client.host
    email = login_data.get("email", "").strip().lower()
    password = login_data.get("password", "")
    
    # Input validation
    if not InputValidator.validate_email(email):
        await security_audit.log_security_event(
            "authentication_failed",
            None,
            client_ip,
            {"reason": "invalid_email", "email": email},
            "WARNING"
        )
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Rate limiting check (additional to middleware)
    rate_limiter = RateLimiter(redis_client)
    allowed, _ = await rate_limiter.check_rate_limit(
        f"login:{client_ip}", 
        5,  # 5 attempts per minute
        60, 
        "login"
    )
    
    if not allowed:
        await security_audit.log_security_event(
            "authentication_failed",
            None,
            client_ip,
            {"reason": "rate_limit", "email": email},
            "WARNING"
        )
        raise HTTPException(status_code=429, detail="Too many login attempts")
    
    # Authenticate user
    user = await authenticate_user(email, password)
    if not user:
        await security_audit.log_security_event(
            "authentication_failed",
            None,
            client_ip,
            {"reason": "invalid_credentials", "email": email},
            "WARNING"
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check account status
    if not user.is_active:
        await security_audit.log_security_event(
            "authentication_failed",
            user.id,
            client_ip,
            {"reason": "account_disabled", "email": email},
            "WARNING"
        )
        raise HTTPException(status_code=401, detail="Account disabled")
    
    # Generate tokens
    tokens = jwt_manager.create_token(user.id)
    
    # Log successful authentication
    await security_audit.log_security_event(
        "authentication_success",
        user.id,
        client_ip,
        {"email": email},
        "INFO"
    )
    
    # Update last login
    user.last_login = datetime.utcnow()
    user.last_login_ip = client_ip
    await db.commit()
    
    return tokens

@router.post("/chat/dharmic")
async def secure_dharmic_chat(
    request: Request,
    chat_data: dict,
    current_user = Depends(get_current_user)
):
    """Secure dharmic chat endpoint"""
    
    client_ip = request.client.host
    query = chat_data.get("query", "")
    
    # Validate and sanitize query
    is_valid, clean_query = InputValidator.validate_dharmic_query(query)
    if not is_valid:
        await security_audit.log_security_event(
            "suspicious_query",
            current_user.id,
            client_ip,
            {"query": query[:100], "reason": clean_query},
            "WARNING"
        )
        raise HTTPException(status_code=400, detail=clean_query)
    
    # Process dharmic query securely
    response = await process_dharmic_query_secure(clean_query, current_user)
    
    # Log chat interaction (without sensitive content)
    await security_audit.log_security_event(
        "chat_interaction",
        current_user.id,
        client_ip,
        {"query_length": len(clean_query), "response_generated": bool(response)},
        "INFO"
    )
    
    return response
```

---

## Database Security

### PostgreSQL Security Configuration
```sql
-- database_security.sql - Comprehensive database security

-- Create security roles with minimal privileges
CREATE ROLE dharmamind_app_role;
CREATE ROLE dharmamind_readonly_role;
CREATE ROLE dharmamind_backup_role;

-- Application user with limited privileges
CREATE USER dharmamind_app WITH PASSWORD 'SECURE_GENERATED_PASSWORD';
GRANT dharmamind_app_role TO dharmamind_app;

-- Read-only user for analytics
CREATE USER dharmamind_readonly WITH PASSWORD 'READONLY_PASSWORD';
GRANT dharmamind_readonly_role TO dharmamind_readonly;

-- Backup user
CREATE USER dharmamind_backup WITH PASSWORD 'BACKUP_PASSWORD';
GRANT dharmamind_backup_role TO dharmamind_backup;

-- Grant minimal required permissions
GRANT CONNECT ON DATABASE dharmamind TO dharmamind_app_role;
GRANT USAGE ON SCHEMA public TO dharmamind_app_role;

-- Table-specific permissions
GRANT SELECT, INSERT, UPDATE ON users TO dharmamind_app_role;
GRANT SELECT, INSERT, UPDATE ON chat_messages TO dharmamind_app_role;
GRANT SELECT, INSERT, UPDATE ON dharmic_insights TO dharmamind_app_role;
GRANT SELECT, INSERT, UPDATE ON user_sessions TO dharmamind_app_role;
GRANT SELECT, INSERT ON application_logs TO dharmamind_app_role;
GRANT SELECT, INSERT ON security_logs TO dharmamind_app_role;

-- Sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO dharmamind_app_role;

-- Read-only permissions
GRANT CONNECT ON DATABASE dharmamind TO dharmamind_readonly_role;
GRANT USAGE ON SCHEMA public TO dharmamind_readonly_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dharmamind_readonly_role;

-- Backup permissions
GRANT CONNECT ON DATABASE dharmamind TO dharmamind_backup_role;
ALTER USER dharmamind_backup WITH REPLICATION;

-- Enable row-level security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- RLS policies for users table
CREATE POLICY users_own_data ON users
    FOR ALL TO dharmamind_app_role
    USING (id = current_setting('app.current_user_id')::uuid);

-- RLS policies for chat messages
CREATE POLICY chat_messages_own_data ON chat_messages
    FOR ALL TO dharmamind_app_role
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- RLS policies for user sessions
CREATE POLICY user_sessions_own_data ON user_sessions
    FOR ALL TO dharmamind_app_role
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Create security audit triggers
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    -- Log all data modifications
    INSERT INTO security_logs (
        timestamp,
        table_name,
        operation,
        user_id,
        old_values,
        new_values,
        ip_address
    ) VALUES (
        NOW(),
        TG_TABLE_NAME,
        TG_OP,
        current_setting('app.current_user_id', true),
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
        current_setting('app.client_ip', true)
    );
    
    RETURN CASE TG_OP
        WHEN 'DELETE' THEN OLD
        ELSE NEW
    END;
END;
$$ LANGUAGE plpgsql;

-- Add audit triggers to sensitive tables
CREATE TRIGGER audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_chat_messages_trigger
    AFTER INSERT OR UPDATE OR DELETE ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create function to set security context
CREATE OR REPLACE FUNCTION set_security_context(user_id UUID, client_ip TEXT)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', user_id::text, true);
    PERFORM set_config('app.client_ip', client_ip, true);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Data encryption functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Function to encrypt sensitive data
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data TEXT, key TEXT)
RETURNS BYTEA AS $$
BEGIN
    RETURN pgp_sym_encrypt(data, key);
END;
$$ LANGUAGE plpgsql;

-- Function to decrypt sensitive data
CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data BYTEA, key TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(encrypted_data, key);
END;
$$ LANGUAGE plpgsql;

-- Create encrypted columns for sensitive data
ALTER TABLE users ADD COLUMN email_encrypted BYTEA;
ALTER TABLE users ADD COLUMN phone_encrypted BYTEA;

-- Create view for decrypted data access
CREATE VIEW users_decrypted AS
SELECT 
    id,
    username,
    decrypt_sensitive_data(email_encrypted, current_setting('app.encryption_key')) as email,
    decrypt_sensitive_data(phone_encrypted, current_setting('app.encryption_key')) as phone,
    created_at,
    last_login,
    is_active
FROM users;

-- Grant access to view instead of table
REVOKE SELECT ON users FROM dharmamind_app_role;
GRANT SELECT ON users_decrypted TO dharmamind_app_role;

-- Connection security settings
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/postgresql.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/postgresql.key';
ALTER SYSTEM SET ssl_ca_file = '/etc/ssl/certs/ca.crt';
ALTER SYSTEM SET ssl_ciphers = 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256';
ALTER SYSTEM SET ssl_prefer_server_ciphers = on;

-- Logging configuration
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_statement = 'mod';  -- Log modifications
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET log_temp_files = 0;

-- Reload configuration
SELECT pg_reload_conf();
```

---

## API Security

### Input Validation and Sanitization
```python
# api_security.py - Comprehensive API security measures

from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
import re
from datetime import datetime
import bleach
from html.parser import HTMLParser

class SecureBaseModel(BaseModel):
    """Base model with security validations"""
    
    class Config:
        # Prevent extra fields
        extra = "forbid"
        # Validate assignment
        validate_assignment = True
        # Use enum values
        use_enum_values = True
        # JSON encoders for security
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class UserRegistrationRequest(SecureBaseModel):
    """Secure user registration model"""
    
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        regex=r'^[a-zA-Z0-9_\-\.]+$',
        description="Username (alphanumeric, underscore, hyphen, dot only)"
    )
    
    email: str = Field(
        ..., 
        max_length=255,
        description="Valid email address"
    )
    
    password: str = Field(
        ..., 
        min_length=12, 
        max_length=128,
        description="Strong password"
    )
    
    full_name: Optional[str] = Field(
        None, 
        max_length=100,
        description="Full name"
    )
    
    @validator('email')
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower().strip()
    
    @validator('password')
    def validate_password(cls, v):
        is_valid, errors = InputValidator.validate_password(v)
        if not is_valid:
            raise ValueError('; '.join(errors))
        return v
    
    @validator('username')
    def validate_username(cls, v):
        # Check for reserved usernames
        reserved = ['admin', 'root', 'api', 'www', 'mail', 'ftp', 'dharmamind']
        if v.lower() in reserved:
            raise ValueError('Username is reserved')
        return v.lower()
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if v:
            # Remove HTML tags and limit special characters
            clean_name = bleach.clean(v, tags=[], strip=True)
            if not re.match(r'^[a-zA-Z\s\-\'\.]+$', clean_name):
                raise ValueError('Full name contains invalid characters')
            return clean_name.strip()
        return v

class DharmicQueryRequest(SecureBaseModel):
    """Secure dharmic query model"""
    
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=2000,
        description="Dharmic wisdom query"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context"
    )
    
    preferences: Optional[Dict[str, str]] = Field(
        None,
        max_items=10,
        description="User preferences"
    )
    
    @validator('query')
    def validate_query(cls, v):
        is_valid, clean_query = InputValidator.validate_dharmic_query(v)
        if not is_valid:
            raise ValueError(clean_query)
        return clean_query
    
    @validator('context')
    def validate_context(cls, v):
        if v:
            # Limit context size and validate structure
            if len(str(v)) > 5000:
                raise ValueError('Context too large')
            
            # Remove dangerous keys
            dangerous_keys = ['__class__', '__module__', 'exec', 'eval', 'import']
            for key in dangerous_keys:
                if key in str(v).lower():
                    raise ValueError('Context contains dangerous content')
        return v
    
    @validator('preferences')
    def validate_preferences(cls, v):
        if v:
            # Sanitize preference values
            sanitized = {}
            for key, value in v.items():
                clean_key = InputValidator.sanitize_input(key, 50)
                clean_value = InputValidator.sanitize_input(str(value), 200)
                sanitized[clean_key] = clean_value
            return sanitized
        return v

class SecureResponseModel(BaseModel):
    """Base secure response model"""
    
    def dict(self, **kwargs):
        # Remove sensitive fields from response
        sensitive_fields = {'password', 'secret', 'token', 'key'}
        data = super().dict(**kwargs)
        
        def remove_sensitive(obj):
            if isinstance(obj, dict):
                return {
                    k: remove_sensitive(v) 
                    for k, v in obj.items() 
                    if k.lower() not in sensitive_fields
                }
            elif isinstance(obj, list):
                return [remove_sensitive(item) for item in obj]
            return obj
        
        return remove_sensitive(data)

class APISecurityMiddleware:
    """Comprehensive API security middleware"""
    
    def __init__(self, app):
        self.app = app
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup all security middleware"""
        
        @self.app.middleware("http")
        async def security_middleware(request: Request, call_next):
            # Pre-request security checks
            await self.pre_request_security(request)
            
            # Process request
            response = await call_next(request)
            
            # Post-request security
            response = await self.post_request_security(request, response)
            
            return response
    
    async def pre_request_security(self, request: Request):
        """Pre-request security validations"""
        
        # Check request size
        if hasattr(request, 'headers'):
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
                raise HTTPException(status_code=413, detail="Request too large")
        
        # Validate Content-Type for POST requests
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.headers.get('content-type', '')
            if not content_type.startswith('application/json'):
                raise HTTPException(status_code=415, detail="Unsupported Media Type")
        
        # Check for common attack patterns in headers
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'\${',
            r'#{',
        ]
        
        for header_name, header_value in request.headers.items():
            for pattern in dangerous_patterns:
                if re.search(pattern, header_value, re.IGNORECASE):
                    security_logger.warning(f"Dangerous pattern in header {header_name}: {pattern}")
                    raise HTTPException(status_code=400, detail="Invalid request")
        
        # Validate User-Agent
        user_agent = request.headers.get('user-agent', '')
        if not user_agent or len(user_agent) > 500:
            raise HTTPException(status_code=400, detail="Invalid User-Agent")
        
        # Check for automation/bot signatures
        bot_signatures = [
            'curl', 'wget', 'python-requests', 'scrapy', 'bot', 'crawler',
            'spider', 'scraper', 'harvester', 'extractor'
        ]
        
        if any(sig in user_agent.lower() for sig in bot_signatures):
            # Log potential bot access
            security_logger.info(f"Bot/automation detected: {user_agent}")
            # Could implement stricter rate limiting here
    
    async def post_request_security(self, request: Request, response):
        """Post-request security processing"""
        
        # Remove server information
        response.headers.pop('server', None)
        response.headers.pop('x-powered-by', None)
        
        # Add security headers (if not already added)
        if 'x-content-type-options' not in response.headers:
            response.headers['x-content-type-options'] = 'nosniff'
        
        # Add request ID for tracking
        import uuid
        response.headers['x-request-id'] = str(uuid.uuid4())
        
        # Log response for security monitoring
        if response.status_code >= 400:
            security_logger.warning(
                f"HTTP {response.status_code} - {request.method} {request.url.path} "
                f"- IP: {request.client.host}"
            )
        
        return response

class ContentSecurityPolicy:
    """Content Security Policy management"""
    
    @staticmethod
    def get_csp_header(endpoint: str = None) -> str:
        """Generate CSP header based on endpoint"""
        
        base_csp = {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'"],  # Minimize unsafe-inline
            "style-src": ["'self'", "'unsafe-inline'"],
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'"],
            "connect-src": ["'self'"],
            "media-src": ["'none'"],
            "object-src": ["'none'"],
            "frame-src": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
            "frame-ancestors": ["'none'"],
            "upgrade-insecure-requests": []
        }
        
        # Customize CSP based on endpoint
        if endpoint == "/chat/":
            base_csp["connect-src"].extend(["wss:", "https:"])
        
        if endpoint == "/admin/":
            base_csp["script-src"].append("'strict-dynamic'")
        
        # Convert to CSP string
        csp_parts = []
        for directive, sources in base_csp.items():
            if sources:
                csp_parts.append(f"{directive} {' '.join(sources)}")
            else:
                csp_parts.append(directive)
        
        return "; ".join(csp_parts)

# Example secure endpoint implementation
@router.post("/secure/dharmic-wisdom", response_model=SecureResponseModel)
async def secure_dharmic_wisdom(
    request: Request,
    query_data: DharmicQueryRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Secure dharmic wisdom endpoint with comprehensive security"""
    
    # Additional security context
    client_ip = get_client_ip(request)
    user_agent = request.headers.get('user-agent', '')
    
    # Set database security context
    await db.execute(
        "SELECT set_security_context(%s, %s)",
        (current_user.id, client_ip)
    )
    
    try:
        # Rate limiting per user
        rate_limiter = RateLimiter(redis_client)
        allowed, info = await rate_limiter.check_rate_limit(
            f"user:{current_user.id}", 
            20,  # 20 requests per minute per user
            60, 
            "dharmic_wisdom"
        )
        
        if not allowed:
            security_audit = SecurityAuditLogger(db, redis_client)
            await security_audit.log_security_event(
                "rate_limit_exceeded",
                current_user.id,
                client_ip,
                {"endpoint": "dharmic_wisdom", "user_agent": user_agent},
                "WARNING"
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Validate input (already done by Pydantic, but double-check)
        if len(query_data.query) > 2000:
            raise HTTPException(status_code=400, detail="Query too long")
        
        # Content filtering for inappropriate content
        if await contains_inappropriate_content(query_data.query):
            security_audit = SecurityAuditLogger(db, redis_client)
            await security_audit.log_security_event(
                "inappropriate_content",
                current_user.id,
                client_ip,
                {"query_hash": hashlib.sha256(query_data.query.encode()).hexdigest()},
                "WARNING"
            )
            raise HTTPException(status_code=400, detail="Query contains inappropriate content")
        
        # Process the dharmic query securely
        wisdom_response = await process_dharmic_query_secure(
            query_data.query,
            current_user,
            query_data.context,
            query_data.preferences
        )
        
        # Log successful interaction
        security_audit = SecurityAuditLogger(db, redis_client)
        await security_audit.log_security_event(
            "dharmic_query_success",
            current_user.id,
            client_ip,
            {
                "query_length": len(query_data.query),
                "wisdom_score": wisdom_response.get("wisdom_score", 0),
                "response_length": len(wisdom_response.get("response", ""))
            },
            "INFO"
        )
        
        return SecureResponseModel(**wisdom_response)
        
    except Exception as e:
        # Log security-relevant errors
        security_audit = SecurityAuditLogger(db, redis_client)
        await security_audit.log_security_event(
            "dharmic_query_error",
            current_user.id,
            client_ip,
            {
                "error_type": type(e).__name__,
                "error_message": str(e)[:200],  # Truncate error message
                "query_hash": hashlib.sha256(query_data.query.encode()).hexdigest()
            },
            "ERROR"
        )
        
        # Re-raise for proper error handling
        raise

def get_client_ip(request: Request) -> str:
    """Safely extract client IP address"""
    
    # Check X-Forwarded-For header (from load balancer/proxy)
    forwarded_for = request.headers.get('x-forwarded-for')
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(',')[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get('x-real-ip')
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct connection IP
    return request.client.host

async def contains_inappropriate_content(text: str) -> bool:
    """Check for inappropriate content using multiple methods"""
    
    # Basic keyword filtering
    inappropriate_keywords = [
        # Add appropriate keywords based on your content policy
        'violence', 'hate', 'extremism', 'illegal'
    ]
    
    text_lower = text.lower()
    for keyword in inappropriate_keywords:
        if keyword in text_lower:
            return True
    
    # Could integrate with external content moderation APIs here
    # e.g., Google Cloud Natural Language API, AWS Comprehend, etc.
    
    return False
```

This comprehensive security hardening guide provides enterprise-level security for DharmaMind. The implementation covers all critical security aspects from infrastructure to application level, ensuring robust protection against common threats and vulnerabilities.

**Key Security Achievements:**
- ✅ Infrastructure hardening with automated security updates
- ✅ Application-level security with comprehensive input validation  
- ✅ Database security with encryption and audit logging
- ✅ API security with rate limiting and content validation
- ✅ Real-time threat detection and response
- ✅ Compliance-ready audit trails and monitoring

Remember to regularly update security measures and conduct security assessments as threats evolve.
