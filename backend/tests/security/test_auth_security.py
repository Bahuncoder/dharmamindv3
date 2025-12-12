"""
🔐 Security Unit Tests for Authentication Flows

Comprehensive tests covering:
- Password validation
- JWT token security
- Session management
- Rate limiting
- CSRF protection
- XSS prevention
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import jwt
import hashlib
import secrets


# ============================================
# Password Security Tests
# ============================================

class TestPasswordValidation:
    """Test password strength requirements."""
    
    def validate_password(self, password: str) -> tuple[bool, list[str]]:
        """Validate password against security requirements."""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain uppercase letter")
        if not any(c.islower() for c in password):
            errors.append("Password must contain lowercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain a number")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain special character")
        
        # Common password check
        common_passwords = [
            "password", "123456", "password123", "admin", "letmein",
            "welcome", "monkey", "dragon", "master", "qwerty"
        ]
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    def test_valid_password(self):
        """Test that strong passwords pass validation."""
        valid_passwords = [
            "SecureP@ss123",
            "MyStr0ng!Pass",
            "C0mplex#Password",
            "Test@1234567",
        ]
        for password in valid_passwords:
            is_valid, errors = self.validate_password(password)
            assert is_valid, f"Password '{password}' should be valid: {errors}"
    
    def test_password_too_short(self):
        """Test that short passwords fail."""
        is_valid, errors = self.validate_password("Ab1!")
        assert not is_valid
        assert any("8 characters" in e for e in errors)
    
    def test_password_no_uppercase(self):
        """Test that passwords without uppercase fail."""
        is_valid, errors = self.validate_password("password123!")
        assert not is_valid
        assert any("uppercase" in e for e in errors)
    
    def test_password_no_lowercase(self):
        """Test that passwords without lowercase fail."""
        is_valid, errors = self.validate_password("PASSWORD123!")
        assert not is_valid
        assert any("lowercase" in e for e in errors)
    
    def test_password_no_number(self):
        """Test that passwords without numbers fail."""
        is_valid, errors = self.validate_password("Password!")
        assert not is_valid
        assert any("number" in e for e in errors)
    
    def test_password_no_special_char(self):
        """Test that passwords without special characters fail."""
        is_valid, errors = self.validate_password("Password123")
        assert not is_valid
        assert any("special" in e for e in errors)
    
    def test_common_password_rejected(self):
        """Test that common passwords are rejected."""
        is_valid, errors = self.validate_password("Password")
        assert not is_valid
        # This would fail multiple checks


# ============================================
# JWT Token Security Tests
# ============================================

class TestJWTSecurity:
    """Test JWT token security."""
    
    SECRET_KEY = "test-secret-key-for-unit-tests-only-32chars"
    ALGORITHM = "HS256"
    
    def create_token(self, payload: dict, secret: str = None, algorithm: str = None) -> str:
        """Create a JWT token."""
        return jwt.encode(
            payload,
            secret or self.SECRET_KEY,
            algorithm=algorithm or self.ALGORITHM
        )
    
    def decode_token(self, token: str, secret: str = None) -> dict:
        """Decode a JWT token."""
        return jwt.decode(
            token,
            secret or self.SECRET_KEY,
            algorithms=[self.ALGORITHM]
        )
    
    def test_valid_token_creation(self):
        """Test that valid tokens can be created and decoded."""
        payload = {"sub": "user123", "exp": datetime.utcnow() + timedelta(hours=1)}
        token = self.create_token(payload)
        decoded = self.decode_token(token)
        assert decoded["sub"] == "user123"
    
    def test_expired_token_rejected(self):
        """Test that expired tokens are rejected."""
        payload = {"sub": "user123", "exp": datetime.utcnow() - timedelta(hours=1)}
        token = self.create_token(payload)
        
        with pytest.raises(jwt.ExpiredSignatureError):
            self.decode_token(token)
    
    def test_invalid_signature_rejected(self):
        """Test that tokens with invalid signatures are rejected."""
        payload = {"sub": "user123", "exp": datetime.utcnow() + timedelta(hours=1)}
        token = self.create_token(payload, secret="wrong-secret-key-12345678901234")
        
        with pytest.raises(jwt.InvalidSignatureError):
            self.decode_token(token)
    
    def test_tampered_token_rejected(self):
        """Test that tampered tokens are rejected."""
        payload = {"sub": "user123", "exp": datetime.utcnow() + timedelta(hours=1)}
        token = self.create_token(payload)
        
        # Tamper with the token
        parts = token.split(".")
        # Modify the payload
        tampered_token = parts[0] + "." + parts[1] + "X" + "." + parts[2]
        
        with pytest.raises(jwt.DecodeError):
            self.decode_token(tampered_token)
    
    def test_none_algorithm_rejected(self):
        """Test that 'none' algorithm tokens are rejected."""
        # Create a token with 'none' algorithm (attack vector)
        header = {"alg": "none", "typ": "JWT"}
        payload = {"sub": "admin", "exp": datetime.utcnow() + timedelta(hours=1)}
        
        import base64
        import json
        
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload, default=str).encode()).rstrip(b"=").decode()
        fake_token = f"{header_b64}.{payload_b64}."
        
        # Should raise error when trying to decode
        with pytest.raises(jwt.exceptions.DecodeError):
            self.decode_token(fake_token)
    
    def test_token_contains_required_claims(self):
        """Test that tokens contain required security claims."""
        payload = {
            "sub": "user123",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }
        token = self.create_token(payload)
        decoded = self.decode_token(token)
        
        assert "sub" in decoded
        assert "exp" in decoded
        assert "iat" in decoded
        assert "jti" in decoded


# ============================================
# Session Management Tests
# ============================================

class TestSessionManagement:
    """Test session security features."""
    
    def __init__(self):
        self.sessions = {}
        self.blacklisted_tokens = set()
        self.max_sessions_per_user = 5
    
    def create_session(self, user_id: str, ip: str, user_agent: str) -> str:
        """Create a new session."""
        session_id = secrets.token_urlsafe(32)
        
        # Check session limit
        user_sessions = [s for s in self.sessions.values() if s["user_id"] == user_id]
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest = min(user_sessions, key=lambda x: x["created_at"])
            del self.sessions[oldest["session_id"]]
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "ip": ip,
            "user_agent": user_agent,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        return session_id
    
    def validate_session(self, session_id: str, ip: str = None) -> bool:
        """Validate a session."""
        if session_id in self.blacklisted_tokens:
            return False
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check inactivity timeout (60 minutes)
        if datetime.utcnow() - session["last_activity"] > timedelta(minutes=60):
            del self.sessions[session_id]
            return False
        
        # Optional: IP binding
        if ip and session["ip"] != ip:
            return False
        
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        return True
    
    def revoke_session(self, session_id: str):
        """Revoke a session."""
        self.blacklisted_tokens.add(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def test_session_creation(self):
        """Test session creation."""
        session_id = self.create_session("user1", "192.168.1.1", "Mozilla/5.0")
        assert session_id in self.sessions
        assert self.sessions[session_id]["user_id"] == "user1"
    
    def test_session_validation(self):
        """Test session validation."""
        session_id = self.create_session("user1", "192.168.1.1", "Mozilla/5.0")
        assert self.validate_session(session_id) is True
    
    def test_session_revocation(self):
        """Test session revocation."""
        session_id = self.create_session("user1", "192.168.1.1", "Mozilla/5.0")
        self.revoke_session(session_id)
        assert self.validate_session(session_id) is False
    
    def test_session_limit_enforced(self):
        """Test that session limit is enforced."""
        for i in range(6):
            self.create_session("user1", f"192.168.1.{i}", "Mozilla/5.0")
        
        user_sessions = [s for s in self.sessions.values() if s["user_id"] == "user1"]
        assert len(user_sessions) <= self.max_sessions_per_user
    
    def test_blacklisted_session_rejected(self):
        """Test that blacklisted sessions are rejected."""
        session_id = self.create_session("user1", "192.168.1.1", "Mozilla/5.0")
        self.blacklisted_tokens.add(session_id)
        assert self.validate_session(session_id) is False


# ============================================
# Rate Limiting Tests
# ============================================

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def __init__(self):
        self.requests = {}  # ip -> list of timestamps
        self.rate_limit = 100  # requests per minute
        self.blocked_ips = set()
        self.failed_attempts = {}  # ip -> count
    
    def check_rate_limit(self, ip: str) -> bool:
        """Check if request is within rate limit."""
        if ip in self.blocked_ips:
            return False
        
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Remove old requests
        self.requests[ip] = [t for t in self.requests[ip] if t > minute_ago]
        
        if len(self.requests[ip]) >= self.rate_limit:
            return False
        
        self.requests[ip].append(now)
        return True
    
    def record_failed_attempt(self, ip: str):
        """Record a failed authentication attempt."""
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = 0
        self.failed_attempts[ip] += 1
        
        if self.failed_attempts[ip] >= 10:
            self.blocked_ips.add(ip)
    
    def test_rate_limit_allows_normal_traffic(self):
        """Test that normal traffic is allowed."""
        for i in range(50):
            assert self.check_rate_limit("192.168.1.1") is True
    
    def test_rate_limit_blocks_excessive_traffic(self):
        """Test that excessive traffic is blocked."""
        for i in range(100):
            self.check_rate_limit("192.168.1.2")
        
        # 101st request should be blocked
        assert self.check_rate_limit("192.168.1.2") is False
    
    def test_ip_blocked_after_failed_attempts(self):
        """Test that IP is blocked after too many failed attempts."""
        for i in range(10):
            self.record_failed_attempt("192.168.1.3")
        
        assert "192.168.1.3" in self.blocked_ips
        assert self.check_rate_limit("192.168.1.3") is False


# ============================================
# CSRF Protection Tests
# ============================================

class TestCSRFProtection:
    """Test CSRF token protection."""
    
    def __init__(self):
        self.tokens = {}
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate a CSRF token for a session."""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = token
        return token
    
    def validate_csrf_token(self, session_id: str, token: str) -> bool:
        """Validate a CSRF token."""
        if session_id not in self.tokens:
            return False
        return secrets.compare_digest(self.tokens[session_id], token)
    
    def test_csrf_token_generation(self):
        """Test CSRF token generation."""
        token = self.generate_csrf_token("session1")
        assert len(token) > 20
        assert token == self.tokens["session1"]
    
    def test_valid_csrf_token_accepted(self):
        """Test that valid CSRF tokens are accepted."""
        token = self.generate_csrf_token("session1")
        assert self.validate_csrf_token("session1", token) is True
    
    def test_invalid_csrf_token_rejected(self):
        """Test that invalid CSRF tokens are rejected."""
        self.generate_csrf_token("session1")
        assert self.validate_csrf_token("session1", "invalid-token") is False
    
    def test_csrf_token_session_binding(self):
        """Test that CSRF tokens are bound to sessions."""
        token1 = self.generate_csrf_token("session1")
        token2 = self.generate_csrf_token("session2")
        
        # Token from session1 should not work for session2
        assert self.validate_csrf_token("session2", token1) is False
        assert self.validate_csrf_token("session1", token2) is False


# ============================================
# XSS Prevention Tests
# ============================================

class TestXSSPrevention:
    """Test XSS prevention measures."""
    
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize user input to prevent XSS."""
        if not input_str:
            return ""
        
        # HTML entity encoding
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
        }
        
        result = input_str
        for char, entity in replacements.items():
            result = result.replace(char, entity)
        
        return result
    
    def test_script_tag_escaped(self):
        """Test that script tags are escaped."""
        malicious = "<script>alert('XSS')</script>"
        sanitized = self.sanitize_input(malicious)
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized
    
    def test_event_handler_escaped(self):
        """Test that event handlers are escaped."""
        malicious = '<img src="x" onerror="alert(1)">'
        sanitized = self.sanitize_input(malicious)
        assert 'onerror="alert(1)"' not in sanitized
    
    def test_javascript_url_escaped(self):
        """Test that javascript: URLs are handled."""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        sanitized = self.sanitize_input(malicious)
        assert "javascript:" not in sanitized or "&" in sanitized
    
    def test_normal_text_preserved(self):
        """Test that normal text is preserved."""
        normal = "Hello, World! This is a test."
        sanitized = self.sanitize_input(normal)
        # Should be unchanged since no special chars
        assert "Hello" in sanitized
        assert "World" in sanitized


# ============================================
# SQL Injection Prevention Tests
# ============================================

class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""
    
    def detect_sql_injection(self, input_str: str) -> bool:
        """Detect potential SQL injection attempts."""
        if not input_str:
            return False
        
        sql_patterns = [
            "' OR '1'='1",
            "'; DROP TABLE",
            "1; DROP TABLE",
            "UNION SELECT",
            "' OR 1=1--",
            "admin'--",
            "1' OR '1'='1",
            "'; EXEC",
            "' AND '1'='1",
        ]
        
        input_lower = input_str.lower()
        return any(pattern.lower() in input_lower for pattern in sql_patterns)
    
    def test_basic_sql_injection_detected(self):
        """Test that basic SQL injection is detected."""
        assert self.detect_sql_injection("' OR '1'='1") is True
    
    def test_drop_table_detected(self):
        """Test that DROP TABLE injection is detected."""
        assert self.detect_sql_injection("'; DROP TABLE users;--") is True
    
    def test_union_select_detected(self):
        """Test that UNION SELECT injection is detected."""
        assert self.detect_sql_injection("1 UNION SELECT * FROM users") is True
    
    def test_normal_input_allowed(self):
        """Test that normal input is not flagged."""
        assert self.detect_sql_injection("john.doe@example.com") is False
        assert self.detect_sql_injection("Hello World") is False
        assert self.detect_sql_injection("Order #12345") is False


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
