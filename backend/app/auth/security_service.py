"""
🛡️ DharmaMind Security Service

Comprehensive security utilities for password hashing, input validation,
encryption, and security policy enforcement.
"""

import secrets
import re
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import bleach
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from passlib.context import CryptContext

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of security threats"""
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALFORMED_INPUT = "malformed_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: ThreatType
    severity: SecurityLevel
    timestamp: datetime
    ip_address: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any]
    action_taken: str


class SecurePasswordService:
    """🔐 Enterprise-grade password security service"""

    def __init__(self):
        # Initialize bcrypt context with optimal settings
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12,  # Good balance of security and performance
            bcrypt__ident="2b"  # Most secure bcrypt variant
        )

        # Password policy settings
        self.min_length = 8
        self.max_length = 128
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special_chars = True
        self.min_special_chars = 1
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"

        # Common passwords to reject
        self.common_passwords = {
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "dragon", "password1",
            "123456789", "football", "iloveyou", "administrator"
        }

        logger.info("🔐 Secure Password Service initialized")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt"""
        # Validate password strength first
        self.validate_password_strength(password)

        try:
            # Hash with bcrypt (includes automatic salt generation)
            hashed = self.pwd_context.hash(password)

            logger.info("Password hashed successfully", extra={
                "hash_algorithm": "bcrypt",
                "rounds": 12,
                "hash_length": len(hashed)
            })

            return hashed

        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise ValueError("Password hashing failed")

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against bcrypt hash"""
        try:
            # Use passlib's verify method (handles all bcrypt variants)
            is_valid = self.pwd_context.verify(password, hashed_password)

            logger.info("Password verification completed", extra={
                "verification_result": is_valid,
                "hash_algorithm": "bcrypt"
            })

            return is_valid

        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Comprehensive password strength validation"""
        issues = []
        score = 0

        # Length validation
        if len(password) < self.min_length:
            msg = f"Password must be at least {self.min_length} characters"
            issues.append(msg)
        elif len(password) >= self.min_length:
            score += 10

        if len(password) > self.max_length:
            msg = f"Password must not exceed {self.max_length} characters"
            issues.append(msg)

        # Character type requirements
        if self.require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain uppercase letter")
        elif any(c.isupper() for c in password):
            score += 15

        if self.require_lowercase and not any(c.islower() for c in password):
            issues.append("Password must contain lowercase letter")
        elif any(c.islower() for c in password):
            score += 15

        if self.require_digits and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        elif any(c.isdigit() for c in password):
            score += 15

        if self.require_special_chars:
            special_count = sum(1 for c in password
                               if c in self.special_chars)
            if special_count < self.min_special_chars:
                msg = (f"Password must contain at least "
                       f"{self.min_special_chars} special character(s)")
                issues.append(msg)
            else:
                score += 15

        # Common password check
        if password.lower() in self.common_passwords:
            issues.append("Password is too common, choose unique password")
        else:
            score += 10

        # Pattern checks
        if re.search(r'(.)\1{2,}', password):  # Repeated characters
            issues.append("Password should not contain repeated characters")
        else:
            score += 5

        pattern = r'(012|123|234|345|456|567|678|789|890|abc|bcd|cde)'
        if re.search(pattern, password.lower()):
            issues.append("Password should not contain sequential characters")
        else:
            score += 5

        # Additional scoring for length
        if len(password) >= 12:
            score += 10
        if len(password) >= 16:
            score += 10

        # Determine strength level
        if score >= 85:
            strength = "very_strong"
        elif score >= 70:
            strength = "strong"
        elif score >= 50:
            strength = "medium"
        elif score >= 30:
            strength = "weak"
        else:
            strength = "very_weak"

        result = {
            "is_valid": len(issues) == 0,
            "strength": strength,
            "score": score,
            "issues": issues,
            "requirements_met": {
                "length": len(password) >= self.min_length,
                "uppercase": any(c.isupper() for c in password),
                "lowercase": any(c.islower() for c in password),
                "digits": any(c.isdigit() for c in password),
                "special_chars": any(c in self.special_chars
                                     for c in password),
                "not_common": password.lower() not in self.common_passwords
            }
        }

        if issues:
            msg = f"Password validation failed: {'; '.join(issues)}"
            raise ValueError(msg)

        return result

    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if password hash needs to be updated"""
        try:
            return self.pwd_context.needs_update(hashed_password)
        except Exception:
            return True  # If we can't verify, assume it needs rehashing

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)


class InputSanitizationService:
    """🧹 Comprehensive input sanitization and validation service"""

    def __init__(self):
        # XSS protection settings
        self.allowed_tags = []  # No HTML tags allowed by default
        self.allowed_attributes = {}

        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\s|^)(select|insert|update|delete|drop|create|alter|exec|"
            r"execute|union|script)\s",
            r"(\s|^)(or|and)\s+\w+\s*=\s*\w+",
            r"(\s|^)\w+\s*=\s*\w+\s+(or|and)\s+",
            r"'.*?'|\".*?\"|`.*?`",  # Quoted strings
            r"--;|/\*|\*/|xp_|sp_",  # SQL comment patterns
            r"\binfo_schema\b|\bsysobjects\b|\bsys\.|sys\s",
            r"@@version|@@servername|@@identity",
            r"char\(|ascii\(|substring\(|concat\(",
            r"waitfor\s+delay|benchmark\(|sleep\(",
        ]

        # Compile regex patterns for performance
        self.sql_regex_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.sql_injection_patterns
        ]

        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
            r"expression\s*\(",
            r"@import",
            r"data:text/html",
        ]

        self.xss_regex_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.xss_patterns
        ]

        logger.info("🧹 Input Sanitization Service initialized")

    def sanitize_input(self,
                       input_text: str,
                       max_length: int = 2000,
                       allow_html: bool = False,
                       security_level: SecurityLevel = SecurityLevel.HIGH
                       ) -> str:
        """Comprehensive input sanitization"""
        if not input_text:
            return ""

        original_text = input_text

        # Length validation
        if len(input_text) > max_length:
            msg = f"Input exceeds maximum length of {max_length} characters"
            raise ValueError(msg)

        # Detect and prevent SQL injection
        if self._detect_sql_injection(input_text):
            logger.warning("SQL injection attempt detected", extra={
                "input_preview": input_text[:100],
                "security_level": security_level.value
            })
            raise ValueError("Input contains potentially malicious SQL")

        # Detect and prevent XSS
        if self._detect_xss(input_text):
            logger.warning("XSS attempt detected", extra={
                "input_preview": input_text[:100],
                "security_level": security_level.value
            })
            raise ValueError("Input contains potentially malicious script")

        # HTML sanitization
        if not allow_html:
            # Remove all HTML tags and decode entities
            input_text = bleach.clean(input_text, tags=[], attributes={},
                                      strip=True)
        else:
            # Allow only safe HTML tags
            input_text = bleach.clean(
                input_text,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )

        # Unicode normalization to prevent homograph attacks
        import unicodedata
        input_text = unicodedata.normalize('NFKC', input_text)

        # Remove null bytes and control characters
        clean_chars = []
        for char in input_text:
            if ord(char) >= 32 or char in '\n\r\t':
                clean_chars.append(char)
        input_text = ''.join(clean_chars)

        # Trim whitespace
        input_text = input_text.strip()

        logger.debug("Input sanitized successfully", extra={
            "original_length": len(original_text),
            "sanitized_length": len(input_text),
            "security_level": security_level.value
        })

        return input_text

    def _detect_sql_injection(self, text: str) -> bool:
        """Detect potential SQL injection patterns"""
        for pattern in self.sql_regex_patterns:
            if pattern.search(text):
                return True
        return False

    def _detect_xss(self, text: str) -> bool:
        """Detect potential XSS patterns"""
        for pattern in self.xss_regex_patterns:
            if pattern.search(text):
                return True
        return False

    def validate_email(self, email: str) -> str:
        """Validate and sanitize email address"""
        if not email:
            raise ValueError("Email is required")

        email = email.strip().lower()

        # Basic email regex validation
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")

        # Check for suspicious patterns
        if any(char in email for char in ['<', '>', '"', "'"]):
            raise ValueError("Email contains invalid characters")

        return email

    def validate_name(self, name: str, field_name: str = "name") -> str:
        """Validate and sanitize name fields"""
        if not name:
            raise ValueError(f"{field_name} is required")

        name = name.strip()

        # Length validation
        if len(name) < 1 or len(name) > 50:
            msg = f"{field_name} must be between 1 and 50 characters"
            raise ValueError(msg)

        # Character validation (letters, spaces, hyphens, apostrophes)
        if not re.match(r"^[a-zA-Z\s\-'\.]+$", name):
            msg = f"{field_name} contains invalid characters"
            raise ValueError(msg)

        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)

        return name


class DataEncryptionService:
    """🔐 Data encryption service for sensitive data at rest"""

    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.fernet = Fernet(master_key)
        else:
            # Generate key from password
            self.master_key = self._generate_key()
            self.fernet = Fernet(self.master_key)

        logger.info("🔐 Data Encryption Service initialized")

    def _generate_key(self) -> bytes:
        """Generate encryption key from environment or create new one"""
        import os

        # Try to get key from environment
        key_env = os.getenv('DHARMAMIND_ENCRYPTION_KEY')
        if key_env:
            return key_env.encode()

        # Get secret key from environment (REQUIRED for security)
        secret_key = os.getenv('SECRET_KEY')
        if not secret_key:
            import secrets
            # Generate a secure random key for development only
            # In production, SECRET_KEY must be set in environment
            import logging
            logging.warning("⚠️ SECRET_KEY not set - using random key (NOT for production)")
            secret_key = secrets.token_urlsafe(32)
        
        password = secret_key.encode()
        salt = b'dharmamind-salt-2024'  # In production, use random salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            raise ValueError("Encryption failed")

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            raise ValueError("Decryption failed")


class SecurityEventLogger:
    """📊 Security event logging and monitoring service"""

    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.logger = logging.getLogger("security")

    async def log_security_event(self,
                                 event_type: ThreatType,
                                 severity: SecurityLevel,
                                 details: Dict[str, Any],
                                 ip_address: Optional[str] = None,
                                 user_id: Optional[str] = None,
                                 action_taken: str = "logged") -> None:
        """Log security event for monitoring and analysis"""

        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_id=user_id,
            details=details,
            action_taken=action_taken
        )

        self.events.append(event)

        # Log to application logger
        self.logger.warning(
            f"Security Event: {event_type.value}",
            extra={
                "event_type": event_type.value,
                "severity": severity.value,
                "ip_address": ip_address,
                "user_id": user_id,
                "details": details,
                "action_taken": action_taken,
                "timestamp": event.timestamp.isoformat()
            }
        )

    def get_recent_events(self, hours: int = 24) -> List[SecurityEvent]:
        """Get security events from the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [event for event in self.events
                if event.timestamp >= cutoff_time]


# Global service instances
password_service = SecurePasswordService()
sanitization_service = InputSanitizationService()
encryption_service = DataEncryptionService()
security_logger = SecurityEventLogger()


# Export main functions for easy use
def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    return password_service.hash_password(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return password_service.verify_password(password, hashed_password)


def sanitize_input(text: str, **kwargs) -> str:
    """Sanitize user input"""
    return sanitization_service.sanitize_input(text, **kwargs)


def encrypt_data(data: str) -> str:
    """Encrypt sensitive data"""
    return encryption_service.encrypt(data)


def decrypt_data(encrypted_data: str) -> str:
    """Decrypt sensitive data"""
    return encryption_service.decrypt(encrypted_data)
