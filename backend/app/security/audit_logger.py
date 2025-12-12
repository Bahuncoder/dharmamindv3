"""
📝 Security Audit Logger

Comprehensive logging for security-related events including:
- Authentication attempts (success/failure)
- Authorization decisions
- Session management events
- Rate limiting triggers
- Suspicious activity detection
- Configuration changes
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum
from functools import wraps
import hashlib
import traceback


class SecurityEventType(str, Enum):
    """Types of security events to audit."""
    # Authentication
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_PASSWORD_CHANGE = "auth.password.change"
    AUTH_PASSWORD_RESET_REQUEST = "auth.password.reset_request"
    AUTH_PASSWORD_RESET_COMPLETE = "auth.password.reset_complete"
    AUTH_MFA_ENABLED = "auth.mfa.enabled"
    AUTH_MFA_DISABLED = "auth.mfa.disabled"
    AUTH_MFA_CHALLENGE = "auth.mfa.challenge"
    
    # Session
    SESSION_CREATED = "session.created"
    SESSION_EXPIRED = "session.expired"
    SESSION_REVOKED = "session.revoked"
    SESSION_SUSPICIOUS = "session.suspicious"
    
    # Authorization
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PRIVILEGE_ESCALATION = "authz.privilege.escalation"
    
    # Rate Limiting
    RATE_LIMIT_WARNING = "rate.limit.warning"
    RATE_LIMIT_EXCEEDED = "rate.limit.exceeded"
    RATE_LIMIT_IP_BLOCKED = "rate.limit.ip_blocked"
    
    # Data Access
    DATA_ACCESS = "data.access"
    DATA_EXPORT = "data.export"
    DATA_DELETE = "data.delete"
    DATA_MODIFICATION = "data.modification"
    
    # Security Threats
    THREAT_SQL_INJECTION = "threat.sql_injection"
    THREAT_XSS_ATTEMPT = "threat.xss_attempt"
    THREAT_CSRF_VIOLATION = "threat.csrf_violation"
    THREAT_BRUTE_FORCE = "threat.brute_force"
    THREAT_SUSPICIOUS_IP = "threat.suspicious_ip"
    
    # System
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_ADMIN_ACTION = "system.admin.action"
    SYSTEM_ERROR = "system.error"


class SecurityAuditLogger:
    """
    Centralized security audit logging system.
    
    Usage:
        audit = SecurityAuditLogger()
        audit.log_event(
            event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
            user_id="user123",
            ip_address="192.168.1.1",
            details={"method": "password"}
        )
    """
    
    def __init__(
        self,
        log_file: str = None,
        log_level: int = logging.INFO,
        console_output: bool = True,
        json_format: bool = True
    ):
        """Initialize the security audit logger."""
        self.log_file = log_file or os.getenv(
            "SECURITY_AUDIT_LOG",
            "logs/security_audit.log"
        )
        self.json_format = json_format
        
        # Create logs directory if needed
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        
        if json_format:
            file_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            ))
        
        self.logger.addHandler(file_handler)
        
        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(
                "🔐 %(asctime)s | %(message)s",
                datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(console_handler)
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in log data."""
        sensitive_fields = [
            "password", "token", "secret", "key", "authorization",
            "credit_card", "ssn", "api_key", "private_key"
        ]
        
        masked = {}
        for key, value in data.items():
            if any(field in key.lower() for field in sensitive_fields):
                if isinstance(value, str) and len(value) > 4:
                    masked[key] = value[:2] + "*" * (len(value) - 4) + value[-2:]
                else:
                    masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive_data(value)
            else:
                masked[key] = value
        
        return masked
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.utcnow().isoformat()
        random_part = os.urandom(8).hex()
        return hashlib.sha256(f"{timestamp}{random_part}".encode()).hexdigest()[:16]
    
    def log_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ) -> str:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            user_id: User identifier (if applicable)
            ip_address: Client IP address
            user_agent: Client user agent
            resource: Resource being accessed
            action: Action being performed
            status: Event status (success, failure, blocked)
            details: Additional event details
            severity: Log severity (info, warning, error, critical)
        
        Returns:
            Event ID for reference
        """
        event_id = self._generate_event_id()
        
        event = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type.value,
            "severity": severity,
            "status": status,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "resource": resource,
            "action": action,
            "details": self._mask_sensitive_data(details or {}),
        }
        
        # Remove None values
        event = {k: v for k, v in event.items() if v is not None}
        
        # Log based on severity
        log_method = getattr(self.logger, severity, self.logger.info)
        
        if self.json_format:
            log_method(json.dumps(event))
        else:
            log_method(f"{event_type.value} | user={user_id} | ip={ip_address} | {details}")
        
        return event_id
    
    # Convenience methods for common events
    
    def log_login_success(self, user_id: str, ip: str, method: str = "password", **kwargs):
        """Log successful login."""
        return self.log_event(
            SecurityEventType.AUTH_LOGIN_SUCCESS,
            user_id=user_id,
            ip_address=ip,
            details={"method": method, **kwargs}
        )
    
    def log_login_failure(self, user_id: str, ip: str, reason: str, **kwargs):
        """Log failed login attempt."""
        return self.log_event(
            SecurityEventType.AUTH_LOGIN_FAILURE,
            user_id=user_id,
            ip_address=ip,
            status="failure",
            severity="warning",
            details={"reason": reason, **kwargs}
        )
    
    def log_access_denied(self, user_id: str, resource: str, ip: str, reason: str):
        """Log access denied event."""
        return self.log_event(
            SecurityEventType.AUTHZ_ACCESS_DENIED,
            user_id=user_id,
            ip_address=ip,
            resource=resource,
            status="denied",
            severity="warning",
            details={"reason": reason}
        )
    
    def log_rate_limit(self, ip: str, endpoint: str, count: int):
        """Log rate limit exceeded."""
        return self.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            ip_address=ip,
            resource=endpoint,
            status="blocked",
            severity="warning",
            details={"request_count": count}
        )
    
    def log_suspicious_activity(self, ip: str, threat_type: str, details: Dict):
        """Log suspicious activity detection."""
        event_map = {
            "sql_injection": SecurityEventType.THREAT_SQL_INJECTION,
            "xss": SecurityEventType.THREAT_XSS_ATTEMPT,
            "csrf": SecurityEventType.THREAT_CSRF_VIOLATION,
            "brute_force": SecurityEventType.THREAT_BRUTE_FORCE,
        }
        event_type = event_map.get(threat_type, SecurityEventType.THREAT_SUSPICIOUS_IP)
        
        return self.log_event(
            event_type,
            ip_address=ip,
            status="detected",
            severity="error",
            details=details
        )
    
    def log_session_event(self, event: str, session_id: str, user_id: str, ip: str):
        """Log session lifecycle events."""
        event_map = {
            "created": SecurityEventType.SESSION_CREATED,
            "expired": SecurityEventType.SESSION_EXPIRED,
            "revoked": SecurityEventType.SESSION_REVOKED,
            "suspicious": SecurityEventType.SESSION_SUSPICIOUS,
        }
        return self.log_event(
            event_map.get(event, SecurityEventType.SESSION_CREATED),
            user_id=user_id,
            ip_address=ip,
            details={"session_id": session_id[:8] + "..."}  # Partial session ID
        )


# Decorator for automatic audit logging
def audit_log(event_type: SecurityEventType, get_user_id=None, get_resource=None):
    """
    Decorator to automatically log security events.
    
    Usage:
        @audit_log(SecurityEventType.DATA_ACCESS, get_user_id=lambda r: r.user.id)
        async def get_user_data(request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            audit = SecurityAuditLogger()
            
            # Extract user_id and resource if possible
            user_id = None
            resource = func.__name__
            ip_address = None
            
            # Try to get request object
            request = kwargs.get("request") or (args[0] if args else None)
            if hasattr(request, "client"):
                ip_address = getattr(request.client, "host", None)
            if get_user_id and request:
                try:
                    user_id = get_user_id(request)
                except:
                    pass
            if get_resource and request:
                try:
                    resource = get_resource(request)
                except:
                    pass
            
            try:
                result = await func(*args, **kwargs)
                audit.log_event(
                    event_type,
                    user_id=user_id,
                    ip_address=ip_address,
                    resource=resource,
                    status="success"
                )
                return result
            except Exception as e:
                audit.log_event(
                    event_type,
                    user_id=user_id,
                    ip_address=ip_address,
                    resource=resource,
                    status="error",
                    severity="error",
                    details={"error": str(e)}
                )
                raise
        
        return wrapper
    return decorator


# Global instance
_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


# Example usage
if __name__ == "__main__":
    audit = SecurityAuditLogger(console_output=True)
    
    # Log various events
    audit.log_login_success("user123", "192.168.1.1", method="oauth_google")
    audit.log_login_failure("user456", "192.168.1.2", reason="invalid_password")
    audit.log_access_denied("user789", "/api/admin/users", "192.168.1.3", "insufficient_permissions")
    audit.log_rate_limit("192.168.1.4", "/api/login", 150)
    audit.log_suspicious_activity("192.168.1.5", "sql_injection", {"payload": "' OR 1=1--"})
    
    print("\n✅ Security audit events logged to:", audit.log_file)
