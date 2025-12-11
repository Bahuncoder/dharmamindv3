<<<<<<< HEAD
"""
🕉️ DharmaMind Authentication & Security Service

Advanced security management for the DharmaMind platform:

Core Features:
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC) 
- API key management and rate limiting
- Session management and security
- Dharmic compliance in security practices
- Advanced threat detection and prevention
- Audit logging and monitoring

Security Principles:
- Ahimsa: No harm through security breaches
- Satya: Truthful identity verification
- Asteya: Protection against unauthorized access
- Dharmic: Righteous security practices

May this service protect all beings with wisdom and compassion 🛡️
"""

import logging
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re

# JWT and cryptography
try:
    import jwt
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logging.warning("Cryptography libraries not installed - using mock implementations")

from pydantic import BaseModel, Field, validator
from ..config import settings
from ..models import UserProfile

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles in the system"""
    ANONYMOUS = "anonymous"           # Anonymous users
    USER = "user"                    # Regular authenticated users
    PREMIUM = "premium"              # Premium subscribers
    MODERATOR = "moderator"          # Content moderators
    DHARMA_GUIDE = "dharma_guide"    # Spiritual guides
    ADMIN = "admin"                  # System administrators
    SUPER_ADMIN = "super_admin"      # Super administrators


class PermissionScope(str, Enum):
    """Permission scopes"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    MODERATION = "moderation"
    ANALYTICS = "analytics"
    SYSTEM = "system"


class SecurityEvent(str, Enum):
    """Security event types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_REFRESH = "token_refresh"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class AuthToken:
    """Authentication token data"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_expires_in: int = 86400


@dataclass
class SecurityAudit:
    """Security audit log entry"""
    event_id: str
    user_id: Optional[str]
    event_type: SecurityEvent
    timestamp: datetime
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    risk_score: float
    action_taken: Optional[str] = None


class AuthenticationService:
    """🛡️ Advanced Authentication and Security Service"""
    
    def __init__(self):
        """Initialize authentication service"""
        self.name = "AuthenticationService"
        
        # Cryptography setup
        if HAS_CRYPTO:
            self.pwd_context = CryptContext(
                schemes=["bcrypt"], 
                deprecated="auto",
                bcrypt__rounds=12
            )
        else:
            self.pwd_context = None
        
        # Token settings
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = timedelta(hours=settings.JWT_EXPIRATION_HOURS)
        self.refresh_token_expire = timedelta(days=7)
        
        # Rate limiting
        self.rate_limits = {}
        self.failed_attempts = {}
        
        # Security audit
        self.audit_log: List[SecurityAudit] = []
        
        # Permission matrix
        self.role_permissions = self._init_role_permissions()
        
        # API keys storage (in production, this would be in database)
        self.api_keys = {}
        
        logger.info("🛡️ Authentication Service initialized")
    
    def _init_role_permissions(self) -> Dict[Role, List[PermissionScope]]:
        """Initialize role-based permissions"""
        return {
            Role.ANONYMOUS: [PermissionScope.READ],
            Role.USER: [PermissionScope.READ, PermissionScope.WRITE],
            Role.PREMIUM: [PermissionScope.READ, PermissionScope.WRITE],
            Role.MODERATOR: [
                PermissionScope.READ, PermissionScope.WRITE, 
                PermissionScope.MODERATION
            ],
            Role.DHARMA_GUIDE: [
                PermissionScope.READ, PermissionScope.WRITE,
                PermissionScope.MODERATION, PermissionScope.ANALYTICS
            ],
            Role.ADMIN: [
                PermissionScope.READ, PermissionScope.WRITE,
                PermissionScope.DELETE, PermissionScope.ADMIN,
                PermissionScope.MODERATION, PermissionScope.ANALYTICS
            ],
            Role.SUPER_ADMIN: [scope for scope in PermissionScope]
        }
    
    async def initialize(self):
        """Initialize authentication service"""
        try:
            logger.info("Initializing Authentication Service...")
            
            # Validate JWT configuration
            if not self.jwt_secret:
                logger.warning("JWT secret key not configured")
            
            # Initialize rate limiting
            await self._init_rate_limiting()
            
            # Load existing API keys (in production, from database)
            await self._load_api_keys()
            
            logger.info("✅ Authentication Service ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Authentication Service: {e}")
            raise
    
    async def _init_rate_limiting(self):
        """Initialize rate limiting system"""
        # Default rate limits per role
        self.rate_limits = {
            Role.ANONYMOUS: {"requests_per_minute": 10, "requests_per_hour": 100},
            Role.USER: {"requests_per_minute": 60, "requests_per_hour": 1000},
            Role.PREMIUM: {"requests_per_minute": 120, "requests_per_hour": 5000},
            Role.MODERATOR: {"requests_per_minute": 200, "requests_per_hour": 10000},
            Role.DHARMA_GUIDE: {"requests_per_minute": 300, "requests_per_hour": 15000},
            Role.ADMIN: {"requests_per_minute": 500, "requests_per_hour": 50000},
            Role.SUPER_ADMIN: {"requests_per_minute": 1000, "requests_per_hour": 100000}
        }
    
    async def _load_api_keys(self):
        """Load API keys from storage"""
        # In production, this would load from database
        # For now, create some default keys
        self.api_keys = {
            "dharma_admin_key": {
                "role": Role.ADMIN,
                "created_at": datetime.now(),
                "last_used": None,
                "active": True,
                "permissions": self.role_permissions[Role.ADMIN]
            }
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        if self.pwd_context:
            return self.pwd_context.hash(password)
        else:
            # Mock implementation for development
            return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        if self.pwd_context:
            return self.pwd_context.verify(plain_password, hashed_password)
        else:
            # Mock implementation
            return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
    
    def create_access_token(self, user_id: str, role: Role, 
                          additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create JWT access token"""
        try:
            now = datetime.utcnow()
            expire = now + self.access_token_expire
            
            payload = {
                "sub": user_id,
                "role": role.value,
                "iat": now,
                "exp": expire,
                "type": "access"
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            if HAS_CRYPTO:
                token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
                return token if isinstance(token, str) else token.decode()
            else:
                # Mock token for development
                return f"mock_access_token_{user_id}_{role.value}"
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        try:
            now = datetime.utcnow()
            expire = now + self.refresh_token_expire
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": expire,
                "type": "refresh"
            }
            
            if HAS_CRYPTO:
                token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
                return token if isinstance(token, str) else token.decode()
            else:
                # Mock token
                return f"mock_refresh_token_{user_id}"
            
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            if HAS_CRYPTO:
                payload = jwt.decode(
                    token, 
                    self.jwt_secret, 
                    algorithms=[self.jwt_algorithm]
                )
                return payload
            else:
                # Mock verification for development
                if token.startswith("mock_"):
                    parts = token.split("_")
                    if len(parts) >= 4:
                        return {
                            "sub": parts[3],
                            "role": parts[4] if len(parts) > 4 else "user",
                            "type": parts[1]
                        }
                return None
                
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    async def authenticate_user(self, email: str, password: str, 
                              ip_address: str, user_agent: str) -> Optional[AuthToken]:
        """Authenticate user with email and password"""
        try:
            # Check rate limiting
            if await self._is_rate_limited(ip_address, "login"):
                await self._log_security_event(
                    None, SecurityEvent.RATE_LIMIT_EXCEEDED,
                    ip_address, user_agent,
                    {"action": "login", "email": email}
                )
                return None
            
            # In production, this would query the database
            # Mock user for development
            mock_users = {
                "admin@dharmamind.com": {
                    "user_id": "admin_001",
                    "password_hash": self.hash_password("dharma_admin"),
                    "role": Role.ADMIN,
                    "active": True
                },
                "user@dharmamind.com": {
                    "user_id": "user_001", 
                    "password_hash": self.hash_password("dharma_user"),
                    "role": Role.USER,
                    "active": True
                }
            }
            
            user_data = mock_users.get(email)
            if not user_data:
                await self._log_security_event(
                    None, SecurityEvent.LOGIN_FAILURE,
                    ip_address, user_agent,
                    {"reason": "user_not_found", "email": email}
                )
                return None
            
            if not user_data["active"]:
                await self._log_security_event(
                    user_data["user_id"], SecurityEvent.LOGIN_FAILURE,
                    ip_address, user_agent,
                    {"reason": "account_inactive", "email": email}
                )
                return None
            
            if not self.verify_password(password, user_data["password_hash"]):
                await self._log_security_event(
                    user_data["user_id"], SecurityEvent.LOGIN_FAILURE,
                    ip_address, user_agent,
                    {"reason": "invalid_password", "email": email}
                )
                return None
            
            # Create tokens
            access_token = self.create_access_token(
                user_data["user_id"], 
                user_data["role"]
            )
            refresh_token = self.create_refresh_token(user_data["user_id"])
            
            # Log successful login
            await self._log_security_event(
                user_data["user_id"], SecurityEvent.LOGIN_SUCCESS,
                ip_address, user_agent,
                {"email": email, "role": user_data["role"].value}
            )
            
            return AuthToken(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=int(self.access_token_expire.total_seconds()),
                refresh_expires_in=int(self.refresh_token_expire.total_seconds())
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data"""
        try:
            key_data = self.api_keys.get(api_key)
            if not key_data or not key_data["active"]:
                return None
            
            # Update last used timestamp
            key_data["last_used"] = datetime.now()
            
            return {
                "api_key": api_key,
                "role": key_data["role"],
                "permissions": key_data["permissions"]
            }
            
        except Exception as e:
            logger.error(f"API key verification error: {e}")
            return None
    
    def check_permission(self, user_role: Role, required_permission: PermissionScope) -> bool:
        """Check if user role has required permission"""
        try:
            role_permissions = self.role_permissions.get(user_role, [])
            return required_permission in role_permissions
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
    
    async def _is_rate_limited(self, identifier: str, action: str) -> bool:
        """Check if identifier is rate limited for action"""
        try:
            now = datetime.now()
            minute_key = f"{identifier}_{action}_{now.strftime('%Y%m%d%H%M')}"
            hour_key = f"{identifier}_{action}_{now.strftime('%Y%m%d%H')}"
            
            # Simple in-memory rate limiting (use Redis in production)
            if not hasattr(self, '_rate_counter'):
                self._rate_counter = {}
            
            minute_count = self._rate_counter.get(minute_key, 0)
            hour_count = self._rate_counter.get(hour_key, 0)
            
            # Default limits for anonymous users
            minute_limit = 10
            hour_limit = 100
            
            return minute_count >= minute_limit or hour_count >= hour_limit
            
        except Exception as e:
            logger.error(f"Rate limiting check error: {e}")
            return False
    
    async def _log_security_event(self, user_id: Optional[str], event_type: SecurityEvent,
                                ip_address: str, user_agent: str, 
                                details: Dict[str, Any], risk_score: float = 0.0):
        """Log security event for audit"""
        try:
            event = SecurityAudit(
                event_id=secrets.token_hex(16),
                user_id=user_id,
                event_type=event_type,
                timestamp=datetime.now(),
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                risk_score=risk_score
            )
            
            self.audit_log.append(event)
            
            # Log to system logger
            logger.info(f"Security Event: {event_type.value} - User: {user_id} - IP: {ip_address}")
            
            # In production, would also store in database and potentially alert
            if risk_score > 0.7:
                logger.warning(f"High-risk security event: {event_type.value} - Score: {risk_score}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def generate_api_key(self, user_id: str, role: Role, 
                             description: str = "") -> str:
        """Generate new API key for user"""
        try:
            api_key = f"dk_{secrets.token_urlsafe(32)}"
            
            self.api_keys[api_key] = {
                "user_id": user_id,
                "role": role,
                "created_at": datetime.now(),
                "last_used": None,
                "active": True,
                "description": description,
                "permissions": self.role_permissions[role]
            }
            
            logger.info(f"Generated API key for user {user_id} with role {role.value}")
            return api_key
            
        except Exception as e:
            logger.error(f"API key generation error: {e}")
            raise
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        try:
            if api_key in self.api_keys:
                self.api_keys[api_key]["active"] = False
                logger.info(f"Revoked API key: {api_key[:10]}...")
                return True
            return False
            
        except Exception as e:
            logger.error(f"API key revocation error: {e}")
            return False
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # In production, would verify user still exists and is active
            # For now, assume user is valid
            
            # Create new tokens
            access_token = self.create_access_token(user_id, Role.USER)  # Default role
            new_refresh_token = self.create_refresh_token(user_id)
            
            return AuthToken(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=int(self.access_token_expire.total_seconds()),
                refresh_expires_in=int(self.refresh_token_expire.total_seconds())
            )
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and analytics"""
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            
            # Filter recent events
            recent_events = [
                event for event in self.audit_log 
                if event.timestamp >= last_24h
            ]
            
            # Calculate metrics
            metrics = {
                "total_events_24h": len(recent_events),
                "login_attempts": len([e for e in recent_events if e.event_type == SecurityEvent.LOGIN_SUCCESS]),
                "failed_logins": len([e for e in recent_events if e.event_type == SecurityEvent.LOGIN_FAILURE]),
                "rate_limit_hits": len([e for e in recent_events if e.event_type == SecurityEvent.RATE_LIMIT_EXCEEDED]),
                "security_violations": len([e for e in recent_events if e.event_type == SecurityEvent.SECURITY_VIOLATION]),
                "active_api_keys": len([k for k in self.api_keys.values() if k["active"]]),
                "high_risk_events": len([e for e in recent_events if e.risk_score > 0.7]),
                "unique_users_24h": len(set(e.user_id for e in recent_events if e.user_id)),
                "avg_risk_score": sum(e.risk_score for e in recent_events) / len(recent_events) if recent_events else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup security service resources"""
        try:
            # Clean old audit logs (keep last 30 days)
            cutoff = datetime.now() - timedelta(days=30)
            self.audit_log = [
                event for event in self.audit_log 
                if event.timestamp >= cutoff
            ]
            
            # Clear rate limiting cache
            if hasattr(self, '_rate_counter'):
                self._rate_counter.clear()
            
            logger.info("Security service cleanup completed")
            
        except Exception as e:
            logger.error(f"Security service cleanup error: {e}")


# Factory function
def create_auth_service() -> AuthenticationService:
    """Create authentication service instance"""
    return AuthenticationService()



async def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """
    Get current user from JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        User information if valid, None if invalid
    """
    try:
        auth_service = get_auth_service()
        return await auth_service.verify_token(token)
    except Exception as e:
        logger.error(f"Failed to get current user: {e}")
        return None

# Global instance
_auth_service = None

def get_auth_service() -> AuthenticationService:
    """Get global authentication service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = create_auth_service()
    return _auth_service
=======
"""
🕉️ DharmaMind Authentication & Security Service

Advanced security management for the DharmaMind platform:

Core Features:
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC) 
- API key management and rate limiting
- Session management and security
- Dharmic compliance in security practices
- Advanced threat detection and prevention
- Audit logging and monitoring

Security Principles:
- Ahimsa: No harm through security breaches
- Satya: Truthful identity verification
- Asteya: Protection against unauthorized access
- Dharmic: Righteous security practices

May this service protect all beings with wisdom and compassion 🛡️
"""

import logging
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re

# JWT and cryptography
try:
    import jwt
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logging.warning("Cryptography libraries not installed - using mock implementations")

from pydantic import BaseModel, Field, validator
from ..config import settings
from ..models import UserProfile, Priority

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles in the system"""
    ANONYMOUS = "anonymous"           # Anonymous users
    USER = "user"                    # Regular authenticated users
    PREMIUM = "premium"              # Premium subscribers
    MODERATOR = "moderator"          # Content moderators
    DHARMA_GUIDE = "dharma_guide"    # Spiritual guides
    ADMIN = "admin"                  # System administrators
    SUPER_ADMIN = "super_admin"      # Super administrators


class PermissionScope(str, Enum):
    """Permission scopes"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    MODERATION = "moderation"
    ANALYTICS = "analytics"
    SYSTEM = "system"


class SecurityEvent(str, Enum):
    """Security event types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_REFRESH = "token_refresh"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class AuthToken:
    """Authentication token data"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_expires_in: int = 86400


@dataclass
class SecurityAudit:
    """Security audit log entry"""
    event_id: str
    user_id: Optional[str]
    event_type: SecurityEvent
    timestamp: datetime
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    risk_score: float
    action_taken: Optional[str] = None


class AuthenticationService:
    """🛡️ Advanced Authentication and Security Service"""
    
    def __init__(self):
        """Initialize authentication service"""
        self.name = "AuthenticationService"
        
        # Cryptography setup
        if HAS_CRYPTO:
            self.pwd_context = CryptContext(
                schemes=["bcrypt"], 
                deprecated="auto",
                bcrypt__rounds=12
            )
        else:
            self.pwd_context = None
        
        # Token settings
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = timedelta(hours=settings.JWT_EXPIRATION_HOURS)
        self.refresh_token_expire = timedelta(days=7)
        
        # Rate limiting
        self.rate_limits = {}
        self.failed_attempts = {}
        
        # Security audit
        self.audit_log: List[SecurityAudit] = []
        
        # Permission matrix
        self.role_permissions = self._init_role_permissions()
        
        # API keys storage (in production, this would be in database)
        self.api_keys = {}
        
        logger.info("🛡️ Authentication Service initialized")
    
    def _init_role_permissions(self) -> Dict[Role, List[PermissionScope]]:
        """Initialize role-based permissions"""
        return {
            Role.ANONYMOUS: [PermissionScope.READ],
            Role.USER: [PermissionScope.READ, PermissionScope.WRITE],
            Role.PREMIUM: [PermissionScope.READ, PermissionScope.WRITE],
            Role.MODERATOR: [
                PermissionScope.READ, PermissionScope.WRITE, 
                PermissionScope.MODERATION
            ],
            Role.DHARMA_GUIDE: [
                PermissionScope.READ, PermissionScope.WRITE,
                PermissionScope.MODERATION, PermissionScope.ANALYTICS
            ],
            Role.ADMIN: [
                PermissionScope.READ, PermissionScope.WRITE,
                PermissionScope.DELETE, PermissionScope.ADMIN,
                PermissionScope.MODERATION, PermissionScope.ANALYTICS
            ],
            Role.SUPER_ADMIN: [scope for scope in PermissionScope]
        }
    
    async def initialize(self):
        """Initialize authentication service"""
        try:
            logger.info("Initializing Authentication Service...")
            
            # Validate JWT configuration
            if not self.jwt_secret:
                logger.warning("JWT secret key not configured")
            
            # Initialize rate limiting
            await self._init_rate_limiting()
            
            # Load existing API keys (in production, from database)
            await self._load_api_keys()
            
            logger.info("✅ Authentication Service ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Authentication Service: {e}")
            raise
    
    async def _init_rate_limiting(self):
        """Initialize rate limiting system"""
        # Default rate limits per role
        self.rate_limits = {
            Role.ANONYMOUS: {"requests_per_minute": 10, "requests_per_hour": 100},
            Role.USER: {"requests_per_minute": 60, "requests_per_hour": 1000},
            Role.PREMIUM: {"requests_per_minute": 120, "requests_per_hour": 5000},
            Role.MODERATOR: {"requests_per_minute": 200, "requests_per_hour": 10000},
            Role.DHARMA_GUIDE: {"requests_per_minute": 300, "requests_per_hour": 15000},
            Role.ADMIN: {"requests_per_minute": 500, "requests_per_hour": 50000},
            Role.SUPER_ADMIN: {"requests_per_minute": 1000, "requests_per_hour": 100000}
        }
    
    async def _load_api_keys(self):
        """Load API keys from storage"""
        # In production, this would load from database
        # For now, create some default keys
        self.api_keys = {
            "dharma_admin_key": {
                "role": Role.ADMIN,
                "created_at": datetime.now(),
                "last_used": None,
                "active": True,
                "permissions": self.role_permissions[Role.ADMIN]
            }
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        if self.pwd_context:
            return self.pwd_context.hash(password)
        else:
            # Mock implementation for development
            return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        if self.pwd_context:
            return self.pwd_context.verify(plain_password, hashed_password)
        else:
            # Mock implementation
            return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
    
    def create_access_token(self, user_id: str, role: Role, 
                          additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create JWT access token"""
        try:
            now = datetime.utcnow()
            expire = now + self.access_token_expire
            
            payload = {
                "sub": user_id,
                "role": role.value,
                "iat": now,
                "exp": expire,
                "type": "access"
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            if HAS_CRYPTO:
                token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
                return token if isinstance(token, str) else token.decode()
            else:
                # Mock token for development
                return f"mock_access_token_{user_id}_{role.value}"
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        try:
            now = datetime.utcnow()
            expire = now + self.refresh_token_expire
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": expire,
                "type": "refresh"
            }
            
            if HAS_CRYPTO:
                token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
                return token if isinstance(token, str) else token.decode()
            else:
                # Mock token
                return f"mock_refresh_token_{user_id}"
            
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            if HAS_CRYPTO:
                payload = jwt.decode(
                    token, 
                    self.jwt_secret, 
                    algorithms=[self.jwt_algorithm]
                )
                return payload
            else:
                # Mock verification for development
                if token.startswith("mock_"):
                    parts = token.split("_")
                    if len(parts) >= 4:
                        return {
                            "sub": parts[3],
                            "role": parts[4] if len(parts) > 4 else "user",
                            "type": parts[1]
                        }
                return None
                
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    async def authenticate_user(self, email: str, password: str, 
                              ip_address: str, user_agent: str) -> Optional[AuthToken]:
        """Authenticate user with email and password"""
        try:
            # Check rate limiting
            if await self._is_rate_limited(ip_address, "login"):
                await self._log_security_event(
                    None, SecurityEvent.RATE_LIMIT_EXCEEDED,
                    ip_address, user_agent,
                    {"action": "login", "email": email}
                )
                return None
            
            # In production, this would query the database
            # Mock user for development
            mock_users = {
                "admin@dharmamind.com": {
                    "user_id": "admin_001",
                    "password_hash": self.hash_password("dharma_admin"),
                    "role": Role.ADMIN,
                    "active": True
                },
                "user@dharmamind.com": {
                    "user_id": "user_001", 
                    "password_hash": self.hash_password("dharma_user"),
                    "role": Role.USER,
                    "active": True
                }
            }
            
            user_data = mock_users.get(email)
            if not user_data:
                await self._log_security_event(
                    None, SecurityEvent.LOGIN_FAILURE,
                    ip_address, user_agent,
                    {"reason": "user_not_found", "email": email}
                )
                return None
            
            if not user_data["active"]:
                await self._log_security_event(
                    user_data["user_id"], SecurityEvent.LOGIN_FAILURE,
                    ip_address, user_agent,
                    {"reason": "account_inactive", "email": email}
                )
                return None
            
            if not self.verify_password(password, user_data["password_hash"]):
                await self._log_security_event(
                    user_data["user_id"], SecurityEvent.LOGIN_FAILURE,
                    ip_address, user_agent,
                    {"reason": "invalid_password", "email": email}
                )
                return None
            
            # Create tokens
            access_token = self.create_access_token(
                user_data["user_id"], 
                user_data["role"]
            )
            refresh_token = self.create_refresh_token(user_data["user_id"])
            
            # Log successful login
            await self._log_security_event(
                user_data["user_id"], SecurityEvent.LOGIN_SUCCESS,
                ip_address, user_agent,
                {"email": email, "role": user_data["role"].value}
            )
            
            return AuthToken(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=int(self.access_token_expire.total_seconds()),
                refresh_expires_in=int(self.refresh_token_expire.total_seconds())
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data"""
        try:
            key_data = self.api_keys.get(api_key)
            if not key_data or not key_data["active"]:
                return None
            
            # Update last used timestamp
            key_data["last_used"] = datetime.now()
            
            return {
                "api_key": api_key,
                "role": key_data["role"],
                "permissions": key_data["permissions"]
            }
            
        except Exception as e:
            logger.error(f"API key verification error: {e}")
            return None
    
    def check_permission(self, user_role: Role, required_permission: PermissionScope) -> bool:
        """Check if user role has required permission"""
        try:
            role_permissions = self.role_permissions.get(user_role, [])
            return required_permission in role_permissions
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
    
    async def _is_rate_limited(self, identifier: str, action: str) -> bool:
        """Check if identifier is rate limited for action"""
        try:
            now = datetime.now()
            minute_key = f"{identifier}_{action}_{now.strftime('%Y%m%d%H%M')}"
            hour_key = f"{identifier}_{action}_{now.strftime('%Y%m%d%H')}"
            
            # Simple in-memory rate limiting (use Redis in production)
            if not hasattr(self, '_rate_counter'):
                self._rate_counter = {}
            
            minute_count = self._rate_counter.get(minute_key, 0)
            hour_count = self._rate_counter.get(hour_key, 0)
            
            # Default limits for anonymous users
            minute_limit = 10
            hour_limit = 100
            
            return minute_count >= minute_limit or hour_count >= hour_limit
            
        except Exception as e:
            logger.error(f"Rate limiting check error: {e}")
            return False
    
    async def _log_security_event(self, user_id: Optional[str], event_type: SecurityEvent,
                                ip_address: str, user_agent: str, 
                                details: Dict[str, Any], risk_score: float = 0.0):
        """Log security event for audit"""
        try:
            event = SecurityAudit(
                event_id=secrets.token_hex(16),
                user_id=user_id,
                event_type=event_type,
                timestamp=datetime.now(),
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                risk_score=risk_score
            )
            
            self.audit_log.append(event)
            
            # Log to system logger
            logger.info(f"Security Event: {event_type.value} - User: {user_id} - IP: {ip_address}")
            
            # In production, would also store in database and potentially alert
            if risk_score > 0.7:
                logger.warning(f"High-risk security event: {event_type.value} - Score: {risk_score}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def generate_api_key(self, user_id: str, role: Role, 
                             description: str = "") -> str:
        """Generate new API key for user"""
        try:
            api_key = f"dk_{secrets.token_urlsafe(32)}"
            
            self.api_keys[api_key] = {
                "user_id": user_id,
                "role": role,
                "created_at": datetime.now(),
                "last_used": None,
                "active": True,
                "description": description,
                "permissions": self.role_permissions[role]
            }
            
            logger.info(f"Generated API key for user {user_id} with role {role.value}")
            return api_key
            
        except Exception as e:
            logger.error(f"API key generation error: {e}")
            raise
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        try:
            if api_key in self.api_keys:
                self.api_keys[api_key]["active"] = False
                logger.info(f"Revoked API key: {api_key[:10]}...")
                return True
            return False
            
        except Exception as e:
            logger.error(f"API key revocation error: {e}")
            return False
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # In production, would verify user still exists and is active
            # For now, assume user is valid
            
            # Create new tokens
            access_token = self.create_access_token(user_id, Role.USER)  # Default role
            new_refresh_token = self.create_refresh_token(user_id)
            
            return AuthToken(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=int(self.access_token_expire.total_seconds()),
                refresh_expires_in=int(self.refresh_token_expire.total_seconds())
            )
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and analytics"""
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            
            # Filter recent events
            recent_events = [
                event for event in self.audit_log 
                if event.timestamp >= last_24h
            ]
            
            # Calculate metrics
            metrics = {
                "total_events_24h": len(recent_events),
                "login_attempts": len([e for e in recent_events if e.event_type == SecurityEvent.LOGIN_SUCCESS]),
                "failed_logins": len([e for e in recent_events if e.event_type == SecurityEvent.LOGIN_FAILURE]),
                "rate_limit_hits": len([e for e in recent_events if e.event_type == SecurityEvent.RATE_LIMIT_EXCEEDED]),
                "security_violations": len([e for e in recent_events if e.event_type == SecurityEvent.SECURITY_VIOLATION]),
                "active_api_keys": len([k for k in self.api_keys.values() if k["active"]]),
                "high_risk_events": len([e for e in recent_events if e.risk_score > 0.7]),
                "unique_users_24h": len(set(e.user_id for e in recent_events if e.user_id)),
                "avg_risk_score": sum(e.risk_score for e in recent_events) / len(recent_events) if recent_events else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup security service resources"""
        try:
            # Clean old audit logs (keep last 30 days)
            cutoff = datetime.now() - timedelta(days=30)
            self.audit_log = [
                event for event in self.audit_log 
                if event.timestamp >= cutoff
            ]
            
            # Clear rate limiting cache
            if hasattr(self, '_rate_counter'):
                self._rate_counter.clear()
            
            logger.info("Security service cleanup completed")
            
        except Exception as e:
            logger.error(f"Security service cleanup error: {e}")


# Factory function
def create_auth_service() -> AuthenticationService:
    """Create authentication service instance"""
    return AuthenticationService()


# Global instance
_auth_service = None

def get_auth_service() -> AuthenticationService:
    """Get global authentication service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = create_auth_service()
    return _auth_service
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
