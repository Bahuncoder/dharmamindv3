"""
🔐 Authentication Service
========================

Core authentication service for DharmaMind with enterprise security features.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import warnings

try:
    from passlib.context import CryptContext
    from jose import JWTError, jwt
    from fastapi import HTTPException, status
    from pydantic import BaseModel
except ImportError as e:
    warnings.warn(f"Auth dependencies not available: {e}")
    
    # Create fallback classes
    class CryptContext:
        def __init__(self, *args, **kwargs):
            pass
        def hash(self, password):
            return hashlib.sha256(password.encode()).hexdigest()
        def verify(self, password, hashed):
            return self.hash(password) == hashed
    
    class BaseModel:
        pass
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
    
    jwt = None
    JWTError = Exception

class Role(str, Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    PREMIUM_USER = "premium_user"
    USER = "user"
    GUEST = "guest"

class PermissionScope(str, Enum):
    """Permission scopes for different features"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    PREMIUM = "premium"

class UserProfile(BaseModel):
    """User profile model"""
    user_id: str
    email: str
    username: str
    role: Role = Role.USER
    is_active: bool = True
    created_at: datetime
    spiritual_preferences: Dict[str, Any] = {}
    
class AuthToken(BaseModel):
    """Authentication token model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class AuthService:
    """🔐 Core Authentication Service"""
    
    def __init__(self, secret_key: str = "dev-secret-key"):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # In-memory user store for development
        self.users_db: Dict[str, Dict] = {}
        self.active_sessions: Dict[str, Dict] = {}
        
        # Initialize with a default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for development"""
        admin_user = {
            "user_id": "admin_001",
            "email": "admin@dharmamind.com",
            "username": "admin",
            "password_hash": self.hash_password("dharma123"),
            "role": Role.ADMIN,
            "is_active": True,
            "created_at": datetime.now(),
            "spiritual_preferences": {
                "preferred_rishi": "vasishtha",
                "meditation_style": "vipassana",
                "spiritual_level": "advanced"
            }
        }
        self.users_db["admin@dharmamind.com"] = admin_user
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({"exp": expire})
            
            if jwt:
                encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
                return encoded_jwt
            else:
                # Fallback token generation
                return secrets.token_urlsafe(32)
                
        except Exception as e:
            warnings.warn(f"Token creation error: {e}")
            return secrets.token_urlsafe(32)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            if jwt:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                return payload
            else:
                # Fallback verification (accept any token for development)
                return {"sub": "admin@dharmamind.com", "role": "admin"}
                
        except Exception as e:
            warnings.warn(f"Token verification error: {e}")
            return None
    
    async def authenticate_user(self, email: str, password: str) -> Optional[UserProfile]:
        """Authenticate user with email and password"""
        user_data = self.users_db.get(email)
        if not user_data:
            return None
        
        if not self.verify_password(password, user_data["password_hash"]):
            return None
        
        return UserProfile(**user_data)
    
    async def create_user(self, email: str, username: str, password: str, role: Role = Role.USER) -> UserProfile:
        """Create a new user"""
        if email in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        user_data = {
            "user_id": f"user_{len(self.users_db) + 1:04d}",
            "email": email,
            "username": username,
            "password_hash": self.hash_password(password),
            "role": role,
            "is_active": True,
            "created_at": datetime.now(),
            "spiritual_preferences": {}
        }
        
        self.users_db[email] = user_data
        return UserProfile(**user_data)
    
    async def get_user(self, email: str) -> Optional[UserProfile]:
        """Get user by email"""
        user_data = self.users_db.get(email)
        if user_data:
            return UserProfile(**user_data)
        return None
    
    def check_permission(self, user_role: Role, required_scope: PermissionScope) -> bool:
        """Check if user role has required permission"""
        role_permissions = {
            Role.ADMIN: [PermissionScope.READ, PermissionScope.WRITE, PermissionScope.DELETE, PermissionScope.ADMIN, PermissionScope.PREMIUM],
            Role.MODERATOR: [PermissionScope.READ, PermissionScope.WRITE, PermissionScope.PREMIUM],
            Role.PREMIUM_USER: [PermissionScope.READ, PermissionScope.WRITE, PermissionScope.PREMIUM],
            Role.USER: [PermissionScope.READ, PermissionScope.WRITE],
            Role.GUEST: [PermissionScope.READ]
        }
        
        return required_scope in role_permissions.get(user_role, [])
    
    async def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh access token using refresh token"""
        # Simple refresh implementation
        payload = self.verify_token(refresh_token)
        if payload:
            new_token = self.create_access_token(data={"sub": payload.get("sub")})
            return AuthToken(
                access_token=new_token,
                expires_in=self.access_token_expire_minutes * 60
            )
        return None

# Global auth service instance
_auth_service: Optional[AuthService] = None

def get_auth_service(secret_key: str = "dev-secret-key") -> AuthService:
    """Get global auth service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(secret_key)
    return _auth_service

def create_auth_service(secret_key: str = "dev-secret-key") -> AuthService:
    """Create new auth service instance"""
    return AuthService(secret_key)

# Export commonly used classes and functions
__all__ = [
    'AuthService',
    'UserProfile', 
    'AuthToken',
    'Role',
    'PermissionScope',
    'get_auth_service',
    'create_auth_service'
]
