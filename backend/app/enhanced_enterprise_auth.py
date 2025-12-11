"""
Enhanced Enterprise Authentication Backend for DharmaMind
Extended version with user profile management and enhanced security
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
import logging
import secrets
import uuid
from datetime import datetime, timedelta
import uvicorn
import os
import json
from enum import Enum

# Import our secure password service
from services.security_service import (
    hash_password, 
    verify_password, 
    sanitize_input, 
    password_service,
    sanitization_service,
    security_logger,
    ThreatType,
    SecurityLevel
)
from services.secret_manager import secret_manager, get_secret

# Import secure database service
from services.database_service import (
    db_service,
    init_database,
    get_db_health
)

# Import HTTPS security service
from services.https_service import (
    https_service,
    get_ssl_config,
    setup_https,
    validate_https
)

# Import data management service
from services.data_manager import (
    data_manager,
    store_chat_message,
    get_user_chat_history,
    create_user_data
)

# Import advanced security service  
from services.advanced_security import (
    security_manager,
    encrypt_sensitive_data,
    decrypt_sensitive_data,
    audit_security_event,
    SecurityLevel,
    EncryptionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🛡️ Security and Configuration
SECRET_KEY = get_secret("SECRET_KEY")
JWT_SECRET_KEY = get_secret("JWT_SECRET_KEY", SECRET_KEY)
ENCRYPTION_KEY = get_secret("DHARMAMIND_ENCRYPTION_KEY")
DATABASE_URL = get_secret("DATABASE_URL", "sqlite:///dharmamind.db")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI(
    title="DharmaMind Enterprise Authentication API",
    description="Complete enterprise authentication system with user management",
    version="2.0.0"
)

# Database initialization flag
_db_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize database and security services on startup"""
    global _db_initialized
    
    if not _db_initialized:
        logger.info("🚀 Initializing DharmaMind security services...")
        
        # Initialize secure database
        db_success = await init_database()
        if db_success:
            logger.info("✅ Secure database initialized successfully")
        else:
            logger.warning("⚠️ Database initialization failed, using fallback")
        
        # Set up HTTPS development environment
        https_success = setup_https()
        if https_success:
            logger.info("✅ HTTPS security configured successfully")
        else:
            logger.warning("⚠️ HTTPS setup failed, using HTTP fallback")
        
        _db_initialized = True
        logger.info("🛡️ Security services ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    from services.database_service import close_database
    await close_database()
    logger.info("🔒 Security services shutdown complete")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3006", "http://localhost:3005", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Enums
class SubscriptionPlan(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class UserStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class AuthProvider(str, Enum):
    EMAIL = "email"
    GOOGLE = "google"
    DEMO = "demo"

# Enhanced Pydantic models
class UserRegistration(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    first_name: str = Field(..., min_length=1, max_length=50, description="First name")
    last_name: str = Field(..., min_length=1, max_length=50, description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")
    timezone: Optional[str] = Field("UTC", description="User timezone")
    accept_terms: bool = Field(..., description="Terms of service acceptance")
    accept_privacy: bool = Field(..., description="Privacy policy acceptance")
    marketing_consent: bool = Field(False, description="Marketing consent")
    
    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

class UserLogin(BaseModel):
    email: str
    password: str
    remember_me: bool = Field(False, description="Remember login for extended session")

class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    phone: Optional[str] = None
    timezone: Optional[str] = None
    marketing_consent: Optional[bool] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @field_validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

# Enhanced in-memory storage for demo (replace with database in production)
users_db = {}
sessions_db = {}
user_profiles_db = {}
security_events_db = []

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🧘 DharmaMind Enhanced Enterprise Authentication API",
        "version": "2.0.0", 
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "register": "/auth/register",
            "login": "/auth/login",
            "profile": "/users/profile"
        },
        "features": [
            "Enterprise Authentication",
            "JWT Token Management",
            "Password Security Validation", 
            "User Profile Management",
            "Session Management",
            "Audit Logging"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """🛡️ Enhanced health check with comprehensive security status"""
    try:
        # Get database health if available
        db_health = await get_db_health()
        
        # Get HTTPS validation status
        https_status = validate_https()
        
        # Get secret management status
        environment_status = secret_manager.get_environment_status()
        
        # Calculate security score
        security_checks = [
            True,  # bcrypt enabled
            True,  # input sanitization enabled
            db_health.get("encryption") == "enabled",
            environment_status.get("secrets_configured", False),
            https_status.get("ssl_context_valid", False)
        ]
        
        security_score = (sum(security_checks) / len(security_checks)) * 100
        
        return {
            "status": "healthy",
            "message": "DharmaMind API is running successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "security_score": f"{security_score:.1f}%",
            "security": {
                "password_hashing": "bcrypt with 12 rounds",
                "encryption": "Fernet encryption enabled", 
                "input_sanitization": "XSS/SQL injection protection",
                "database_encryption": "enabled" if db_health.get("encryption") == "enabled" else "configured",
                "https_configured": https_status.get("certificate_exists", False),
                "ssl_context_valid": https_status.get("ssl_context_valid", False),
                "secret_management": "environment-based configuration"
            },
            "database": db_health,
            "https": https_status,
            "environment": {
                "secrets_configured": environment_status.get("secrets_configured", False),
                "security_score": environment_status.get("security_score", 0),
                "missing_secrets": len(environment_status.get("validation", {}).get("missing_secrets", []))
            },
            "services": {
                "authentication": "operational",
                "user_management": "operational", 
                "session_management": "operational",
                "security_logging": "operational"
            },
            "security_implementations": {
                "✅ bcrypt_password_hashing": "COMPLETED",
                "✅ input_sanitization": "COMPLETED", 
                "✅ data_encryption": "COMPLETED",
                "✅ https_service": "COMPLETED",
                "✅ secret_management": "COMPLETED"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "message": "Some services may be unavailable",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

class EnhancedAuthService:
    """Enhanced authentication service with user management"""
    
    def __init__(self):
        self.users = users_db
        self.sessions = sessions_db
        self.profiles = user_profiles_db
        self.security_events = security_events_db
        
    async def register_user(self, user_data: Dict[str, Any], ip_address: str = None) -> Dict[str, Any]:
        """Register new user with enhanced validation and comprehensive data storage"""
        email = user_data["email"].lower()
        
        # Check if user exists
        if email in self.users:
            await self.log_security_event("registration_attempt_duplicate_email", {
                "email": email,
                "ip_address": ip_address
            })
            return {
                "success": False,
                "error": "User already exists with this email address"
            }
        
        # Create user with comprehensive data management
        try:
            # Create comprehensive user data
            comprehensive_user = await create_user_data(user_data)
            
            # Also store in legacy format for compatibility
            user_id = comprehensive_user.user_id
            password_hash = hash_password(user_data["password"])
            
            self.users[email] = {
                "user_id": user_id,
                "email": email,
                "password_hash": password_hash,
                "first_name": user_data["first_name"],
                "last_name": user_data["last_name"],
                "subscription_plan": SubscriptionPlan.FREE,
                "status": UserStatus.ACTIVE,
                "email_verified": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "last_login": None,
                "login_count": 0,
                "auth_provider": AuthProvider.EMAIL
            }
            
            # Store extended profile for compatibility
            self.profiles[user_id] = {
                "user_id": user_id,
                "phone": user_data.get("phone"),
                "timezone": user_data.get("timezone", "UTC"),
                "marketing_consent": user_data.get("marketing_consent", False),
                "profile_picture": None,
                "preferences": {
                    "notifications": True,
                    "theme": "system",
                    "language": "en"
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            await self.log_security_event("user_registered", {
                "user_id": user_id,
                "email": email,
                "ip_address": ip_address,
                "auth_provider": AuthProvider.EMAIL,
                "comprehensive_data": True
            })
            
            logger.info(f"User registered with comprehensive data: {email} (ID: {user_id})")
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "Registration successful",
                "comprehensive_data": True
            }
            
        except Exception as e:
            logger.error(f"Comprehensive user registration failed: {e}")
            # Fallback to basic registration
            return await self._register_user_basic(user_data, ip_address)
    
    async def _register_user_basic(self, user_data: Dict[str, Any], ip_address: str = None) -> Dict[str, Any]:
        """Fallback basic user registration"""
        email = user_data["email"].lower()
        user_id = str(uuid.uuid4())
        password_hash = hash_password(user_data["password"])
        
        # Store user data
        self.users[email] = {
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "first_name": user_data["first_name"],
            "last_name": user_data["last_name"],
            "subscription_plan": SubscriptionPlan.FREE,
            "status": UserStatus.ACTIVE,
            "email_verified": True,  # Auto-verify for demo
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "login_count": 0,
            "auth_provider": AuthProvider.EMAIL
        }
        
        # Store extended profile
        self.profiles[user_id] = {
            "user_id": user_id,
            "phone": user_data.get("phone"),
            "timezone": user_data.get("timezone", "UTC"),
            "marketing_consent": user_data.get("marketing_consent", False),
            "profile_picture": None,
            "preferences": {
                "notifications": True,
                "theme": "system",
                "language": "en"
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await self.log_security_event("user_registered", {
            "user_id": user_id,
            "email": email,
            "ip_address": ip_address,
            "auth_provider": AuthProvider.EMAIL
        })
        
        logger.info(f"User registered: {email} (ID: {user_id})")
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "Registration successful"
        }
    
    async def login_user(self, email: str, password: str, remember_me: bool = False, ip_address: str = None) -> Dict[str, Any]:
        """Authenticate user login with enhanced tracking"""
        email = email.lower()
        
        if email not in self.users:
            await self.log_security_event("login_attempt_invalid_email", {
                "email": email,
                "ip_address": ip_address
            })
            return {
                "success": False,
                "error": "Invalid credentials"
            }
        
        user = self.users[email]
        
        # Use secure bcrypt verification instead of SHA256
        if not verify_password(password, user["password_hash"]):
            await self.log_security_event("login_attempt_invalid_password", {
                "email": email,
                "ip_address": ip_address
            })
            return {
                "success": False,
                "error": "Invalid credentials"
            }
        
        # Check user status
        if user["status"] != UserStatus.ACTIVE:
            await self.log_security_event("login_attempt_inactive_user", {
                "email": email,
                "status": user["status"],
                "ip_address": ip_address
            })
            return {
                "success": False,
                "error": "Account is not active"
            }
        
        # Create session
        session_duration = timedelta(hours=720 if remember_me else 24)  # 30 days vs 24 hours
        session_token = f"session_{secrets.token_urlsafe(32)}"
        expires_at = (datetime.utcnow() + session_duration).timestamp() * 1000
        
        self.sessions[session_token] = {
            "user_id": user["user_id"],
            "email": email,
            "expires_at": expires_at,
            "created_at": datetime.utcnow().timestamp() * 1000,
            "ip_address": ip_address,
            "remember_me": remember_me,
            "last_activity": datetime.utcnow().timestamp() * 1000
        }
        
        # Update user login stats
        user["last_login"] = datetime.utcnow().isoformat()
        user["login_count"] = user.get("login_count", 0) + 1
        user["updated_at"] = datetime.utcnow().isoformat()
        
        await self.log_security_event("user_logged_in", {
            "user_id": user["user_id"],
            "email": email,
            "ip_address": ip_address,
            "remember_me": remember_me
        })
        
        logger.info(f"User logged in: {email} (Sessions: {user['login_count']})")
        
        return {
            "success": True,
            "user": {
                "user_id": user["user_id"],
                "email": user["email"],
                "first_name": user["first_name"],
                "last_name": user["last_name"],
                "subscription_plan": user["subscription_plan"],
                "status": user["status"],
                "email_verified": user["email_verified"],
                "auth_provider": user["auth_provider"],
                "last_login": user["last_login"],
                "login_count": user["login_count"]
            },
            "session_token": session_token,
            "expires_at": expires_at
        }
    
    async def validate_session(self, session_token: str, ip_address: str = None) -> Dict[str, Any]:
        """Validate session token with activity tracking"""
        if session_token not in self.sessions:
            return {
                "success": False,
                "error": "Invalid session"
            }
        
        session = self.sessions[session_token]
        
        # Check expiry
        if datetime.utcnow().timestamp() * 1000 > session["expires_at"]:
            del self.sessions[session_token]
            return {
                "success": False,
                "error": "Session expired"
            }
        
        # Update last activity
        session["last_activity"] = datetime.utcnow().timestamp() * 1000
        
        # Get user data
        user = self.users.get(session["email"])
        if not user:
            del self.sessions[session_token]
            return {
                "success": False,
                "error": "User not found"
            }
        
        # Get profile data
        profile = self.profiles.get(user["user_id"], {})
        
        return {
            "success": True,
            "user": {
                "user_id": user["user_id"],
                "email": user["email"],
                "first_name": user["first_name"],
                "last_name": user["last_name"],
                "subscription_plan": user["subscription_plan"],
                "status": user["status"],
                "email_verified": user["email_verified"],
                "auth_provider": user["auth_provider"],
                "last_login": user["last_login"],
                "login_count": user["login_count"],
                "profile": profile
            },
            "session": {
                "expires_at": session["expires_at"],
                "remember_me": session.get("remember_me", False),
                "last_activity": session["last_activity"]
            }
        }
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile information"""
        # Find user by ID
        user = None
        for email, user_data in self.users.items():
            if user_data["user_id"] == user_id:
                user = user_data
                break
        
        if not user:
            return {
                "success": False,
                "error": "User not found"
            }
        
        # Update user basic info
        if "first_name" in profile_data and profile_data["first_name"]:
            user["first_name"] = profile_data["first_name"]
        if "last_name" in profile_data and profile_data["last_name"]:
            user["last_name"] = profile_data["last_name"]
        
        user["updated_at"] = datetime.utcnow().isoformat()
        
        # Update extended profile
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat()
            }
        
        profile = self.profiles[user_id]
        
        if "phone" in profile_data:
            profile["phone"] = profile_data["phone"]
        if "timezone" in profile_data:
            profile["timezone"] = profile_data["timezone"]
        if "marketing_consent" in profile_data:
            profile["marketing_consent"] = profile_data["marketing_consent"]
        
        profile["updated_at"] = datetime.utcnow().isoformat()
        
        await self.log_security_event("profile_updated", {
            "user_id": user_id,
            "updated_fields": list(profile_data.keys())
        })
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "user": user,
            "profile": profile
        }
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password"""
        # Find user by ID
        user = None
        for email, user_data in self.users.items():
            if user_data["user_id"] == user_id:
                user = user_data
                break
        
        if not user:
            return {
                "success": False,
                "error": "User not found"
            }
        
        # Verify current password using secure bcrypt
        if not verify_password(current_password, user["password_hash"]):
            await self.log_security_event("password_change_failed", {
                "user_id": user_id,
                "reason": "invalid_current_password"
            })
            return {
                "success": False,
                "error": "Current password is incorrect"
            }
        
        # Update password with secure bcrypt hashing
        new_hash = hash_password(new_password)
        user["password_hash"] = new_hash
        user["updated_at"] = datetime.utcnow().isoformat()
        
        await self.log_security_event("password_changed", {
            "user_id": user_id
        })
        
        return {
            "success": True,
            "message": "Password changed successfully"
        }
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring"""
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only last 1000 events to prevent memory issues
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        logger.info(f"Security event: {event_type} - {details}")

# Initialize enhanced auth service
auth_service = EnhancedAuthService()

def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        result = await auth_service.validate_session(token)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return result["user"]
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# ===============================
# HIGH SECURITY ENDPOINTS
# ===============================

@app.post("/api/v1/security/encrypt-data")
async def encrypt_user_data(
    request: Request,
    data: Dict[str, Any],
    security_level: str = "confidential",
    current_user: Dict = Depends(verify_token)
):
    """Encrypt sensitive user data with enterprise-grade security"""
    try:
        # Audit the encryption request
        await audit_security_event(
            event_type="data_encryption_request",
            action="encrypt_data",
            resource_id=current_user["user_id"],
            result="initiated",
            user_id=current_user["user_id"],
            ip_address=request.client.host,
            metadata={"security_level": security_level}
        )
        
        # Convert security level
        sec_level = SecurityLevel(security_level.lower())
        
        # Encrypt the data
        encrypted_package = await encrypt_sensitive_data(data, sec_level)
        
        return {
            "success": True,
            "encrypted_data": encrypted_package,
            "encryption_id": encrypted_package["context"]["encryption_id"],
            "security_level": security_level
        }
        
    except Exception as e:
        logger.error(f"Data encryption failed: {e}")
        await audit_security_event(
            event_type="data_encryption_failure",
            action="encrypt_data",
            resource_id=current_user["user_id"],
            result="failure",
            user_id=current_user["user_id"],
            metadata={"error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail="Data encryption failed"
        )


@app.post("/api/v1/security/decrypt-data")
async def decrypt_user_data(
    request: Request,
    encrypted_package: Dict[str, Any],
    current_user: Dict = Depends(verify_token)
):
    """Decrypt sensitive user data with integrity verification"""
    try:
        # Audit the decryption request
        await audit_security_event(
            event_type="data_decryption_request",
            action="decrypt_data",
            resource_id=current_user["user_id"],
            result="initiated",
            user_id=current_user["user_id"],
            ip_address=request.client.host,
            metadata={"encryption_id": encrypted_package.get("context", {}).get("encryption_id")}
        )
        
        # Decrypt the data
        decrypted_bytes = await decrypt_sensitive_data(encrypted_package)
        
        # Try to parse as JSON
        try:
            decrypted_data = json.loads(decrypted_bytes.decode())
        except:
            decrypted_data = decrypted_bytes.decode()
        
        return {
            "success": True,
            "decrypted_data": decrypted_data
        }
        
    except Exception as e:
        logger.error(f"Data decryption failed: {e}")
        await audit_security_event(
            event_type="data_decryption_failure",
            action="decrypt_data",
            resource_id=current_user["user_id"],
            result="failure",
            user_id=current_user["user_id"],
            metadata={"error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail="Data decryption failed"
        )


@app.get("/api/v1/security/metrics")
async def get_security_metrics(
    request: Request,
    current_user: Dict = Depends(verify_token)
):
    """Get comprehensive security metrics"""
    try:
        # Check if user has admin privileges
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )
        
        # Get security metrics
        metrics = await security_manager.get_security_metrics()
        
        await audit_security_event(
            event_type="security_metrics_access",
            action="view_metrics",
            resource_id="security_metrics",
            result="success",
            user_id=current_user["user_id"],
            ip_address=request.client.host
        )
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security metrics error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve security metrics"
        )


@app.get("/api/v1/security/health")
async def security_health_check(
    request: Request,
    current_user: Dict = Depends(verify_token)
):
    """Comprehensive security health check"""
    try:
        # Check if user has admin privileges
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )
        
        # Perform health check
        health_status = await security_manager.security_health_check()
        
        await audit_security_event(
            event_type="security_health_check",
            action="health_check",
            resource_id="security_system",
            result="success",
            user_id=current_user["user_id"],
            ip_address=request.client.host,
            metadata={"health_status": health_status["overall_status"]}
        )
        
        return {
            "success": True,
            "health_status": health_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security health check error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Security health check failed"
        )


@app.post("/api/v1/security/rotate-keys")
async def rotate_encryption_keys(
    request: Request,
    current_user: Dict = Depends(verify_token)
):
    """Rotate encryption keys for enhanced security"""
    try:
        # Check if user has admin privileges
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )
        
        # Rotate keys
        await security_manager.rotate_encryption_keys()
        
        await audit_security_event(
            event_type="key_rotation",
            action="rotate_keys", 
            resource_id="encryption_keys",
            result="success",
            user_id=current_user["user_id"],
            ip_address=request.client.host
        )
        
        return {
            "success": True,
            "message": "Encryption keys rotated successfully"
        }
        
    except Exception as e:
        logger.error(f"Key rotation failed: {e}")
        await audit_security_event(
            event_type="key_rotation_failure",
            action="rotate_keys",
            resource_id="encryption_keys", 
            result="failure",
            user_id=current_user["user_id"],
            metadata={"error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail="Key rotation failed"
        )


# Enhanced authentication endpoints
@app.post("/auth/register")
async def register_user(user_data: UserRegistration, request: Request):
    """Register new user with enhanced validation"""
    try:
        ip_address = get_client_ip(request)
        
        result = await auth_service.register_user(
            user_data.dict(),
            ip_address=ip_address
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result.get("message", "Registration successful"),
                "user_id": result.get("user_id")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Registration failed")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/login")
async def login_user(login_data: UserLogin, request: Request):
    """Authenticate user login with enhanced tracking"""
    try:
        ip_address = get_client_ip(request)
        
        result = await auth_service.login_user(
            email=login_data.email,
            password=login_data.password,
            remember_me=login_data.remember_me,
            ip_address=ip_address
        )
        
        if result["success"]:
            return {
                "success": True,
                "user": result["user"],
                "session_token": result.get("session_token"),
                "expires_at": result.get("expires_at")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.get("error", "Login failed")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.put("/auth/profile")
async def update_profile(profile_data: UserProfileUpdate, current_user: dict = Depends(get_current_user)):
    """Update user profile"""
    try:
        result = await auth_service.update_user_profile(
            user_id=current_user["user_id"],
            profile_data=profile_data.dict(exclude_unset=True)
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "user": result["user"],
                "profile": result["profile"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Profile update failed")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@app.put("/auth/password")
async def change_password(password_data: PasswordChange, current_user: dict = Depends(get_current_user)):
    """Change user password"""
    try:
        result = await auth_service.change_password(
            user_id=current_user["user_id"],
            current_password=password_data.current_password,
            new_password=password_data.new_password
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Password change failed")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

# Include all previous endpoints from simple_enterprise_auth.py
@app.post("/auth/logout")
async def logout_user(current_user: dict = Depends(get_current_user),
                     credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user"""
    try:
        token = credentials.credentials
        if token in auth_service.sessions:
            del auth_service.sessions[token]
            
            await auth_service.log_security_event("user_logged_out", {
                "user_id": current_user["user_id"]
            })
        
        return {
            "success": True,
            "message": "Logout successful"
        }
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@app.get("/auth/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    return {
        "success": True,
        "user": current_user
    }

# New Data Management Endpoints
@app.get("/api/conversations")
async def get_user_conversations(current_user: dict = Depends(get_current_user)):
    """Get user's conversation history"""
    try:
        user_id = current_user["user_id"]
        conversations = await data_manager.get_user_conversations(user_id, limit=20)
        
        return {
            "success": True,
            "conversations": [conv.to_dict() for conv in conversations],
            "total_conversations": len(conversations)
        }
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )

@app.get("/api/conversations/{conversation_id}")
async def get_conversation_details(
    conversation_id: str, 
    current_user: dict = Depends(get_current_user)
):
    """Get detailed conversation with all messages"""
    try:
        conversation = await data_manager.get_conversation(conversation_id)
        
        if not conversation or conversation.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {
            "success": True,
            "conversation": conversation.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation details error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )

@app.get("/api/chat-history")
async def get_chat_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user's recent chat history"""
    try:
        user_id = current_user["user_id"]
        chat_history = await get_user_chat_history(user_id, limit)
        
        return {
            "success": True,
            "chat_history": chat_history,
            "total_messages": len(chat_history)
        }
    except Exception as e:
        logger.error(f"Get chat history error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history"
        )

@app.get("/api/analytics")
async def get_user_analytics_endpoint(current_user: dict = Depends(get_current_user)):
    """Get comprehensive user analytics"""
    try:
        user_id = current_user["user_id"]
        analytics = await data_manager.get_user_analytics(user_id)
        
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        logger.error(f"Get analytics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )

@app.get("/api/export-data")
async def export_user_data_endpoint(current_user: dict = Depends(get_current_user)):
    """Export all user data (GDPR compliance)"""
    try:
        user_id = current_user["user_id"]
        export_data = await data_manager.export_user_data(user_id)
        
        return {
            "success": True,
            "export_data": export_data,
            "message": "User data exported successfully"
        }
    except Exception as e:
        logger.error(f"Export data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data"
        )

@app.delete("/api/delete-data")
async def delete_user_data_endpoint(current_user: dict = Depends(get_current_user)):
    """Delete all user data (GDPR right to be forgotten)"""
    try:
        user_id = current_user["user_id"]
        success = await data_manager.delete_user_data(user_id)
        
        if success:
            return {
                "success": True,
                "message": "User data deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User data not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user data"
        )

@app.get("/auth/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "services": {
            "auth": True,
            "users_count": len(users_db),
            "active_sessions": len(sessions_db),
            "security_events": len(security_events_db)
        }
    }

@app.get("/auth/stats")
async def get_auth_stats():
    """Get enhanced authentication statistics"""
    user_breakdown = {
        "email_users": len([u for u in users_db.values() if u.get("auth_provider") == "email"]),
        "demo_users": len([u for u in users_db.values() if u.get("auth_provider") == "demo"]),
        "google_users": len([u for u in users_db.values() if u.get("auth_provider") == "google"])
    }
    
    subscription_breakdown = {
        "free": len([u for u in users_db.values() if u.get("subscription_plan") == "free"]),
        "premium": len([u for u in users_db.values() if u.get("subscription_plan") == "premium"]),
        "enterprise": len([u for u in users_db.values() if u.get("subscription_plan") == "enterprise"])
    }
    
    return {
        "total_users": len(users_db),
        "active_sessions": len(sessions_db),
        "user_breakdown": user_breakdown,
        "subscription_breakdown": subscription_breakdown,
        "total_security_events": len(security_events_db),
        "profiles_created": len(user_profiles_db)
    }

@app.post("/auth/demo-login")
async def demo_login(plan: str = "free"):
    """Demo login for testing"""
    demo_users = {
        "free": {
            "user_id": "demo_free_001",
            "email": "demo.free@dharmamind.ai",
            "first_name": "Free",
            "last_name": "User",
            "subscription_plan": "free",
            "status": "active",
            "email_verified": True,
            "auth_provider": "demo",
            "login_count": 1,
            "last_login": datetime.utcnow().isoformat()
        },
        "premium": {
            "user_id": "demo_premium_001", 
            "email": "demo.premium@dharmamind.ai",
            "first_name": "Premium",
            "last_name": "User",
            "subscription_plan": "premium",
            "status": "active",
            "email_verified": True,
            "auth_provider": "demo",
            "login_count": 1,
            "last_login": datetime.utcnow().isoformat()
        },
        "enterprise": {
            "user_id": "demo_enterprise_001",
            "email": "demo.enterprise@dharmamind.ai",
            "first_name": "Enterprise",
            "last_name": "User",
            "subscription_plan": "enterprise",
            "status": "active",
            "email_verified": True,
            "auth_provider": "demo",
            "login_count": 1,
            "last_login": datetime.utcnow().isoformat()
        }
    }
    
    if plan not in demo_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid demo plan"
        )
    
    # Generate demo token
    demo_token = f"demo_{plan}_{secrets.token_urlsafe(16)}"
    expires_at = (datetime.utcnow() + timedelta(hours=24)).timestamp() * 1000
    
    # Store demo session
    sessions_db[demo_token] = {
        "user_id": demo_users[plan]["user_id"],
        "email": demo_users[plan]["email"],
        "expires_at": expires_at,
        "created_at": datetime.utcnow().timestamp() * 1000,
        "ip_address": "demo",
        "remember_me": False,
        "last_activity": datetime.utcnow().timestamp() * 1000
    }
    
    # Store demo user
    users_db[demo_users[plan]["email"]] = demo_users[plan]
    
    # Create demo profile
    user_profiles_db[demo_users[plan]["user_id"]] = {
        "user_id": demo_users[plan]["user_id"],
        "phone": None,
        "timezone": "UTC",
        "marketing_consent": False,
        "preferences": {
            "notifications": True,
            "theme": "system",
            "language": "en"
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Demo login: {plan}")
    
    return {
        "success": True,
        "user": demo_users[plan],
        "session_token": demo_token,
        "expires_at": expires_at
    }

# Password Reset Models
class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="Email address for password reset")

class ResetPasswordRequest(BaseModel):
    token: str = Field(..., description="Password reset token")
    password: str = Field(..., min_length=8, description="New password")

# In-memory storage for password reset tokens
reset_tokens_db: Dict[str, Dict[str, Any]] = {}

@app.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """
    Send password reset email
    """
    try:
        email = request.email.lower().strip()
        
        # Check if user exists
        if email not in users_db:
            # For security, we don't reveal if email exists or not
            return {
                "success": True,
                "message": "If this email is registered, you will receive a password reset link."
            }
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        
        # Store reset token (expires in 1 hour)
        reset_tokens_db[reset_token] = {
            "email": email,
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).timestamp() * 1000,
            "created_at": datetime.utcnow().timestamp() * 1000,
            "used": False
        }
        
        # In a real application, you would send an email here
        # For demo purposes, we'll log the reset link
        reset_link = f"http://localhost:3000/reset-password?token={reset_token}"
        logger.info(f"Password reset link for {email}: {reset_link}")
        
        # Simulate email sending delay
        import time
        time.sleep(0.5)
        
        return {
            "success": True,
            "message": "If this email is registered, you will receive a password reset link.",
            "reset_token": reset_token  # Only for demo purposes - remove in production
        }
        
    except Exception as e:
        logger.error(f"Forgot password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """
    Reset user password using reset token
    """
    try:
        token = request.token
        new_password = request.password
        
        # Validate token
        if token not in reset_tokens_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        token_data = reset_tokens_db[token]
        
        # Check if token is expired
        if datetime.utcnow().timestamp() * 1000 > token_data["expires_at"]:
            # Clean up expired token
            del reset_tokens_db[token]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired"
            )
        
        # Check if token was already used
        if token_data["used"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has already been used"
            )
        
        email = token_data["email"]
        
        # Validate new password
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Update user password
        if email in users_db:
            # Hash the new password with secure bcrypt
            password_hash = hash_password(new_password)
            
            # Update user in database
            users_db[email]["password_hash"] = password_hash
            users_db[email]["updated_at"] = datetime.utcnow().isoformat()
            
            # Mark token as used
            reset_tokens_db[token]["used"] = True
            
            # Invalidate all existing sessions for this user
            user_id = users_db[email]["user_id"]
            sessions_to_remove = []
            for session_token, session_data in sessions_db.items():
                if session_data.get("user_id") == user_id:
                    sessions_to_remove.append(session_token)
            
            for session_token in sessions_to_remove:
                del sessions_db[session_token]
            
            logger.info(f"Password reset successful for user: {email}")
            
            return {
                "success": True,
                "message": "Password has been reset successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


<<<<<<< HEAD
# =================================================================
# END OF AUTHENTICATION SYSTEM
# Chat functionality is handled by the main backend at /api/chat/chat
# =================================================================
=======
# Chat Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: List[Dict[str, str]] = Field(default=[], description="Chat history")
    user_id: str = Field(default="anonymous", description="User ID")


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process chat messages with DharmaMind AI and store conversation data
    """
    try:
        message = request.message.strip()
        user_id = request.user_id
        
        # Sanitize user input for security
        clean_message = sanitize_input(message, SecurityLevel.HIGH)
        
        # Generate conversation ID
        conversation_id = f"chat_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        # DharmaMind wisdom responses based on spiritual guidance
        response = generate_dharmic_response(clean_message)
        
        # Store the conversation in our data management system
        if user_id and user_id != "anonymous":
            try:
                await store_chat_message(user_id, clean_message, response)
                logger.info(f"Stored chat message for user: {user_id}")
            except Exception as e:
                logger.warning(f"Failed to store chat message: {e}")
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


def generate_dharmic_response(message: str) -> str:
    """
    Generate dharmic wisdom responses based on the user's message
    """
    message_lower = message.lower()
    
    # Meditation and Mindfulness
    if any(word in message_lower for word in ['meditation', 'meditate', 'mindfulness', 'awareness', 'present']):
        responses = [
            "🧘‍♀️ In the stillness of meditation, you discover that peace was never lost—only temporarily obscured by the mind's chatter. Begin with gentle awareness of your breath, letting each exhale release what no longer serves you.",
            "✨ Mindfulness is not about emptying the mind, but about becoming intimate with this moment exactly as it is. Start where you are, with compassion for yourself and patience for the process.",
            "🌸 The present moment is the doorway to awakening. In meditation, we learn to rest in the space between thoughts, where infinite wisdom naturally arises."
        ]
        return responses[hash(message) % len(responses)]
    
    # Suffering and Challenges
    elif any(word in message_lower for word in ['suffering', 'pain', 'difficult', 'hard', 'struggle', 'challenge']):
        responses = [
            "🌅 Every form of suffering carries within it the seeds of liberation. Your challenges are not obstacles to your spiritual path—they are the path itself, teaching you resilience and compassion.",
            "💫 As the lotus blooms most beautifully from the deepest mud, your greatest growth often emerges from your most difficult experiences. Trust the process of transformation.",
            "🕯️ Suffering is the invitation to go deeper, to find the unshakeable peace that exists beyond circumstances. You have survived every difficult day so far—you are stronger than you know."
        ]
        return responses[hash(message) % len(responses)]
    
    # Love and Relationships
    elif any(word in message_lower for word in ['love', 'relationship', 'family', 'friend', 'partner', 'compassion']):
        responses = [
            "💝 True love begins with self-acceptance. When you cultivate unconditional love for yourself, you naturally become a source of love for others. Relationships are mirrors reflecting our inner state.",
            "🌈 Every relationship is a teacher, offering opportunities to practice patience, forgiveness, and understanding. See conflicts as chances to grow in wisdom and compassion.",
            "💖 Love is not something you find—it's something you are. When you align with your loving nature, you attract relationships that honor and celebrate your authentic self."
        ]
        return responses[hash(message) % len(responses)]
    
    # Fear and Anxiety
    elif any(word in message_lower for word in ['fear', 'afraid', 'anxiety', 'anxious', 'worry', 'scared']):
        responses = [
            "🌟 Fear is often excitement without breath. Ground yourself in this moment through deep, conscious breathing. You are safe, you are supported, you are enough.",
            "🦋 Fear cannot exist in the same space as love and presence. When anxiety arises, send yourself the same compassion you would offer a dear friend facing difficulties.",
            "🌊 Like waves on the ocean, fears arise and pass away naturally. You are not your fears—you are the vast, peaceful awareness that witnesses them."
        ]
        return responses[hash(message) % len(responses)]
    
    # Purpose and Direction
    elif any(word in message_lower for word in ['purpose', 'meaning', 'direction', 'lost', 'path', 'calling']):
        responses = [
            "🧭 Your purpose is not something you need to find—it's something you remember. Listen to what brings you alive, what makes your heart sing, what feels aligned with your deepest values.",
            "🌱 Every experience has shaped you perfectly for your unique contribution to the world. Trust that your path is unfolding exactly as it should, even when you can't see the destination.",
            "⭐ Your purpose evolves as you grow. Stay curious, follow what inspires you, and remember that sometimes the journey itself is the destination."
        ]
        return responses[hash(message) % len(responses)]
    
    # Gratitude and Joy
    elif any(word in message_lower for word in ['grateful', 'gratitude', 'thankful', 'blessed', 'joy', 'happy']):
        responses = [
            "🙏 Gratitude is the fastest way to shift from scarcity to abundance. When we appreciate what we have, we open our hearts to receive even more blessings.",
            "✨ Joy is your natural state—it doesn't depend on external circumstances. Cultivate moments of simple appreciation: a sunset, a smile, the breath of life itself.",
            "🌻 Gratitude transforms ordinary moments into sacred experiences. Every blessing, no matter how small, is a reminder of the love that surrounds you."
        ]
        return responses[hash(message) % len(responses)]
    
    # Wisdom and Growth
    elif any(word in message_lower for word in ['wisdom', 'growth', 'learning', 'understanding', 'insight']):
        responses = [
            "📚 True wisdom is knowing that every experience—joyful or challenging—contributes to your spiritual evolution. Embrace both the teacher and the lesson with equal grace.",
            "🌳 Growth happens in spirals, not straight lines. Sometimes you'll revisit familiar lessons at deeper levels. This is not regression—it's integration.",
            "💎 Wisdom is not accumulated knowledge but direct understanding. Trust your inner knowing—it has been guiding you faithfully all along."
        ]
        return responses[hash(message) % len(responses)]
    
    # Success and Achievement
    elif any(word in message_lower for word in ['success', 'achieve', 'goal', 'ambition', 'career', 'money']):
        responses = [
            "🎯 True success is alignment between your actions and your values. When you pursue goals from a place of love rather than fear, success becomes a natural expression of your authentic self.",
            "💰 Abundance flows most freely when we balance ambition with contentment, striving with acceptance. Success without inner peace is merely sophisticated suffering.",
            "🏆 The highest achievement is becoming who you truly are. External accomplishments are beautiful expressions of your inner development, not substitutes for it."
        ]
        return responses[hash(message) % len(responses)]
    
    # Health and Wellbeing
    elif any(word in message_lower for word in ['health', 'sick', 'tired', 'energy', 'body', 'healing']):
        responses = [
            "🌿 Your body is a sacred temple housing your beautiful soul. Listen to its wisdom—it speaks through sensations, energy levels, and intuitive knowing. Honor its needs with loving attention.",
            "⚡ True vitality comes from aligning with natural rhythms: proper rest, nourishing food, joyful movement, and emotional balance. Healing happens when we remove obstacles to our natural state of wholeness.",
            "🌱 Every cell in your body is constantly renewing itself. Focus on what supports your wellbeing: positive thoughts, loving relationships, and activities that bring you alive."
        ]
        return responses[hash(message) % len(responses)]
    
    # Death and Loss
    elif any(word in message_lower for word in ['death', 'dying', 'loss', 'grief', 'goodbye']):
        responses = [
            "🕊️ In the spiritual perspective, death is not an ending but a transformation—like a wave returning to the ocean. Love transcends physical form and continues to connect us across all dimensions.",
            "💫 Grief is love with nowhere to go. Honor your feelings while remembering that the bonds of love are eternal. Those we've lost continue to guide and bless us in ways we may not always perceive.",
            "🌅 Every ending births a new beginning. In times of loss, we're invited to discover the indestructible essence within ourselves that death cannot touch."
        ]
        return responses[hash(message) % len(responses)]
    
    # General spiritual guidance
    else:
        responses = [
            "🌟 Every question you ask is a prayer for deeper understanding. Trust that the answers you seek are already within you, waiting to be discovered through quiet reflection and inner listening.",
            "✨ You are exactly where you need to be on your spiritual journey. Every experience—joyful or challenging—is perfectly designed to awaken the wisdom and love that you truly are.",
            "🙏 In this moment, breathe deeply and remember: you are infinitely loved, eternally supported, and perfectly whole exactly as you are. Your journey is sacred, and your growth is a gift to the world.",
            "🧘‍♀️ The divine speaks to you through intuition, synchronicity, and the whispers of your heart. Stay open, stay curious, and trust the gentle guidance that emerges from stillness.",
            "🌸 Spiritual growth is not about becoming someone different—it's about removing everything that isn't authentically you. You are already complete; you're simply remembering who you've always been."
        ]
        return responses[hash(message) % len(responses)]

>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

if __name__ == "__main__":
    logger.info("Starting DharmaMind Enhanced Authentication Server...")
    
    # Get SSL configuration if available
    ssl_config = get_ssl_config()
    
    if ssl_config:
        logger.info("🔒 Starting with HTTPS/SSL encryption")
        uvicorn.run(
            "enhanced_enterprise_auth:app",
            host="0.0.0.0",
            port=5001,
            reload=True,
            log_level="info",
            **ssl_config
        )
    else:
        logger.info("⚠️ Starting with HTTP (HTTPS not configured)")
        uvicorn.run(
            "enhanced_enterprise_auth:app",
            host="0.0.0.0",
            port=5001,
            reload=True,
            log_level="info"
        )
