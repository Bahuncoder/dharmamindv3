"""
Enhanced Authentication routes for DharmaMind API
Handles user authentication, registration, and session management with verification codes
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import jwt
from jwt import PyJWTError as JWTError
import secrets
import json
import os

from ..services.notification_service import send_verification_code

router = APIRouter()
security = HTTPBearer()

# Secure password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dharmamind-secure-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Simple user storage (in production, use a proper database)
USERS_FILE = "users.json"
VERIFICATION_FILE = "verifications.json"

class UserRegistration(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    plan: str = "free"
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements"""
        issues = []
        
        # Minimum length
        if len(v) < 8:
            issues.append("Password must be at least 8 characters")
        
        # Maximum length
        if len(v) > 128:
            issues.append("Password must not exceed 128 characters")
        
        # Require uppercase
        if not any(c.isupper() for c in v):
            issues.append("Password must contain at least one uppercase letter")
        
        # Require lowercase
        if not any(c.islower() for c in v):
            issues.append("Password must contain at least one lowercase letter")
        
        # Require digit
        if not any(c.isdigit() for c in v):
            issues.append("Password must contain at least one number")
        
        # Require special character
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        if not any(c in special_chars for c in v):
            issues.append("Password must contain at least one special character")
        
        # Check for common passwords
        common = ['password', '123456', 'qwerty', 'admin', 'letmein', 'welcome']
        if v.lower() in common:
            issues.append("Password is too common")
        
        if issues:
            raise ValueError("; ".join(issues))
        
        return v

class VerificationRequest(BaseModel):
    email: EmailStr
    code: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    email: str
    name: str
    subscription_plan: str = "free"
    created_at: str

class AuthResponse(BaseModel):
    success: bool
    user: Optional[User] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    message: Optional[str] = None

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_verifications():
    """Load pending verifications from JSON file"""
    if os.path.exists(VERIFICATION_FILE):
        try:
            with open(VERIFICATION_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_verifications(verifications):
    """Save pending verifications to JSON file"""
    with open(VERIFICATION_FILE, 'w') as f:
        json.dump(verifications, f, indent=2)

def hash_password(password: str) -> str:
    """Securely hash password using bcrypt with salt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT authentication token"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None or email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Check if token is expired (handled by jwt.decode)
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.post("/register", response_model=AuthResponse)
async def register_user(user_data: UserRegistration):
    """Register a new user (sends verification code)"""
    users = load_users()
    verifications = load_verifications()
    
    # Check if user already exists
    if user_data.email in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )
    
    # Generate 6-digit verification code
    code = str(secrets.randbelow(1000000)).zfill(6)
    
    # Store verification data
    verifications[user_data.email] = {
        "code": code,
        "expires": (datetime.now() + timedelta(minutes=10)).isoformat(),
        "data": user_data.dict()
    }
    save_verifications(verifications)
    
    # Send verification code
    await send_verification_code(user_data.email, user_data.phone, code, user_data.name)
    
    return AuthResponse(
        success=True,
        message=f"Verification code sent to {user_data.email}. Please verify to complete registration."
    )

@router.post("/verify", response_model=AuthResponse)
async def verify_code(verification_data: VerificationRequest):
    """Verify code and create account with subscription plan"""
    users = load_users()
    verifications = load_verifications()
    
    entry = verifications.get(verification_data.email)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No verification pending for this email"
        )
    
    if entry["code"] != verification_data.code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code"
        )
    
    if datetime.now() > datetime.fromisoformat(entry["expires"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code expired"
        )
    
    data = entry["data"]
    
    # Create new user with chosen subscription plan
    user_id = f"user_{secrets.token_hex(8)}"
    hashed_password = hash_password(data["password"])
    plan = data.get("plan", "free")
    
    user = {
        "id": user_id,
        "email": data["email"],
        "name": data["name"],
        "phone": data.get("phone"),
        "password": hashed_password,
        "subscription_plan": plan,
        "created_at": datetime.now().isoformat(),
        "verified": True
    }
    
    users[data["email"]] = user
    save_users(users)
    
    # Remove verification entry
    verifications.pop(verification_data.email)
    save_verifications(verifications)
    
    # Generate secure JWT tokens
    token_data = {"sub": user_id, "email": data["email"], "plan": plan}
    access_token = create_access_token(data=token_data)
    
    return AuthResponse(
        success=True,
        user=User(
            id=user_id,
            email=data["email"],
            name=data["name"],
            subscription_plan=plan,
            created_at=user["created_at"]
        ),
        access_token=access_token,
        refresh_token=access_token,
        message=f"Account created successfully with {plan} plan!"
    )


@router.post("/resend-code", response_model=AuthResponse)
async def resend_verification_code(email: EmailStr):
    """Resend verification code to email"""
    verifications = load_verifications()
    
    if email not in verifications:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending verification for this email"
        )
    
    # Generate new code
    code = str(secrets.randbelow(1000000)).zfill(6)
    
    # Update verification entry with new code and expiry
    verifications[email]["code"] = code
    verifications[email]["expires"] = (datetime.now() + timedelta(minutes=10)).isoformat()
    save_verifications(verifications)
    
    # Get user name from stored data
    name = verifications[email]["data"].get("name", "User")
    
    # Send new verification code
    await send_verification_code(email, None, code, name)
    
    return AuthResponse(
        success=True,
        message="New verification code sent to your email"
    )

@router.post("/login", response_model=AuthResponse)
async def login_user(login_data: UserLogin):
    """Login user with email and password"""
    users = load_users()
    
    # Check if user exists
    if login_data.email not in users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    user = users[login_data.email]
    # Verify password using secure verification
    if not verify_password(login_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate secure JWT tokens
    token_data = {"sub": user["id"], "email": login_data.email, "plan": user["subscription_plan"]}
    access_token = create_access_token(data=token_data)
    
    return AuthResponse(
        success=True,
        user=User(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            subscription_plan=user["subscription_plan"],
            created_at=user["created_at"]
        ),
        access_token=access_token,
        refresh_token=access_token,
        message="Login successful"
    )

@router.post("/logout")
async def logout_user(current_user: dict = Depends(verify_token)):
    """Logout user (demo implementation)"""
    return {"success": True, "message": "Logged out successfully"}

@router.post("/refresh")
async def refresh_token(refresh_data: dict):
    """Refresh access token"""
    refresh_token = refresh_data.get("refresh_token")
    
    if not refresh_token or not refresh_token.startswith("demo_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    try:
        # Verify and decode the current refresh token
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Generate new token with same data
        token_data = {
            "sub": payload["sub"], 
            "email": payload["email"], 
            "plan": payload.get("plan", "free")
        }
        new_token = create_access_token(data=token_data)
        
        return {
            "access_token": new_token,
            "refresh_token": new_token
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/demo-login", response_model=AuthResponse)
async def demo_login(plan: str = "free"):
    """Demo login for testing purposes"""
    # Create a temporary demo user
    demo_user_id = f"demo_{secrets.token_hex(8)}"
    demo_email = f"demo@{secrets.token_hex(4)}.com"
    demo_name = f"Demo User {secrets.token_hex(2)}"
    
    # Valid plans
    valid_plans = ["free", "premium", "enterprise"]
    if plan not in valid_plans:
        plan = "free"
    
    # Generate secure demo token
    token_data = {"sub": demo_user_id, "email": demo_email, "plan": plan}
    access_token = create_access_token(data=token_data)
    
    return AuthResponse(
        success=True,
        user=User(
            id=demo_user_id,
            email=demo_email,
            name=demo_name,
            subscription_plan=plan,
            created_at=datetime.now().isoformat()
        ),
        access_token=access_token,
        refresh_token=access_token,
        message="Demo login successful"
    )

async def get_current_user(current_user: dict = Depends(verify_token)):
    """Get current user information"""
    users = load_users()
    
    user_email = current_user["email"]
    if user_email in users:
        user = users[user_email]
        return User(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            subscription_plan=user["subscription_plan"],
            created_at=user["created_at"]
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found"
    )

@router.get("/me")
async def get_me(current_user: dict = Depends(verify_token)):
    """Get current user information"""
    return await get_current_user(current_user)


# ================================
# 🔐 SESSION MANAGEMENT ENDPOINTS
# ================================

@router.get("/sessions")
async def get_user_sessions(current_user: dict = Depends(verify_token)):
    """Get all active sessions for current user"""
    try:
        from ..security.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        sessions = session_manager.get_user_sessions(current_user["sub"])
        
        return {
            "success": True,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at,
                    "last_activity": s.last_activity,
                    "ip_address": s.ip_address[:20] + "..." if len(s.ip_address) > 20 else s.ip_address,
                    "user_agent": s.user_agent[:50] + "..." if len(s.user_agent) > 50 else s.user_agent,
                    "is_active": s.is_active
                }
                for s in sessions
            ],
            "total": len(sessions)
        }
    except ImportError:
        return {"success": True, "sessions": [], "total": 0, "message": "Session management not available"}


@router.post("/sessions/revoke/{session_id}")
async def revoke_session(session_id: str, current_user: dict = Depends(verify_token)):
    """Revoke a specific session"""
    try:
        from ..security.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        # Verify session belongs to user
        sessions = session_manager.get_user_sessions(current_user["sub"])
        if not any(s.session_id == session_id for s in sessions):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        session_manager.revoke_session(session_id, "user_revoked")
        
        return {"success": True, "message": "Session revoked successfully"}
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session management not available"
        )


@router.post("/sessions/revoke-all")
async def revoke_all_sessions(current_user: dict = Depends(verify_token)):
    """Revoke all sessions (logout everywhere)"""
    try:
        from ..security.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        session_manager.revoke_all_user_sessions(current_user["sub"], "logout_all")
        
        return {"success": True, "message": "All sessions revoked successfully"}
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session management not available"
        )


@router.post("/logout")
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(verify_token)
):
    """Logout and blacklist current token"""
    try:
        from ..security.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        token = credentials.credentials
        session_manager.blacklist_token(token, current_user["sub"], "logout")
        
        return {"success": True, "message": "Logged out successfully"}
    except ImportError:
        # Graceful degradation - just acknowledge logout
        return {"success": True, "message": "Logged out"}

