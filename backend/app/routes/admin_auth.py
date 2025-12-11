"""
Secure Admin Authentication API for DharmaMind
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
import os
import json

router = APIRouter()
security = HTTPBearer()

# Secure password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: Optional[str] = None
    expires_in: Optional[int] = None

# Secure admin credentials storage (in production, use proper database)
ADMIN_USERS_FILE = "admin_users.json"

def load_admin_users():
    """Load admin users from secure storage"""
    if os.path.exists(ADMIN_USERS_FILE):
        try:
            with open(ADMIN_USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_admin_users(users):
    """Save admin users to secure storage"""
    with open(ADMIN_USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_admin_token(email: str) -> str:
    """Create a secure JWT token for admin"""
    token_data = {
        "sub": email,
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        
        if email is None or role != "admin":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin token"
            )
        
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )

<<<<<<< HEAD
@router.post("/login", response_model=AdminLoginResponse, summary="Admin Login")
async def admin_login(credentials: AdminLoginRequest):
    """
    Authenticate admin user and return access token
    
    - **email**: Admin email address
    - **password**: Admin password
    
    Returns access token for authenticated admin
    """
    try:
        admin_users = load_admin_users()
        
        # Initialize admin users if none exist
        if not admin_users:
            initialize_admin_users()
            admin_users = load_admin_users()
        
        # Check if user exists and is active
        if credentials.email not in admin_users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        admin_user = admin_users[credentials.email]
        
        if not admin_user.get("active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Verify password
        if not pwd_context.verify(credentials.password, admin_user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create JWT token
        access_token_expires = timedelta(hours=JWT_EXPIRY_HOURS)
        token_data = {
            "sub": credentials.email,
            "email": credentials.email,
            "name": admin_user.get("name", "Administrator"),
            "role": admin_user.get("role", "admin"),
            "exp": datetime.utcnow() + access_token_expires
        }
        
        access_token = jwt.encode(token_data, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Update last login
        admin_user["last_login"] = datetime.now().isoformat()
        admin_users[credentials.email] = admin_user
        save_admin_users(admin_users)
        
        return AdminLoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            admin_info=AdminInfo(
                email=credentials.email,
                name=admin_user.get("name", "Administrator"),
                role=admin_user.get("role", "admin")
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )
=======
@router.post("/auth", response_model=AdminLoginResponse)
async def admin_login(login_data: AdminLoginRequest):
    """Secure admin authentication endpoint"""
    admin_users = load_admin_users()
    
    # Check if admin user exists
    if login_data.email not in admin_users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials"
        )
    
    admin_user = admin_users[login_data.email]
    
    # Verify password using secure bcrypt verification
    if not pwd_context.verify(login_data.password, admin_user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials"
        )
    
    # Check if admin account is active
    if not admin_user.get("active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin account is deactivated"
        )
    
    # Generate secure admin token
    token = create_admin_token(login_data.email)
    
    # Update last login
    admin_users[login_data.email]["last_login"] = datetime.now().isoformat()
    save_admin_users(admin_users)
    
    return AdminLoginResponse(
        success=True,
        token=token,
        message="Admin authentication successful",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    )
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc

@router.get("/verify")
async def verify_admin_session(current_admin: dict = Depends(verify_admin_token)):
    """Verify admin session and return admin info"""
    return {
        "success": True,
        "admin": {
            "email": current_admin["sub"],
            "role": current_admin["role"]
        },
        "message": "Admin session valid"
    }

@router.post("/logout")
async def admin_logout(current_admin: dict = Depends(verify_admin_token)):
    """Admin logout endpoint"""
    # In a real implementation, you might want to blacklist the token
    return {
        "success": True,
        "message": "Admin logged out successfully"
    }

# Initialize default admin user if not exists
def initialize_admin_users():
    """Initialize default admin users with secure passwords"""
    admin_users = load_admin_users()
    
    if not admin_users:
<<<<<<< HEAD
        # Create default admin with secure password (truncate to 72 bytes for bcrypt)
        secure_password = "SecureAdminPassword2025!"[:72]  # Truncate for bcrypt compatibility
        default_admin = {
            "admin@dharmamind.com": {
                "password_hash": pwd_context.hash(secure_password),
=======
        # Create default admin with secure password
        default_admin = {
            "admin@dharmamind.com": {
                "password_hash": pwd_context.hash("SecureAdminPassword2025!"),
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                "name": "System Administrator",
                "role": "admin",
                "active": True,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
        }
        save_admin_users(default_admin)
        print("🔐 Default admin user created with secure password")
        print("📧 Email: admin@dharmamind.com")
        print("🔑 Password: SecureAdminPassword2025!")
        print("⚠️  Please change this password immediately after first login")

<<<<<<< HEAD
# Note: Admin users will be initialized on first login attempt
=======
# Initialize admin users on import
initialize_admin_users()
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
