"""
Enhanced Authentication routes for DharmaMind API
Handles user authentication, registration, and session management with verification codes
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets
import json
import os

router = APIRouter()
security = HTTPBearer()

# Simple user storage (in production, use a proper database)
USERS_FILE = "users.json"
VERIFICATION_FILE = "verifications.json"

class UserRegistration(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    plan: str = "free"

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

def send_verification_code(email: str, phone: Optional[str], code: str) -> bool:
    """Send verification code via email or SMS (stub implementation)"""
    # TODO: Replace with real email/SMS service
    print(f"🔔 [VERIFICATION] Sending code {code} to {email or phone}")
    return True

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token(user_id: str, email: str, plan: str) -> str:
    """Generate a simple demo token"""
    payload = {
        "user_id": user_id,
        "email": email,
        "plan": plan,
        "exp": (datetime.now() + timedelta(days=30)).isoformat()
    }
    # In production, use proper JWT
    import base64
    return f"demo_{base64.b64encode(json.dumps(payload).encode()).decode()}"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    token = credentials.credentials
    
    if token.startswith("demo_"):
        try:
            import base64
            payload = json.loads(base64.b64decode(token[5:]).decode())
            
            # Check if token is expired
            exp_time = datetime.fromisoformat(payload["exp"])
            if datetime.now() > exp_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return payload
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token format"
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
    send_verification_code(user_data.email, user_data.phone, code)
    
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
    
    # Generate tokens
    access_token = generate_token(user_id, data["email"], plan)
    
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
    hashed_password = hash_password(login_data.password)
    
    # Verify password
    if user["password"] != hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate tokens
    access_token = generate_token(user["id"], login_data.email, user["subscription_plan"])
    
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
        import base64
        payload = json.loads(base64.b64decode(refresh_token[5:]).decode())
        
        # Generate new token with same data
        new_token = generate_token(
            payload["user_id"], 
            payload["email"], 
            payload["plan"]
        )
        
        return {
            "access_token": new_token,
            "refresh_token": new_token
        }
    except Exception:
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
    
    # Generate demo token
    access_token = generate_token(demo_user_id, demo_email, plan)
    
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
