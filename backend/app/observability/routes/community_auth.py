"""
Cross-Domain SSO Authentication for DharmaMind Community
Handles token exchange and validation between different DharmaMind platforms
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import jwt
import os
import requests

router = APIRouter()
security = HTTPBearer()

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
MAIN_PLATFORM_URL = os.getenv("MAIN_PLATFORM_URL", "https://dharmamind.com")
ALGORITHM = "HS256"

class TokenExchangeRequest(BaseModel):
    auth_code: str
    platform: str = "community"

class SSOVerifyRequest(BaseModel):
    token: str
    platform: str = "community"

class CommunityUser(BaseModel):
    user_id: str
    email: str
    first_name: str
    last_name: str
    subscription_plan: str
    status: str
    email_verified: bool
    auth_provider: str
    community_role: str = "member"
    joined_date: str
    last_login: str

@router.post("/auth/exchange-token")
async def exchange_auth_code(request: TokenExchangeRequest):
    """
    Exchange auth code from main platform for community access token
    """
    try:
        # Verify auth code with main platform
        verification_response = requests.post(
            f"{MAIN_PLATFORM_URL}/api/auth/verify-code",
            json={
                "auth_code": request.auth_code,
                "platform": request.platform
            },
            timeout=10
        )
        
        if not verification_response.ok:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication code"
            )
        
        user_data = verification_response.json()
        
        # Create community-specific token
        community_token = create_community_token(user_data["user"])
        
        # Sync user with community database
        await sync_community_user(user_data["user"])
        
        return {
            "success": True,
            "token": community_token,
            "user": map_to_community_user(user_data["user"]),
            "expires_at": (datetime.utcnow() + timedelta(days=7)).timestamp()
        }
        
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@router.get("/auth/verify")
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify and validate community access token
    """
    try:
        # Decode and validate JWT token
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        
        # Get user from community database
        user = await get_community_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update last seen
        await update_user_last_seen(user_id)
        
        return {
            "success": True,
            "user": user,
            "expires_at": payload.get("exp")
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.get("/auth/me")
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current authenticated user information
    """
    try:
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        
        user_id = payload.get("user_id")
        user = await get_community_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "user": user
        }
        
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

@router.post("/auth/logout")
async def logout_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout user and invalidate token
    """
    try:
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        
        user_id = payload.get("user_id")
        
        # Add token to blacklist (implement as needed)
        await blacklist_token(credentials.credentials)
        
        # Update user logout time
        await update_user_logout(user_id)
        
        return {"success": True, "message": "Logged out successfully"}
        
    except jwt.JWTError:
        return {"success": True, "message": "Already logged out"}

@router.post("/community/sync-user")
async def sync_user_profile(
    user_data: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Sync user profile with community platform
    """
    try:
        # Verify token
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        
        # Sync user data
        await sync_community_user(user_data)
        
        return {"success": True, "message": "User profile synced"}
        
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Helper functions
def create_community_token(user: dict) -> str:
    """Create JWT token for community access"""
    payload = {
        "user_id": user.get("user_id") or user.get("id"),
        "email": user.get("email"),
        "platform": "community",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)

def map_to_community_user(user: dict) -> dict:
    """Map main platform user to community user format"""
    return {
        "user_id": user.get("user_id") or user.get("id"),
        "email": user.get("email"),
        "first_name": user.get("first_name") or user.get("name", "").split()[0],
        "last_name": user.get("last_name") or " ".join(user.get("name", "").split()[1:]),
        "subscription_plan": user.get("subscription_plan") or user.get("plan", "basic"),
        "status": user.get("status", "active"),
        "email_verified": user.get("email_verified", False),
        "auth_provider": user.get("auth_provider", "dharmamind"),
        "community_role": user.get("community_role", "member"),
        "joined_date": user.get("created_at") or datetime.utcnow().isoformat(),
        "last_login": datetime.utcnow().isoformat()
    }

async def sync_community_user(user: dict):
    """Sync user with community database"""
    # Implement database sync logic here
    # This would typically update/create user in community database
    pass

async def get_community_user(user_id: str) -> Optional[dict]:
    """Get user from community database"""
    # Implement database query here
    # Return user data or None if not found
    return None

async def update_user_last_seen(user_id: str):
    """Update user's last seen timestamp"""
    # Implement database update here
    pass

async def update_user_logout(user_id: str):
    """Update user's logout timestamp"""
    # Implement database update here
    pass

async def blacklist_token(token: str):
    """Add token to blacklist"""
    # Implement token blacklisting here
    pass
