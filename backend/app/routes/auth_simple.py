"""
Simple Authentication Routes for DharmaMind API
Minimal functional authentication without complex dependencies
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

# Create router
router = APIRouter()

# Simple models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str

# Logger
logger = logging.getLogger("dharmamind.auth")

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Simple login endpoint for testing"""
    logger.info(f"Login attempt for: {request.email}")
    
    # Simple demo authentication
    if request.email == "demo@dharmamind.com" and request.password == "demo123":
        return TokenResponse(
            access_token="demo_token_12345",
            user_id="demo_user"
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password"
    )

@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Simple registration endpoint for testing"""
    logger.info(f"Registration attempt for: {request.email}")
    
    return TokenResponse(
        access_token="demo_token_67890",
        user_id="new_user"
    )

@router.get("/me")
async def get_current_user():
    """Get current user info"""
    return {
        "user_id": "demo_user",
        "email": "demo@dharmamind.com",
        "name": "Demo User",
        "role": "user"
    }

@router.post("/logout")
async def logout():
    """Logout endpoint"""
    return {"message": "Successfully logged out"}
