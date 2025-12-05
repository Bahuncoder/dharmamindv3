"""Authentication routes - SECURE IMPLEMENTATION"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from ..security.authentication import (
    get_authenticator,
    UserAuthenticator,
    UserCreate,
    UserLogin
)
from ..security.jwt_manager import get_jwt_manager, JWTManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Registration request model"""
    username: str
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class PasswordChangeRequest(BaseModel):
    """Password change request model"""
    old_password: str
    new_password: str


def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """Dependency to get current user from JWT token"""
    if not jwt_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    
    token = credentials.credentials
    payload = jwt_manager.verify_token(token, expected_type="access")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    return username


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    authenticator: UserAuthenticator = Depends(get_authenticator),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    Register a new user
    
    - **username**: Unique username (3-50 chars, alphanumeric + _ -)
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, 3 of: lowercase, uppercase, digit, special)
    """
    if not jwt_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    
    try:
        # Create user data
        user_data = UserCreate(
            username=request.username,
            email=request.email,
            password=request.password
        )
        
        # Create user
        user = authenticator.create_user(user_data)
        
        # Generate tokens
        token_data = {"sub": user.username, "email": user.email}
        access_token = jwt_manager.create_access_token(token_data)
        refresh_token = jwt_manager.create_refresh_token(token_data)
        
        logger.info(f"✓ User registered successfully: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=jwt_manager.access_token_expire_minutes * 60
        )
        
    except ValueError as e:
        logger.warning(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    authenticator: UserAuthenticator = Depends(get_authenticator),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    User login endpoint with secure authentication
    
    - **username**: User's username
    - **password**: User's password
    
    Returns access and refresh tokens
    """
    if not jwt_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    
    # Authenticate user
    user = authenticator.authenticate_user(
        request.username,
        request.password
    )
    
    if not user:
        # Don't reveal if user exists or password is wrong
        logger.warning(f"Failed login attempt for: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if account is locked
    if user.locked_until:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=(
                "Account temporarily locked due to too many failed attempts. "
                "Please try again later."
            )
        )
    
    # Generate tokens
    token_data = {"sub": user.username, "email": user.email}
    access_token = jwt_manager.create_access_token(token_data)
    refresh_token = jwt_manager.create_refresh_token(token_data)
    
    logger.info(f"✓ User logged in successfully: {user.username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=jwt_manager.access_token_expire_minutes * 60
    )


@router.post("/logout")
async def logout(
    current_user: str = Depends(get_current_user_from_token),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    User logout endpoint - Revokes the current access token
    """
    if jwt_manager:
        # Revoke the token
        jwt_manager.revoke_token(credentials.credentials)
        logger.info(f"✓ User logged out: {current_user}")
    
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    Refresh access token using refresh token
    
    Provide a valid refresh token to get a new access token
    """
    if not jwt_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system not initialized"
        )
    
    # Verify refresh token
    token = credentials.credentials
    payload = jwt_manager.verify_token(token, expected_type="refresh")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate new access token
    username = payload.get("sub")
    email = payload.get("email")
    
    token_data = {"sub": username, "email": email}
    new_access_token = jwt_manager.create_access_token(token_data)
    
    logger.info(f"✓ Token refreshed for user: {username}")
    
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=token,  # Return same refresh token
        expires_in=jwt_manager.access_token_expire_minutes * 60
    )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: str = Depends(get_current_user_from_token),
    authenticator: UserAuthenticator = Depends(get_authenticator)
):
    """
    Change user password
    
    Requires:
    - Valid access token
    - Current password
    - New password meeting strength requirements
    """
    success = authenticator.change_password(
        current_user,
        request.old_password,
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password change failed. Check your current password."
        )
    
    logger.info(f"✓ Password changed for user: {current_user}")
    return {"message": "Password changed successfully"}


@router.get("/me")
async def get_current_user(
    current_user: str = Depends(get_current_user_from_token),
    authenticator: UserAuthenticator = Depends(get_authenticator)
):
    """
    Get current user information
    
    Requires valid access token
    """
    user = authenticator.get_user(current_user)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }

