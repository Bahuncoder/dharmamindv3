"""
🔐 Multi-Factor Authentication API Routes
Advanced MFA endpoints for DharmaMind security

Endpoints:
- POST /mfa/setup - Setup MFA for user
- POST /mfa/verify - Verify MFA token
- GET /mfa/status - Get MFA status
- POST /mfa/disable - Disable MFA
- POST /mfa/regenerate-backup - Generate new backup codes
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import json

from ...security.mfa_manager import (
    MFAManager,
    MFASetupRequest,
    MFAVerificationRequest,
    MFASetupResponse,
    MFAVerificationResponse,
    get_mfa_manager
)
from ..routes.auth import verify_token

router = APIRouter(prefix="/api/v1/mfa", tags=["Multi-Factor Authentication"])
security = HTTPBearer()

@router.post("/setup", response_model=MFASetupResponse)
async def setup_mfa(
    mfa_request: MFASetupRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mfa_manager: MFAManager = Depends(get_mfa_manager)
):
    """
    Setup Multi-Factor Authentication for user
    
    Supports:
    - TOTP (Time-based One-Time Password)
    - SMS backup (future implementation)  
    - Email backup (future implementation)
    """
    try:
        # Verify user token
        payload = verify_token(credentials)
        user_id = payload.get("sub")
        user_email = payload.get("email")
        
        if not user_id or not user_email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Check if MFA is already enabled
        if mfa_manager.is_mfa_enabled(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled for this user"
            )
        
        # Setup based on method
        if mfa_request.method == "totp":
            response = mfa_manager.setup_totp(user_id, user_email)
            return response
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported MFA method"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MFA setup failed: {str(e)}"
        )

@router.post("/verify", response_model=MFAVerificationResponse)
async def verify_mfa(
    verification_request: MFAVerificationRequest,
    request: Request,
    mfa_manager: MFAManager = Depends(get_mfa_manager)
):
    """
    Verify MFA token
    
    Supports:
    - TOTP tokens from authenticator apps
    - Backup codes
    - Trusted device tokens
    """
    try:
        user_id = verification_request.user_id
        token = verification_request.token
        method = verification_request.method
        
        # Get client info for logging
        ip_address = request.client.host if request.client else ""
        user_agent = request.headers.get("user-agent", "")
        
        success = False
        trusted_device_token = None
        
        # Verify based on method
        if method == "totp":
            success = mfa_manager.verify_totp(user_id, token)
        elif method == "backup":
            success = mfa_manager.verify_backup_code(user_id, token)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported verification method"
            )
        
        # Log attempt
        mfa_manager.log_mfa_attempt(
            user_id, method, success, ip_address, user_agent
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA token"
            )
        
        # Generate trusted device token if requested
        if verification_request.remember_device:
            device_info = {
                'ip_address': ip_address,
                'user_agent': user_agent,
                'verified_at': str(request.state.timestamp) if hasattr(request.state, 'timestamp') else ""
            }
            trusted_device_token = mfa_manager.generate_trusted_device_token(
                user_id, device_info
            )
        
        return MFAVerificationResponse(
            success=True,
            trusted_device_token=trusted_device_token,
            message="MFA verification successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MFA verification failed: {str(e)}"
        )

@router.get("/status")
async def get_mfa_status(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mfa_manager: MFAManager = Depends(get_mfa_manager)
):
    """Get MFA status for authenticated user"""
    try:
        # Verify user token
        payload = verify_token(credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        status_info = mfa_manager.get_mfa_status(user_id)
        return {
            "success": True,
            "mfa_status": status_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MFA status: {str(e)}"
        )

@router.post("/disable")
async def disable_mfa(
    verification_request: MFAVerificationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mfa_manager: MFAManager = Depends(get_mfa_manager)
):
    """Disable MFA for user (requires MFA verification)"""
    try:
        # Verify user token
        payload = verify_token(credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Verify MFA before disabling
        if verification_request.method == "totp":
            success = mfa_manager.verify_totp(user_id, verification_request.token)
        elif verification_request.method == "backup":
            success = mfa_manager.verify_backup_code(user_id, verification_request.token)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification method"
            )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="MFA verification required to disable MFA"
            )
        
        # Disable MFA by removing from database
        import sqlite3
        with sqlite3.connect(mfa_manager.db_path) as conn:
            conn.execute("DELETE FROM user_mfa WHERE user_id = ?", (user_id,))
        
        return {
            "success": True,
            "message": "MFA disabled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable MFA: {str(e)}"
        )

@router.post("/regenerate-backup")
async def regenerate_backup_codes(
    verification_request: MFAVerificationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mfa_manager: MFAManager = Depends(get_mfa_manager)
):
    """Generate new backup codes (requires MFA verification)"""
    try:
        # Verify user token
        payload = verify_token(credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Verify MFA before regenerating
        if verification_request.method == "totp":
            success = mfa_manager.verify_totp(user_id, verification_request.token)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TOTP verification required for backup code regeneration"
            )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="MFA verification failed"
            )
        
        # Generate new backup codes
        backup_codes = mfa_manager._generate_backup_codes()
        
        # Update database
        import sqlite3
        with sqlite3.connect(mfa_manager.db_path) as conn:
            conn.execute("""
                UPDATE user_mfa 
                SET backup_codes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (mfa_manager._encrypt_data(json.dumps(backup_codes)), user_id))
        
        return {
            "success": True,
            "backup_codes": backup_codes,
            "message": "New backup codes generated. Store them securely!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate backup codes: {str(e)}"
        )
