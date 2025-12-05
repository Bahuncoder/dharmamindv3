"""Admin authentication routes"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])


class AdminLoginRequest(BaseModel):
    """Admin login request"""
    username: str
    password: str
    admin_key: str


@router.post("/login")
async def admin_login(request: AdminLoginRequest):
    """Admin login endpoint"""
    logger.info(f"Admin login attempt: {request.username}")
    return {
        "access_token": "admin_placeholder_token",
        "token_type": "bearer",
        "role": "admin"
    }


@router.get("/status")
async def admin_status():
    """Check admin authentication status"""
    return {"authenticated": False, "role": None}
