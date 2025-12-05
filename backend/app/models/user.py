"""
User model for DharmaMind platform

Defines the core user entity with authentication and profile information.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User roles in the system"""
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    RISHI = "rishi"  # Advanced spiritual guide

class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

class User(BaseModel):
    """Core user model"""
    id: str = Field(..., description="User unique identifier")
    email: EmailStr = Field(..., description="User email address")
    username: Optional[str] = Field(default=None, description="Username")
    full_name: Optional[str] = Field(default=None, description="User's full name")
    
    # Account status
    role: UserRole = Field(default=UserRole.USER, description="User role")
    status: UserStatus = Field(default=UserStatus.PENDING_VERIFICATION, description="Account status")
    is_verified: bool = Field(default=False, description="Email verification status")
    is_active: bool = Field(default=True, description="Account active status")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    
    # Profile information
    avatar_url: Optional[str] = Field(default=None, description="Profile avatar URL")
    bio: Optional[str] = Field(default=None, description="User biography")
    location: Optional[str] = Field(default=None, description="User location")
    timezone: Optional[str] = Field(default="UTC", description="User timezone")
    language: str = Field(default="en", description="Preferred language")
    
    # Spiritual journey
    spiritual_level: Optional[str] = Field(default="beginner", description="Spiritual development level")
    meditation_experience: Optional[str] = Field(default="none", description="Meditation experience level")
    interests: List[str] = Field(default_factory=list, description="Spiritual interests")
    
    # Privacy and preferences
    privacy_settings: Dict[str, Any] = Field(default_factory=dict, description="Privacy preferences")
    notification_preferences: Dict[str, bool] = Field(default_factory=dict, description="Notification settings")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    username: Optional[str] = Field(default=None, description="Desired username")
    full_name: Optional[str] = Field(default=None, description="User's full name")
    language: str = Field(default="en", description="Preferred language")

class UserUpdate(BaseModel):
    """User update model"""
    username: Optional[str] = Field(default=None, description="Username")
    full_name: Optional[str] = Field(default=None, description="Full name")
    bio: Optional[str] = Field(default=None, description="Biography")
    location: Optional[str] = Field(default=None, description="Location")
    timezone: Optional[str] = Field(default=None, description="Timezone")
    language: Optional[str] = Field(default=None, description="Language preference")
    spiritual_level: Optional[str] = Field(default=None, description="Spiritual level")
    meditation_experience: Optional[str] = Field(default=None, description="Meditation experience")
    interests: Optional[List[str]] = Field(default=None, description="Interests")

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(default=False, description="Remember login session")

class UserResponse(BaseModel):
    """User response model (without sensitive data)"""
    id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="Email address")
    username: Optional[str] = Field(default=None, description="Username")
    full_name: Optional[str] = Field(default=None, description="Full name")
    role: UserRole = Field(..., description="User role")
    status: UserStatus = Field(..., description="Account status")
    is_verified: bool = Field(..., description="Verification status")
    avatar_url: Optional[str] = Field(default=None, description="Avatar URL")
    spiritual_level: Optional[str] = Field(default=None, description="Spiritual level")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login")