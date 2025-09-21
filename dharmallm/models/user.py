"""
User Model for DharmaMind Authentication System
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    # Spiritual profile fields
    spiritual_level = Column(String(50), default="beginner")
    dharmic_path = Column(String(100), nullable=True)
    meditation_experience = Column(Integer, default=0)  # in days
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Profile
    bio = Column(Text, nullable=True)
    location = Column(String(100), nullable=True)

class UserCreate(BaseModel):
    """User creation schema"""
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    spiritual_level: Optional[str] = "beginner"

class UserResponse(BaseModel):
    """User response schema"""
    id: int
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    spiritual_level: str
    dharmic_path: Optional[str] = None
    meditation_experience: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    """User update schema"""
    full_name: Optional[str] = None
    spiritual_level: Optional[str] = None
    dharmic_path: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
