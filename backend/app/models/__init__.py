"""
DharmaMind Models Package

This package contains all data models and schemas used throughout
the DharmaMind platform for API requests, responses, and data validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ===============================
# CORE ENUMERATIONS
# ===============================

class MessageRole(str, Enum):
    """Message role in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SubscriptionTier(str, Enum):
    """Subscription tier levels"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ===============================
# USER MODELS
# ===============================

class UserProfile(BaseModel):
    """User profile model"""
    user_id: str
    email: str
    name: Optional[str] = None
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True


class User(BaseModel):
    """Basic user model"""
    id: str
    email: str
    name: str
    subscription_plan: str = "free"
    created_at: str


# ===============================
# CHAT MODELS
# ===============================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    model_config = {'protected_namespaces': ()}  # Allow 'model_' prefix in fields
    
    response: str
    conversation_id: str
    message_id: str
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None


# ===============================
# SUBSCRIPTION MODELS
# ===============================

class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"


class SubscriptionPlan(BaseModel):
    """Subscription plan details"""
    tier: SubscriptionTier
    name: str
    price_monthly: float
    price_yearly: float
    features: List[str]


class UserSubscription(BaseModel):
    """User subscription details"""
    user_id: str
    tier: SubscriptionTier
    status: SubscriptionStatus
    started_at: datetime
    expires_at: Optional[datetime] = None


# ===============================
# SYSTEM MODELS
# ===============================

class SystemHealth(BaseModel):
    """System health status"""
    status: str
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime = Field(default_factory=datetime.now)


__all__ = [
    # Enums
    "MessageRole",
    "SubscriptionTier",
    "SubscriptionStatus",
    "Priority",
    
    # User models
    "User",
    "UserProfile",
    
    # Chat models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    
    # Subscription models
    "SubscriptionPlan",
    "UserSubscription",
    
    # System models
    "SystemHealth",
]
