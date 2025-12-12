"""
Subscription models for DharmaMind platform

Defines subscription tiers, features, and billing models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class SubscriptionTier(str, Enum):
    """Available subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    RISHI = "rishi"  # Advanced spiritual guidance tier

class BillingCycle(str, Enum):
    """Billing cycle options"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    LIFETIME = "lifetime"

class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"
    TRIALING = "trialing"

class FeatureAccess(BaseModel):
    """Feature access definition"""
    feature_name: str = Field(..., description="Name of the feature")
    enabled: bool = Field(default=False, description="Whether feature is enabled")
    usage_limit: Optional[int] = Field(default=None, description="Usage limit per billing cycle")
    current_usage: int = Field(default=0, description="Current usage count")

class SubscriptionPlan(BaseModel):
    """Subscription plan definition"""
    tier: SubscriptionTier = Field(..., description="Subscription tier")
    name: str = Field(..., description="Plan display name")
    description: str = Field(..., description="Plan description")
    price_monthly: float = Field(default=0.0, description="Monthly price")
    price_yearly: float = Field(default=0.0, description="Yearly price")
    features: List[FeatureAccess] = Field(default_factory=list, description="Available features")
    
    # Limits
    max_conversations: Optional[int] = Field(default=None, description="Max conversations per month")
    max_messages: Optional[int] = Field(default=None, description="Max messages per month")
    max_wisdom_queries: Optional[int] = Field(default=None, description="Max wisdom queries per month")
    priority_support: bool = Field(default=False, description="Priority customer support")
    advanced_ai_models: bool = Field(default=False, description="Access to advanced AI models")
    
class UserSubscription(BaseModel):
    """User's subscription details"""
    user_id: str = Field(..., description="User identifier")
    subscription_id: str = Field(..., description="Subscription identifier")
    tier: SubscriptionTier = Field(..., description="Current subscription tier")
    status: SubscriptionStatus = Field(..., description="Subscription status")
    billing_cycle: BillingCycle = Field(..., description="Billing cycle")
    
    # Dates
    created_at: datetime = Field(default_factory=datetime.now, description="Subscription creation date")
    trial_start: Optional[datetime] = Field(default=None, description="Trial start date")
    trial_end: Optional[datetime] = Field(default=None, description="Trial end date")
    current_period_start: datetime = Field(..., description="Current billing period start")
    current_period_end: datetime = Field(..., description="Current billing period end")
    cancel_at: Optional[datetime] = Field(default=None, description="Scheduled cancellation date")
    cancelled_at: Optional[datetime] = Field(default=None, description="Actual cancellation date")
    
    # Usage tracking
    features: List[FeatureAccess] = Field(default_factory=list, description="Feature usage")
    usage_statistics: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    
    # Billing
    last_payment_date: Optional[datetime] = Field(default=None, description="Last payment date")
    next_payment_date: Optional[datetime] = Field(default=None, description="Next payment date")
    payment_method_id: Optional[str] = Field(default=None, description="Payment method identifier")

class SubscriptionUsage(BaseModel):
    """Subscription usage tracking"""
    user_id: str = Field(..., description="User identifier")
    subscription_id: str = Field(..., description="Subscription identifier")
    period_start: datetime = Field(..., description="Usage period start")
    period_end: datetime = Field(..., description="Usage period end")
    
    # Usage metrics
    conversations_count: int = Field(default=0, description="Number of conversations")
    messages_count: int = Field(default=0, description="Number of messages")
    wisdom_queries_count: int = Field(default=0, description="Number of wisdom queries")
    ai_model_usage: Dict[str, int] = Field(default_factory=dict, description="AI model usage by type")
    
    # Feature usage
    feature_usage: Dict[str, int] = Field(default_factory=dict, description="Feature usage counts")
    
    # Costs
    compute_costs: float = Field(default=0.0, description="Compute costs incurred")
    api_costs: float = Field(default=0.0, description="API costs incurred")
    storage_costs: float = Field(default=0.0, description="Storage costs incurred")

class SubscriptionChange(BaseModel):
    """Subscription change request"""
    user_id: str = Field(..., description="User identifier")
    current_tier: SubscriptionTier = Field(..., description="Current subscription tier")
    new_tier: SubscriptionTier = Field(..., description="Requested new tier")
    billing_cycle: Optional[BillingCycle] = Field(default=None, description="New billing cycle")
    effective_date: Optional[datetime] = Field(default=None, description="When change should take effect")
    proration: bool = Field(default=True, description="Whether to prorate the change")
    reason: Optional[str] = Field(default=None, description="Reason for change")
