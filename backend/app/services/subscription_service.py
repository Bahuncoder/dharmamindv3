"""
Subscription Service for DharmaMind platform

Manages user subscriptions, billing, and feature access control
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field

from ..models.subscription import SubscriptionTier, UserSubscription

logger = logging.getLogger(__name__)

class SubscriptionStatus(str, Enum):
    """Subscription status options"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"
    SUSPENDED = "suspended"

class BillingCycle(str, Enum):
    """Billing cycle options"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    LIFETIME = "lifetime"

class FeatureAccess(BaseModel):
    """Feature access permissions"""
    feature_name: str = Field(..., description="Feature name")
    has_access: bool = Field(..., description="Whether user has access")
    usage_limit: Optional[int] = Field(default=None, description="Usage limit if applicable")
    current_usage: int = Field(default=0, description="Current usage count")
    
class SubscriptionLimits(BaseModel):
    """Subscription usage limits"""
    daily_messages: Optional[int] = Field(default=None, description="Daily message limit")
    monthly_sessions: Optional[int] = Field(default=None, description="Monthly session limit")
    storage_mb: Optional[int] = Field(default=None, description="Storage limit in MB")
    advanced_features: bool = Field(default=False, description="Access to advanced features")
    priority_support: bool = Field(default=False, description="Priority support access")

class SubscriptionService:
    """Service for managing user subscriptions"""
    
    def __init__(self):
        self.tier_features = {
            SubscriptionTier.FREE: {
                "daily_messages": 10,
                "monthly_sessions": 5,
                "storage_mb": 100,
                "advanced_features": False,
                "priority_support": False
            },
            SubscriptionTier.BASIC: {
                "daily_messages": 50,
                "monthly_sessions": 20,
                "storage_mb": 500,
                "advanced_features": True,
                "priority_support": False
            },
            SubscriptionTier.PREMIUM: {
                "daily_messages": 200,
                "monthly_sessions": 100,
                "storage_mb": 2000,
                "advanced_features": True,
                "priority_support": True
            },
            SubscriptionTier.ENTERPRISE: {
                "daily_messages": None,  # Unlimited
                "monthly_sessions": None,  # Unlimited
                "storage_mb": 10000,
                "advanced_features": True,
                "priority_support": True
            }
        }
        
        self.user_subscriptions: Dict[str, UserSubscription] = {}
        self.usage_tracking: Dict[str, Dict[str, int]] = {}
    
    async def initialize(self) -> bool:
        """Initialize the subscription service"""
        try:
            logger.info("Subscription service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize subscription service: {e}")
            return False
    
    async def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Get user's current subscription"""
        try:
            # In production, this would query the database
            subscription = self.user_subscriptions.get(user_id)
            
            if not subscription:
                # Create default free subscription
                subscription = UserSubscription(
                    user_id=user_id,
                    tier=SubscriptionTier.FREE,
                    status=SubscriptionStatus.ACTIVE.value,
                    start_date=datetime.now(),
                    billing_cycle=BillingCycle.MONTHLY.value
                )
                self.user_subscriptions[user_id] = subscription
            
            return subscription
        except Exception as e:
            logger.error(f"Error getting user subscription: {e}")
            return None
    
    async def upgrade_subscription(
        self,
        user_id: str,
        new_tier: SubscriptionTier,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY
    ) -> bool:
        """Upgrade user subscription"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if not subscription:
                return False
            
            # Update subscription
            subscription.tier = new_tier
            subscription.billing_cycle = billing_cycle.value
            subscription.status = SubscriptionStatus.ACTIVE.value
            subscription.updated_at = datetime.now()
            
            # Set expiry based on billing cycle
            if billing_cycle == BillingCycle.MONTHLY:
                subscription.expires_at = datetime.now() + timedelta(days=30)
            elif billing_cycle == BillingCycle.QUARTERLY:
                subscription.expires_at = datetime.now() + timedelta(days=90)
            elif billing_cycle == BillingCycle.YEARLY:
                subscription.expires_at = datetime.now() + timedelta(days=365)
            
            self.user_subscriptions[user_id] = subscription
            logger.info(f"Upgraded subscription for user {user_id} to {new_tier}")
            return True
            
        except Exception as e:
            logger.error(f"Error upgrading subscription: {e}")
            return False
    
    async def cancel_subscription(self, user_id: str) -> bool:
        """Cancel user subscription"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if not subscription:
                return False
            
            subscription.status = SubscriptionStatus.CANCELLED.value
            subscription.updated_at = datetime.now()
            
            logger.info(f"Cancelled subscription for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling subscription: {e}")
            return False
    
    async def check_feature_access(self, user_id: str, feature_name: str) -> FeatureAccess:
        """Check if user has access to a specific feature"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if not subscription:
                return FeatureAccess(
                    feature_name=feature_name,
                    has_access=False
                )
            
            tier_features = self.tier_features.get(subscription.tier, {})
            feature_config = tier_features.get(feature_name)
            
            if feature_config is None:
                # Feature not defined for this tier
                return FeatureAccess(
                    feature_name=feature_name,
                    has_access=False
                )
            
            # Check if it's a boolean feature
            if isinstance(feature_config, bool):
                return FeatureAccess(
                    feature_name=feature_name,
                    has_access=feature_config
                )
            
            # Check usage limits
            current_usage = self._get_current_usage(user_id, feature_name)
            
            return FeatureAccess(
                feature_name=feature_name,
                has_access=feature_config is None or current_usage < feature_config,
                usage_limit=feature_config,
                current_usage=current_usage
            )
            
        except Exception as e:
            logger.error(f"Error checking feature access: {e}")
            return FeatureAccess(
                feature_name=feature_name,
                has_access=False
            )
    
    async def track_usage(self, user_id: str, feature_name: str, amount: int = 1) -> bool:
        """Track feature usage for a user"""
        try:
            if user_id not in self.usage_tracking:
                self.usage_tracking[user_id] = {}
            
            if feature_name not in self.usage_tracking[user_id]:
                self.usage_tracking[user_id][feature_name] = 0
            
            self.usage_tracking[user_id][feature_name] += amount
            return True
            
        except Exception as e:
            logger.error(f"Error tracking usage: {e}")
            return False
    
    def _get_current_usage(self, user_id: str, feature_name: str) -> int:
        """Get current usage for a feature"""
        return self.usage_tracking.get(user_id, {}).get(feature_name, 0)
    
    async def get_subscription_limits(self, user_id: str) -> SubscriptionLimits:
        """Get subscription limits for a user"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if not subscription:
                return SubscriptionLimits()
            
            tier_features = self.tier_features.get(subscription.tier, {})
            
            return SubscriptionLimits(
                daily_messages=tier_features.get("daily_messages"),
                monthly_sessions=tier_features.get("monthly_sessions"),
                storage_mb=tier_features.get("storage_mb"),
                advanced_features=tier_features.get("advanced_features", False),
                priority_support=tier_features.get("priority_support", False)
            )
            
        except Exception as e:
            logger.error(f"Error getting subscription limits: {e}")
            return SubscriptionLimits()
    
    async def is_subscription_active(self, user_id: str) -> bool:
        """Check if user has an active subscription"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if not subscription:
                return False
            
            if subscription.status != SubscriptionStatus.ACTIVE.value:
                return False
            
            # Check expiry
            if subscription.expires_at and subscription.expires_at < datetime.now():
                # Mark as expired
                subscription.status = SubscriptionStatus.EXPIRED.value
                subscription.updated_at = datetime.now()
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking subscription status: {e}")
            return False
    
    async def process_payment(
        self,
        user_id: str,
        amount: float,
        currency: str = "USD",
        payment_method: str = "stripe"
    ) -> bool:
        """Process subscription payment (mock implementation)"""
        try:
            # Mock payment processing
            logger.info(f"Processing payment for user {user_id}: {amount} {currency}")
            
            # In production, integrate with actual payment processor
            await asyncio.sleep(0.1)  # Simulate API call
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing payment: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            "status": "healthy",
            "service": "subscription",
            "active_subscriptions": len(self.user_subscriptions),
            "tracked_users": len(self.usage_tracking)
        }

# Global service instance
_subscription_service: Optional[SubscriptionService] = None

async def get_subscription_service() -> SubscriptionService:
    """Get the global subscription service instance"""
    global _subscription_service
    
    if _subscription_service is None:
        _subscription_service = SubscriptionService()
        await _subscription_service.initialize()
    
    return _subscription_service
