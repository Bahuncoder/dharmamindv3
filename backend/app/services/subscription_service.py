"""
💳 Subscription Service
======================

Manages user subscriptions and premium features for DharmaMind.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class SubscriptionTier(str, Enum):
    """Subscription tiers"""
    FREE = "free"
    PREMIUM = "premium"
    PREMIUM_PLUS = "premium_plus"
    LIFETIME = "lifetime"

class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    TRIAL = "trial"

class SubscriptionService:
    """💳 Subscription management service"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # In-memory storage for development
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # Subscription tiers and features
        self.tier_features = {
            SubscriptionTier.FREE: {
                "daily_messages": 10,
                "rishi_access": ["vasishtha"],
                "emotional_analysis": True,
                "basic_guidance": True,
                "premium_content": False,
                "priority_support": False
            },
            SubscriptionTier.PREMIUM: {
                "daily_messages": 100,
                "rishi_access": ["vasishtha", "atri", "bharadwaja"],
                "emotional_analysis": True,
                "basic_guidance": True,
                "premium_content": True,
                "priority_support": True,
                "advanced_insights": True
            },
            SubscriptionTier.PREMIUM_PLUS: {
                "daily_messages": 500,
                "rishi_access": ["vasishtha", "atri", "bharadwaja", "gautama", "jamadagni", "kashyapa", "bhrigu"],
                "emotional_analysis": True,
                "basic_guidance": True,
                "premium_content": True,
                "priority_support": True,
                "advanced_insights": True,
                "personalized_practices": True,
                "unlimited_sanskrit": True
            },
            SubscriptionTier.LIFETIME: {
                "daily_messages": -1,  # Unlimited
                "rishi_access": ["vasishtha", "atri", "bharadwaja", "gautama", "jamadagni", "kashyapa", "bhrigu"],
                "emotional_analysis": True,
                "basic_guidance": True,
                "premium_content": True,
                "priority_support": True,
                "advanced_insights": True,
                "personalized_practices": True,
                "unlimited_sanskrit": True,
                "exclusive_content": True
            }
        }
        
        self.logger.info("💳 Subscription service initialized")
    
    def get_user_subscription(self, user_id: str) -> Dict[str, Any]:
        """Get user's current subscription"""
        return self.subscriptions.get(user_id, {
            "user_id": user_id,
            "tier": SubscriptionTier.FREE,
            "status": SubscriptionStatus.ACTIVE,
            "start_date": datetime.now(),
            "end_date": None,
            "features": self.tier_features[SubscriptionTier.FREE]
        })
    
    def check_feature_access(self, user_id: str, feature: str) -> bool:
        """Check if user has access to a specific feature"""
        subscription = self.get_user_subscription(user_id)
        features = subscription.get("features", {})
        return features.get(feature, False)
    
    def get_remaining_messages(self, user_id: str) -> int:
        """Get remaining daily messages for user"""
        subscription = self.get_user_subscription(user_id)
        daily_limit = subscription["features"].get("daily_messages", 10)
        
        if daily_limit == -1:  # Unlimited
            return 999999
        
        # In a real implementation, track daily usage
        return daily_limit  # Simplified for development
    
    def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        duration_days: int = 30
    ) -> Dict[str, Any]:
        """Create a new subscription"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days) if duration_days > 0 else None
        
        subscription = {
            "user_id": user_id,
            "tier": tier,
            "status": SubscriptionStatus.ACTIVE,
            "start_date": start_date,
            "end_date": end_date,
            "features": self.tier_features[tier],
            "created_at": start_date
        }
        
        self.subscriptions[user_id] = subscription
        self.logger.info(f"Created {tier.value} subscription for user {user_id}")
        
        return subscription

# Global subscription service instance
_subscription_service: Optional[SubscriptionService] = None

def get_subscription_service() -> SubscriptionService:
    """Get global subscription service instance"""
    global _subscription_service
    if _subscription_service is None:
        _subscription_service = SubscriptionService()
    return _subscription_service

# Export commonly used classes and functions
__all__ = [
    'SubscriptionService',
    'SubscriptionTier',
    'SubscriptionStatus',
    'get_subscription_service'
]
