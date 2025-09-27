"""
Comprehensive Subscription Management Service for DharmaMind
==========================================================

Enterprise-grade subscription service handling multi-tier subscriptions,
upgrades, downgrades, trials, and feature management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models.subscription_models import (
    Subscription, SubscriptionTier, SubscriptionStatus, BillingCycle,
    CurrencyCode, SubscriptionChange, Usage
)
from .payment_service import PaymentService
from .usage_service import UsageService

logger = logging.getLogger(__name__)

class SubscriptionService:
    """
    Core subscription management service with enterprise features
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.payment_service = PaymentService(db_session)
        self.usage_service = UsageService(db_session)
        
        # Tier pricing configuration (in USD)
        self.tier_pricing = {
            SubscriptionTier.FREE: {
                BillingCycle.MONTHLY: 0.0,
                BillingCycle.ANNUALLY: 0.0,
                BillingCycle.LIFETIME: 0.0
            },
            SubscriptionTier.PREMIUM: {
                BillingCycle.MONTHLY: 19.99,
                BillingCycle.ANNUALLY: 199.99,  # 17% discount
                BillingCycle.LIFETIME: 999.99
            },
            SubscriptionTier.ENTERPRISE: {
                BillingCycle.MONTHLY: 99.99,
                BillingCycle.ANNUALLY: 999.99,  # 17% discount
                BillingCycle.LIFETIME: 4999.99
            },
            SubscriptionTier.FAMILY: {
                BillingCycle.MONTHLY: 29.99,
                BillingCycle.ANNUALLY: 299.99,
                BillingCycle.LIFETIME: 1499.99
            },
            SubscriptionTier.STUDENT: {
                BillingCycle.MONTHLY: 9.99,
                BillingCycle.ANNUALLY: 99.99,
                BillingCycle.LIFETIME: 499.99
            },
            SubscriptionTier.CORPORATE: {
                BillingCycle.MONTHLY: 199.99,
                BillingCycle.ANNUALLY: 1999.99,
                BillingCycle.LIFETIME: 9999.99
            }
        }
    
    async def create_subscription(
        self, 
        user_id: str,
        tier: SubscriptionTier,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        payment_method_id: Optional[str] = None,
        trial_days: int = 0,
        discount_code: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Subscription:
        """
        Create a new subscription with comprehensive setup
        """
        try:
            # Calculate pricing
            base_price = self.tier_pricing[tier][billing_cycle]
            current_price = base_price
            
            # Apply discount if provided
            if discount_code:
                discount = await self._apply_discount(discount_code, current_price)
                if discount:
                    current_price = max(0, current_price - discount)
            
            # Setup subscription
            subscription = Subscription(
                user_id=user_id,
                tier=tier,
                status=SubscriptionStatus.TRIALING if trial_days > 0 else SubscriptionStatus.ACTIVE,
                billing_cycle=billing_cycle,
                base_price=base_price,
                current_price=current_price,
                payment_method_id=payment_method_id,
                features_enabled=self._get_tier_features(tier),
                usage_limits=self._get_tier_limits(tier),
                metadata=metadata or {}
            )
            
            # Set dates
            now = datetime.utcnow()
            subscription.created_at = now
            
            if trial_days > 0:
                subscription.trial_start = now
                subscription.trial_end = now + timedelta(days=trial_days)
                subscription.current_period_start = subscription.trial_end
            else:
                subscription.activated_at = now
                subscription.current_period_start = now
            
            # Set period end based on billing cycle
            subscription.current_period_end = self._calculate_period_end(
                subscription.current_period_start, billing_cycle
            )
            
            # Save subscription
            self.db.add(subscription)
            self.db.commit()
            
            # Initialize usage tracking
            await self.usage_service.initialize_usage_tracking(subscription.id)
            
            logger.info(f"Created subscription {subscription.id} for user {user_id} with tier {tier}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {str(e)}")
            self.db.rollback()
            raise
    
    async def upgrade_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        immediate: bool = False
    ) -> Tuple[Subscription, float]:
        """
        Upgrade subscription to higher tier with proration
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            if not self._is_upgrade(subscription.tier, new_tier):
                raise ValueError("New tier must be higher than current tier")
            
            old_tier = subscription.tier
            old_price = subscription.current_price
            
            # Calculate new pricing
            new_price = self.tier_pricing[new_tier][subscription.billing_cycle]
            
            # Calculate proration
            proration_amount = 0.0
            if immediate and subscription.current_period_end:
                days_remaining = (subscription.current_period_end - datetime.utcnow()).days
                if subscription.billing_cycle == BillingCycle.MONTHLY:
                    daily_rate_old = old_price / 30
                    daily_rate_new = new_price / 30
                    proration_amount = (daily_rate_new - daily_rate_old) * days_remaining
                elif subscription.billing_cycle == BillingCycle.ANNUALLY:
                    daily_rate_old = old_price / 365
                    daily_rate_new = new_price / 365
                    proration_amount = (daily_rate_new - daily_rate_old) * days_remaining
            
            # Update subscription
            subscription.tier = new_tier
            subscription.current_price = new_price
            subscription.features_enabled = self._get_tier_features(new_tier)
            subscription.usage_limits = self._get_tier_limits(new_tier)
            subscription.updated_at = datetime.utcnow()
            
            # Record change
            change = SubscriptionChange(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                change_type="upgrade",
                from_tier=old_tier,
                to_tier=new_tier,
                old_price=old_price,
                new_price=new_price,
                proration_amount=proration_amount,
                effective_date=datetime.utcnow() if immediate else subscription.current_period_end,
                reason="user_request"
            )
            
            self.db.add(change)
            self.db.commit()
            
            logger.info(f"Upgraded subscription {subscription_id} from {old_tier} to {new_tier}")
            return subscription, proration_amount
            
        except Exception as e:
            logger.error(f"Failed to upgrade subscription: {str(e)}")
            self.db.rollback()
            raise
    
    async def downgrade_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        immediate: bool = False
    ) -> Subscription:
        """
        Downgrade subscription to lower tier
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            if not self._is_downgrade(subscription.tier, new_tier):
                raise ValueError("New tier must be lower than current tier")
            
            old_tier = subscription.tier
            old_price = subscription.current_price
            new_price = self.tier_pricing[new_tier][subscription.billing_cycle]
            
            # For downgrades, typically apply at next billing cycle
            effective_date = datetime.utcnow() if immediate else subscription.current_period_end
            
            if immediate:
                subscription.tier = new_tier
                subscription.current_price = new_price
                subscription.features_enabled = self._get_tier_features(new_tier)
                subscription.usage_limits = self._get_tier_limits(new_tier)
                subscription.updated_at = datetime.utcnow()
            else:
                # Schedule downgrade
                subscription.metadata = subscription.metadata or {}
                subscription.metadata['scheduled_downgrade'] = {
                    'new_tier': new_tier.value,
                    'effective_date': effective_date.isoformat(),
                    'new_price': new_price
                }
            
            # Record change
            change = SubscriptionChange(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                change_type="downgrade",
                from_tier=old_tier,
                to_tier=new_tier,
                old_price=old_price,
                new_price=new_price,
                effective_date=effective_date,
                reason="user_request"
            )
            
            self.db.add(change)
            self.db.commit()
            
            logger.info(f"Downgraded subscription {subscription_id} from {old_tier} to {new_tier}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to downgrade subscription: {str(e)}")
            self.db.rollback()
            raise
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
        reason: Optional[str] = None
    ) -> Subscription:
        """
        Cancel subscription with option for immediate or end-of-period
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            if subscription.status == SubscriptionStatus.CANCELLED:
                raise ValueError("Subscription already cancelled")
            
            # Set cancellation
            subscription.cancelled_at = datetime.utcnow()
            
            if immediate:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.ended_at = datetime.utcnow()
            else:
                # Cancel at end of current period
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.ended_at = subscription.current_period_end
                
            subscription.updated_at = datetime.utcnow()
            
            # Record change
            change = SubscriptionChange(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                change_type="cancel",
                from_tier=subscription.tier,
                effective_date=subscription.ended_at,
                reason=reason or "user_request"
            )
            
            self.db.add(change)
            
            # Cancel recurring payments
            if subscription.stripe_subscription_id:
                await self.payment_service.cancel_recurring_payment(
                    subscription.stripe_subscription_id
                )
            
            self.db.commit()
            
            logger.info(f"Cancelled subscription {subscription_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to cancel subscription: {str(e)}")
            self.db.rollback()
            raise
    
    async def pause_subscription(
        self,
        subscription_id: str,
        pause_until: Optional[datetime] = None
    ) -> Subscription:
        """
        Pause subscription temporarily
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            if subscription.status != SubscriptionStatus.ACTIVE:
                raise ValueError("Can only pause active subscriptions")
            
            subscription.status = SubscriptionStatus.PAUSED
            subscription.metadata = subscription.metadata or {}
            subscription.metadata['paused_at'] = datetime.utcnow().isoformat()
            if pause_until:
                subscription.metadata['pause_until'] = pause_until.isoformat()
            
            subscription.updated_at = datetime.utcnow()
            
            # Record change
            change = SubscriptionChange(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                change_type="pause",
                from_tier=subscription.tier,
                to_tier=subscription.tier,
                effective_date=datetime.utcnow(),
                reason="user_request",
                notes=f"Paused until {pause_until}" if pause_until else "Indefinite pause"
            )
            
            self.db.add(change)
            self.db.commit()
            
            logger.info(f"Paused subscription {subscription_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to pause subscription: {str(e)}")
            self.db.rollback()
            raise
    
    async def resume_subscription(self, subscription_id: str) -> Subscription:
        """
        Resume a paused subscription
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            if subscription.status != SubscriptionStatus.PAUSED:
                raise ValueError("Can only resume paused subscriptions")
            
            subscription.status = SubscriptionStatus.ACTIVE
            
            # Update period dates
            now = datetime.utcnow()
            subscription.current_period_start = now
            subscription.current_period_end = self._calculate_period_end(
                now, subscription.billing_cycle
            )
            
            # Clear pause metadata
            if subscription.metadata:
                subscription.metadata.pop('paused_at', None)
                subscription.metadata.pop('pause_until', None)
            
            subscription.updated_at = now
            
            # Record change
            change = SubscriptionChange(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                change_type="resume",
                from_tier=subscription.tier,
                to_tier=subscription.tier,
                effective_date=now,
                reason="user_request"
            )
            
            self.db.add(change)
            self.db.commit()
            
            logger.info(f"Resumed subscription {subscription_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to resume subscription: {str(e)}")
            self.db.rollback()
            raise
    
    def get_user_subscription(self, user_id: str) -> Optional[Subscription]:
        """
        Get active subscription for user
        """
        return self.db.query(Subscription).filter(
            and_(
                Subscription.user_id == user_id,
                Subscription.status.in_([
                    SubscriptionStatus.ACTIVE,
                    SubscriptionStatus.TRIALING,
                    SubscriptionStatus.PAST_DUE
                ])
            )
        ).first()
    
    def get_subscription_history(self, user_id: str) -> List[Subscription]:
        """
        Get all subscriptions for user
        """
        return self.db.query(Subscription).filter(
            Subscription.user_id == user_id
        ).order_by(desc(Subscription.created_at)).all()
    
    def check_feature_access(self, user_id: str, feature: str) -> bool:
        """
        Check if user has access to specific feature
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            # Default to free tier features
            free_features = self._get_tier_features(SubscriptionTier.FREE)
            return free_features.get(feature, False)
        
        features = subscription.get_tier_features()
        return features.get(feature, False)
    
    def get_usage_limits(self, user_id: str) -> Dict[str, Any]:
        """
        Get current usage limits for user
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return self._get_tier_limits(SubscriptionTier.FREE)
        
        return subscription.usage_limits or self._get_tier_limits(subscription.tier)
    
    async def process_subscription_renewals(self):
        """
        Process upcoming subscription renewals
        """
        try:
            # Find subscriptions due for renewal in next 24 hours
            tomorrow = datetime.utcnow() + timedelta(days=1)
            
            subscriptions = self.db.query(Subscription).filter(
                and_(
                    Subscription.status == SubscriptionStatus.ACTIVE,
                    Subscription.current_period_end <= tomorrow,
                    Subscription.current_period_end > datetime.utcnow()
                )
            ).all()
            
            for subscription in subscriptions:
                await self._process_renewal(subscription)
                
        except Exception as e:
            logger.error(f"Failed to process renewals: {str(e)}")
    
    async def _process_renewal(self, subscription: Subscription):
        """
        Process individual subscription renewal
        """
        try:
            # Check for scheduled changes
            if subscription.metadata and 'scheduled_downgrade' in subscription.metadata:
                scheduled = subscription.metadata['scheduled_downgrade']
                effective_date = datetime.fromisoformat(scheduled['effective_date'])
                
                if datetime.utcnow() >= effective_date:
                    # Apply scheduled downgrade
                    new_tier = SubscriptionTier(scheduled['new_tier'])
                    subscription.tier = new_tier
                    subscription.current_price = scheduled['new_price']
                    subscription.features_enabled = self._get_tier_features(new_tier)
                    subscription.usage_limits = self._get_tier_limits(new_tier)
                    
                    # Clear scheduled change
                    del subscription.metadata['scheduled_downgrade']
            
            # Process payment for next period
            if subscription.current_price > 0:
                payment_result = await self.payment_service.process_subscription_payment(
                    subscription.id,
                    subscription.current_price,
                    subscription.currency
                )
                
                if not payment_result.success:
                    subscription.status = SubscriptionStatus.PAST_DUE
                    logger.warning(f"Payment failed for subscription {subscription.id}")
                    return
            
            # Update period dates
            subscription.current_period_start = subscription.current_period_end
            subscription.current_period_end = self._calculate_period_end(
                subscription.current_period_start, subscription.billing_cycle
            )
            subscription.updated_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"Renewed subscription {subscription.id}")
            
        except Exception as e:
            logger.error(f"Failed to renew subscription {subscription.id}: {str(e)}")
            self.db.rollback()
    
    def _get_tier_features(self, tier: SubscriptionTier) -> Dict[str, Any]:
        """
        Get default features for tier
        """
        return {
            SubscriptionTier.FREE: {
                "daily_messages": 10,
                "ai_models": ["basic"],
                "chat_history_days": 7,
                "export_chats": False,
                "priority_support": False,
                "advanced_analytics": False,
                "custom_prompts": False,
                "api_access": False
            },
            SubscriptionTier.PREMIUM: {
                "daily_messages": 200,
                "ai_models": ["basic", "advanced"],
                "chat_history_days": 90,
                "export_chats": True,
                "priority_support": True,
                "advanced_analytics": True,
                "custom_prompts": True,
                "api_access": "limited"
            },
            SubscriptionTier.ENTERPRISE: {
                "daily_messages": -1,  # unlimited
                "ai_models": ["basic", "advanced", "premium"],
                "chat_history_days": -1,  # unlimited
                "export_chats": True,
                "priority_support": True,
                "advanced_analytics": True,
                "custom_prompts": True,
                "api_access": "full",
                "team_management": True,
                "white_labeling": True
            }
        }.get(tier, {})
    
    def _get_tier_limits(self, tier: SubscriptionTier) -> Dict[str, Any]:
        """
        Get usage limits for tier
        """
        return {
            SubscriptionTier.FREE: {
                "messages_per_day": 10,
                "api_calls_per_month": 0,
                "storage_mb": 50,
                "team_members": 1
            },
            SubscriptionTier.PREMIUM: {
                "messages_per_day": 200,
                "api_calls_per_month": 1000,
                "storage_mb": 1000,
                "team_members": 5
            },
            SubscriptionTier.ENTERPRISE: {
                "messages_per_day": -1,
                "api_calls_per_month": -1,
                "storage_mb": -1,
                "team_members": -1
            }
        }.get(tier, {})
    
    def _calculate_period_end(self, start_date: datetime, billing_cycle: BillingCycle) -> datetime:
        """
        Calculate period end date based on billing cycle
        """
        if billing_cycle == BillingCycle.DAILY:
            return start_date + timedelta(days=1)
        elif billing_cycle == BillingCycle.WEEKLY:
            return start_date + timedelta(weeks=1)
        elif billing_cycle == BillingCycle.MONTHLY:
            return start_date + timedelta(days=30)
        elif billing_cycle == BillingCycle.QUARTERLY:
            return start_date + timedelta(days=90)
        elif billing_cycle == BillingCycle.ANNUALLY:
            return start_date + timedelta(days=365)
        else:
            return start_date + timedelta(days=30)  # default to monthly
    
    def _is_upgrade(self, current_tier: SubscriptionTier, new_tier: SubscriptionTier) -> bool:
        """
        Check if new tier is an upgrade
        """
        tier_hierarchy = [
            SubscriptionTier.FREE,
            SubscriptionTier.STUDENT,
            SubscriptionTier.PREMIUM,
            SubscriptionTier.FAMILY,
            SubscriptionTier.ENTERPRISE,
            SubscriptionTier.CORPORATE,
            SubscriptionTier.LIFETIME
        ]
        
        try:
            current_index = tier_hierarchy.index(current_tier)
            new_index = tier_hierarchy.index(new_tier)
            return new_index > current_index
        except ValueError:
            return False
    
    def _is_downgrade(self, current_tier: SubscriptionTier, new_tier: SubscriptionTier) -> bool:
        """
        Check if new tier is a downgrade
        """
        return self._is_upgrade(new_tier, current_tier)
    
    async def _apply_discount(self, discount_code: str, amount: float) -> Optional[float]:
        """
        Apply discount code to amount
        """
        # This would integrate with discount management
        # For now, return None (no discount)
        return None