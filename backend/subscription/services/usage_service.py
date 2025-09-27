"""
Usage Tracking and Quota Management Service for DharmaMind
=========================================================

Comprehensive usage tracking service for monitoring and enforcing
subscription limits, analytics, and billing usage.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from ..models.subscription_models import (
    Usage, Subscription, SubscriptionTier, SubscriptionStatus
)

logger = logging.getLogger(__name__)

class UsageService:
    """
    Usage tracking and quota management service
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
        # Feature usage types
        self.usage_features = {
            'messages': 'Chat messages sent',
            'api_calls': 'API requests made',
            'storage': 'Storage used (MB)',
            'minutes': 'Session minutes',
            'exports': 'Chat exports',
            'custom_prompts': 'Custom prompts created',
            'team_members': 'Team members added',
            'meditation_sessions': 'Meditation sessions',
            'scripture_searches': 'Scripture searches',
            'dharma_insights': 'Dharma insights generated'
        }
    
    async def initialize_usage_tracking(self, subscription_id: str):
        """
        Initialize usage tracking for new subscription
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            # Create initial usage records for current period
            now = datetime.utcnow()
            period_start = subscription.current_period_start or now
            period_end = subscription.current_period_end or (now + timedelta(days=30))
            
            for feature in self.usage_features.keys():
                usage = Usage(
                    subscription_id=subscription_id,
                    user_id=subscription.user_id,
                    feature=feature,
                    usage_count=0,
                    usage_value=0.0,
                    period_start=period_start,
                    period_end=period_end
                )
                self.db.add(usage)
            
            self.db.commit()
            logger.info(f"Initialized usage tracking for subscription {subscription_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize usage tracking: {str(e)}")
            self.db.rollback()
            raise
    
    async def track_usage(
        self,
        user_id: str,
        feature: str,
        count: int = 1,
        value: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Track usage for a specific feature
        """
        try:
            # Get active subscription
            subscription = self.db.query(Subscription).filter(
                and_(
                    Subscription.user_id == user_id,
                    Subscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING
                    ])
                )
            ).first()
            
            if not subscription:
                # Track for free tier
                await self._track_free_tier_usage(user_id, feature, count, value)
                return True
            
            # Get or create usage record for current period
            usage_record = await self._get_current_usage_record(
                subscription.id, feature
            )
            
            # Update usage
            usage_record.usage_count += count
            usage_record.usage_value += value
            usage_record.recorded_at = datetime.utcnow()
            
            if metadata:
                existing_details = usage_record.details or {}
                existing_details.update(metadata)
                usage_record.details = existing_details
            
            self.db.commit()
            
            # Check quotas
            await self._check_usage_limits(subscription, feature, usage_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Usage tracking failed: {str(e)}")
            self.db.rollback()
            return False
    
    async def check_usage_limit(
        self,
        user_id: str,
        feature: str,
        requested_count: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user can perform action within usage limits
        """
        try:
            # Get active subscription
            subscription = self.db.query(Subscription).filter(
                and_(
                    Subscription.user_id == user_id,
                    Subscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING
                    ])
                )
            ).first()
            
            # Get tier limits
            if subscription:
                limits = subscription.usage_limits or self._get_tier_limits(subscription.tier)
            else:
                limits = self._get_tier_limits(SubscriptionTier.FREE)
            
            feature_limit = limits.get(feature, 0)
            
            # Unlimited usage
            if feature_limit == -1:
                return True, {
                    'allowed': True,
                    'limit': -1,
                    'current_usage': 0,
                    'remaining': -1
                }
            
            # Get current usage
            if subscription:
                usage_record = await self._get_current_usage_record(
                    subscription.id, feature
                )
                current_usage = usage_record.usage_count
            else:
                current_usage = await self._get_free_tier_usage(user_id, feature)
            
            # Check if request would exceed limit
            would_exceed = (current_usage + requested_count) > feature_limit
            remaining = max(0, feature_limit - current_usage)
            
            return not would_exceed, {
                'allowed': not would_exceed,
                'limit': feature_limit,
                'current_usage': current_usage,
                'remaining': remaining,
                'requested': requested_count
            }
            
        except Exception as e:
            logger.error(f"Usage limit check failed: {str(e)}")
            return False, {'error': str(e)}
    
    def get_usage_summary(
        self,
        user_id: str,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get comprehensive usage summary for user
        """
        try:
            # Get active subscription
            subscription = self.db.query(Subscription).filter(
                and_(
                    Subscription.user_id == user_id,
                    Subscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING
                    ])
                )
            ).first()
            
            if not subscription:
                return self._get_free_tier_summary(user_id, period_days)
            
            # Get usage records for period
            period_start = datetime.utcnow() - timedelta(days=period_days)
            
            usage_records = self.db.query(Usage).filter(
                and_(
                    Usage.subscription_id == subscription.id,
                    Usage.recorded_at >= period_start
                )
            ).all()
            
            # Aggregate usage by feature
            usage_by_feature = defaultdict(lambda: {'count': 0, 'value': 0.0})
            
            for record in usage_records:
                usage_by_feature[record.feature]['count'] += record.usage_count
                usage_by_feature[record.feature]['value'] += record.usage_value
            
            # Get limits
            limits = subscription.usage_limits or self._get_tier_limits(subscription.tier)
            
            # Format summary
            summary = {
                'subscription_id': subscription.id,
                'tier': subscription.tier.value,
                'period_days': period_days,
                'period_start': period_start.isoformat(),
                'period_end': datetime.utcnow().isoformat(),
                'features': {}
            }
            
            for feature, usage_data in usage_by_feature.items():
                limit = limits.get(feature, 0)
                remaining = limit - usage_data['count'] if limit != -1 else -1
                
                summary['features'][feature] = {
                    'usage_count': usage_data['count'],
                    'usage_value': usage_data['value'],
                    'limit': limit,
                    'remaining': remaining,
                    'percentage_used': (usage_data['count'] / limit * 100) if limit > 0 else 0,
                    'is_unlimited': limit == -1
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get usage summary: {str(e)}")
            return {'error': str(e)}
    
    def get_usage_analytics(
        self,
        subscription_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed usage analytics for subscription
        """
        try:
            period_start = datetime.utcnow() - timedelta(days=days)
            
            # Get usage records
            usage_records = self.db.query(Usage).filter(
                and_(
                    Usage.subscription_id == subscription_id,
                    Usage.recorded_at >= period_start
                )
            ).all()
            
            # Daily usage breakdown
            daily_usage = defaultdict(lambda: defaultdict(int))
            
            for record in usage_records:
                day = record.recorded_at.date()
                daily_usage[str(day)][record.feature] += record.usage_count
            
            # Feature popularity
            feature_totals = defaultdict(int)
            for record in usage_records:
                feature_totals[record.feature] += record.usage_count
            
            # Usage trends
            trends = self._calculate_usage_trends(usage_records, days)
            
            return {
                'subscription_id': subscription_id,
                'period_days': days,
                'daily_usage': dict(daily_usage),
                'feature_totals': dict(feature_totals),
                'trends': trends,
                'total_records': len(usage_records)
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage analytics: {str(e)}")
            return {'error': str(e)}
    
    async def reset_usage_period(self, subscription_id: str):
        """
        Reset usage tracking for new billing period
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            # Archive current period usage
            current_usage = self.db.query(Usage).filter(
                Usage.subscription_id == subscription_id
            ).all()
            
            for usage in current_usage:
                usage.period_end = subscription.current_period_end
            
            # Create new usage records for next period
            await self.initialize_usage_tracking(subscription_id)
            
            self.db.commit()
            logger.info(f"Reset usage period for subscription {subscription_id}")
            
        except Exception as e:
            logger.error(f"Failed to reset usage period: {str(e)}")
            self.db.rollback()
            raise
    
    def get_top_usage_features(
        self,
        subscription_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most used features for subscription
        """
        try:
            # Get usage aggregation
            results = self.db.query(
                Usage.feature,
                func.sum(Usage.usage_count).label('total_count'),
                func.sum(Usage.usage_value).label('total_value'),
                func.count(Usage.id).label('usage_sessions')
            ).filter(
                Usage.subscription_id == subscription_id
            ).group_by(
                Usage.feature
            ).order_by(
                desc('total_count')
            ).limit(limit).all()
            
            features = []
            for result in results:
                features.append({
                    'feature': result.feature,
                    'feature_name': self.usage_features.get(result.feature, result.feature),
                    'total_count': result.total_count,
                    'total_value': result.total_value,
                    'usage_sessions': result.usage_sessions
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to get top usage features: {str(e)}")
            return []
    
    async def _get_current_usage_record(
        self,
        subscription_id: str,
        feature: str
    ) -> Usage:
        """
        Get or create usage record for current period
        """
        subscription = self.db.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()
        
        if not subscription:
            raise ValueError("Subscription not found")
        
        # Look for existing usage record in current period
        usage_record = self.db.query(Usage).filter(
            and_(
                Usage.subscription_id == subscription_id,
                Usage.feature == feature,
                Usage.period_start == subscription.current_period_start,
                Usage.period_end == subscription.current_period_end
            )
        ).first()
        
        if not usage_record:
            # Create new usage record
            usage_record = Usage(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                feature=feature,
                usage_count=0,
                usage_value=0.0,
                period_start=subscription.current_period_start,
                period_end=subscription.current_period_end
            )
            self.db.add(usage_record)
            self.db.flush()  # Get ID
        
        return usage_record
    
    async def _track_free_tier_usage(
        self,
        user_id: str,
        feature: str,
        count: int,
        value: float
    ):
        """
        Track usage for free tier users (stored differently)
        """
        # For free tier, we can use a simpler approach
        # or create a temporary subscription record
        pass
    
    async def _get_free_tier_usage(self, user_id: str, feature: str) -> int:
        """
        Get current usage for free tier user
        """
        # Implementation for free tier usage tracking
        return 0
    
    def _get_free_tier_summary(self, user_id: str, period_days: int) -> Dict[str, Any]:
        """
        Get usage summary for free tier user
        """
        return {
            'tier': 'free',
            'period_days': period_days,
            'features': {
                'messages': {
                    'usage_count': 0,
                    'limit': 10,
                    'remaining': 10,
                    'percentage_used': 0,
                    'is_unlimited': False
                }
            }
        }
    
    def _get_tier_limits(self, tier: SubscriptionTier) -> Dict[str, int]:
        """
        Get usage limits for subscription tier
        """
        limits = {
            SubscriptionTier.FREE: {
                'messages': 10,
                'api_calls': 0,
                'storage': 50,
                'minutes': 30,
                'exports': 0,
                'custom_prompts': 0,
                'team_members': 1,
                'meditation_sessions': 3,
                'scripture_searches': 5,
                'dharma_insights': 3
            },
            SubscriptionTier.PREMIUM: {
                'messages': 200,
                'api_calls': 1000,
                'storage': 1000,
                'minutes': 600,
                'exports': 10,
                'custom_prompts': 50,
                'team_members': 5,
                'meditation_sessions': 20,
                'scripture_searches': 100,
                'dharma_insights': 50
            },
            SubscriptionTier.ENTERPRISE: {
                'messages': -1,  # unlimited
                'api_calls': -1,
                'storage': -1,
                'minutes': -1,
                'exports': -1,
                'custom_prompts': -1,
                'team_members': -1,
                'meditation_sessions': -1,
                'scripture_searches': -1,
                'dharma_insights': -1
            }
        }
        
        return limits.get(tier, limits[SubscriptionTier.FREE])
    
    async def _check_usage_limits(
        self,
        subscription: Subscription,
        feature: str,
        usage_record: Usage
    ):
        """
        Check and enforce usage limits
        """
        limits = subscription.usage_limits or self._get_tier_limits(subscription.tier)
        feature_limit = limits.get(feature, 0)
        
        if feature_limit > 0 and usage_record.usage_count > feature_limit:
            logger.warning(
                f"Usage limit exceeded for subscription {subscription.id}, "
                f"feature {feature}: {usage_record.usage_count}/{feature_limit}"
            )
            
            # Could trigger notifications, suspend features, etc.
            await self._handle_limit_exceeded(subscription, feature, usage_record)
    
    async def _handle_limit_exceeded(
        self,
        subscription: Subscription,
        feature: str,
        usage_record: Usage
    ):
        """
        Handle usage limit exceeded scenarios
        """
        # Implementation for handling exceeded limits
        # Could include:
        # - Sending notifications
        # - Temporarily disabling features
        # - Offering upgrade prompts
        # - Logging for analytics
        pass
    
    def _calculate_usage_trends(
        self,
        usage_records: List[Usage],
        period_days: int
    ) -> Dict[str, Any]:
        """
        Calculate usage trends and patterns
        """
        if not usage_records:
            return {'trends': 'no_data'}
        
        # Group by feature and date
        daily_totals = defaultdict(lambda: defaultdict(int))
        
        for record in usage_records:
            day = record.recorded_at.date()
            daily_totals[record.feature][day] += record.usage_count
        
        # Calculate trends for each feature
        trends = {}
        
        for feature, daily_data in daily_totals.items():
            if len(daily_data) < 2:
                trends[feature] = 'insufficient_data'
                continue
            
            # Simple trend calculation
            dates = sorted(daily_data.keys())
            first_half = dates[:len(dates)//2]
            second_half = dates[len(dates)//2:]
            
            first_avg = sum(daily_data[d] for d in first_half) / len(first_half)
            second_avg = sum(daily_data[d] for d in second_half) / len(second_half)
            
            if second_avg > first_avg * 1.1:
                trends[feature] = 'increasing'
            elif second_avg < first_avg * 0.9:
                trends[feature] = 'decreasing'
            else:
                trends[feature] = 'stable'
        
        return trends