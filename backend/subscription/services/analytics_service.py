"""
Subscription Analytics Service for DharmaMind
===========================================

Advanced analytics service providing insights into subscription
metrics, revenue analytics, churn analysis, and business intelligence.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, extract

from ..models.subscription_models import (
    Subscription, SubscriptionTier, SubscriptionStatus,
    Payment, PaymentStatus, SubscriptionChange, SubscriptionMetrics
)

logger = logging.getLogger(__name__)

class SubscriptionAnalyticsService:
    """
    Comprehensive subscription analytics and reporting service
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_subscription_overview(self) -> Dict[str, Any]:
        """
        Get high-level subscription overview metrics
        """
        try:
            # Active subscriptions by tier
            active_by_tier = self.db.query(
                Subscription.tier,
                func.count(Subscription.id).label('count')
            ).filter(
                Subscription.status == SubscriptionStatus.ACTIVE
            ).group_by(Subscription.tier).all()
            
            # Total revenue (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_revenue = self.db.query(
                func.sum(Payment.amount)
            ).filter(
                and_(
                    Payment.status == PaymentStatus.SUCCEEDED,
                    Payment.processed_at >= thirty_days_ago
                )
            ).scalar() or 0
            
            # Churn rate (last 30 days)
            cancellations = self.db.query(func.count(Subscription.id)).filter(
                and_(
                    Subscription.status == SubscriptionStatus.CANCELLED,
                    Subscription.cancelled_at >= thirty_days_ago
                )
            ).scalar() or 0
            
            total_active = self.db.query(func.count(Subscription.id)).filter(
                Subscription.status == SubscriptionStatus.ACTIVE
            ).scalar() or 0
            
            churn_rate = (cancellations / total_active * 100) if total_active > 0 else 0
            
            # New subscriptions (last 30 days)
            new_subscriptions = self.db.query(func.count(Subscription.id)).filter(
                Subscription.created_at >= thirty_days_ago
            ).scalar() or 0
            
            return {
                'active_subscriptions': {
                    'total': total_active,
                    'by_tier': {tier.value: count for tier, count in active_by_tier}
                },
                'revenue': {
                    'last_30_days': float(recent_revenue),
                    'currency': 'USD'
                },
                'churn_rate': round(churn_rate, 2),
                'new_subscriptions_30d': new_subscriptions,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Subscription overview failed: {str(e)}")
            return {'error': str(e)}
    
    def get_revenue_analytics(self, days: int = 90) -> Dict[str, Any]:
        """
        Get detailed revenue analytics
        """
        try:
            period_start = datetime.utcnow() - timedelta(days=days)
            
            # Daily revenue
            daily_revenue = self.db.query(
                func.date(Payment.processed_at).label('date'),
                func.sum(Payment.amount).label('revenue'),
                func.count(Payment.id).label('payment_count')
            ).filter(
                and_(
                    Payment.status == PaymentStatus.SUCCEEDED,
                    Payment.processed_at >= period_start
                )
            ).group_by(func.date(Payment.processed_at)).all()
            
            # Revenue by tier
            revenue_by_tier = self.db.query(
                Subscription.tier,
                func.sum(Payment.amount).label('revenue')
            ).join(
                Payment, Subscription.id == Payment.subscription_id
            ).filter(
                and_(
                    Payment.status == PaymentStatus.SUCCEEDED,
                    Payment.processed_at >= period_start
                )
            ).group_by(Subscription.tier).all()
            
            # Calculate MRR (Monthly Recurring Revenue)
            mrr = self._calculate_mrr()
            
            # Calculate ARR (Annual Recurring Revenue)
            arr = mrr * 12
            
            # Average revenue per user
            total_revenue = sum(day.revenue for day in daily_revenue)
            unique_users = self.db.query(func.count(func.distinct(Payment.user_id))).filter(
                and_(
                    Payment.status == PaymentStatus.SUCCEEDED,
                    Payment.processed_at >= period_start
                )
            ).scalar() or 0
            
            arpu = total_revenue / unique_users if unique_users > 0 else 0
            
            return {
                'period_days': days,
                'total_revenue': float(total_revenue),
                'mrr': float(mrr),
                'arr': float(arr),
                'arpu': round(float(arpu), 2),
                'daily_revenue': [
                    {
                        'date': day.date.isoformat(),
                        'revenue': float(day.revenue),
                        'payment_count': day.payment_count
                    } for day in daily_revenue
                ],
                'revenue_by_tier': {
                    tier.value: float(revenue) for tier, revenue in revenue_by_tier
                }
            }
            
        except Exception as e:
            logger.error(f"Revenue analytics failed: {str(e)}")
            return {'error': str(e)}
    
    def get_churn_analysis(self, months: int = 6) -> Dict[str, Any]:
        """
        Get detailed churn analysis
        """
        try:
            period_start = datetime.utcnow() - timedelta(days=months * 30)
            
            # Monthly churn rates
            monthly_churn = []
            
            for i in range(months):
                month_start = datetime.utcnow() - timedelta(days=(i + 1) * 30)
                month_end = datetime.utcnow() - timedelta(days=i * 30)
                
                # Active at start of month
                active_start = self.db.query(func.count(Subscription.id)).filter(
                    and_(
                        Subscription.created_at <= month_start,
                        or_(
                            Subscription.cancelled_at.is_(None),
                            Subscription.cancelled_at > month_start
                        )
                    )
                ).scalar() or 0
                
                # Churned during month
                churned = self.db.query(func.count(Subscription.id)).filter(
                    and_(
                        Subscription.cancelled_at >= month_start,
                        Subscription.cancelled_at < month_end
                    )
                ).scalar() or 0
                
                churn_rate = (churned / active_start * 100) if active_start > 0 else 0
                
                monthly_churn.append({
                    'month': month_start.strftime('%Y-%m'),
                    'active_start': active_start,
                    'churned': churned,
                    'churn_rate': round(churn_rate, 2)
                })
            
            # Churn reasons analysis
            churn_reasons = self.db.query(
                SubscriptionChange.reason,
                func.count(SubscriptionChange.id).label('count')
            ).filter(
                and_(
                    SubscriptionChange.change_type == 'cancel',
                    SubscriptionChange.effective_date >= period_start
                )
            ).group_by(SubscriptionChange.reason).all()
            
            # Cohort retention analysis
            cohort_retention = self._calculate_cohort_retention(months)
            
            return {
                'period_months': months,
                'monthly_churn': monthly_churn,
                'churn_reasons': {
                    reason: count for reason, count in churn_reasons
                },
                'cohort_retention': cohort_retention,
                'average_churn_rate': round(
                    sum(month['churn_rate'] for month in monthly_churn) / len(monthly_churn), 2
                ) if monthly_churn else 0
            }
            
        except Exception as e:
            logger.error(f"Churn analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_customer_lifetime_value(self) -> Dict[str, Any]:
        """
        Calculate Customer Lifetime Value (CLV) analytics
        """
        try:
            # Average subscription duration
            completed_subscriptions = self.db.query(
                Subscription.created_at,
                Subscription.cancelled_at
            ).filter(
                Subscription.status == SubscriptionStatus.CANCELLED
            ).all()
            
            if not completed_subscriptions:
                return {'error': 'Insufficient data for CLV calculation'}
            
            durations = []
            for sub in completed_subscriptions:
                if sub.cancelled_at:
                    duration = (sub.cancelled_at - sub.created_at).days
                    durations.append(duration)
            
            avg_duration_days = sum(durations) / len(durations)
            avg_duration_months = avg_duration_days / 30
            
            # Average monthly revenue per user
            mrr = self._calculate_mrr()
            total_active = self.db.query(func.count(Subscription.id)).filter(
                Subscription.status == SubscriptionStatus.ACTIVE
            ).scalar() or 1
            
            avg_monthly_revenue = mrr / total_active
            
            # Customer Lifetime Value
            clv = avg_monthly_revenue * avg_duration_months
            
            # CLV by tier
            clv_by_tier = {}
            for tier in SubscriptionTier:
                tier_revenue = self.db.query(func.avg(Subscription.current_price)).filter(
                    and_(
                        Subscription.tier == tier,
                        Subscription.status == SubscriptionStatus.ACTIVE
                    )
                ).scalar() or 0
                
                clv_by_tier[tier.value] = round(float(tier_revenue) * avg_duration_months, 2)
            
            return {
                'average_lifetime_months': round(avg_duration_months, 1),
                'average_monthly_revenue': round(float(avg_monthly_revenue), 2),
                'customer_lifetime_value': round(float(clv), 2),
                'clv_by_tier': clv_by_tier,
                'sample_size': len(completed_subscriptions)
            }
            
        except Exception as e:
            logger.error(f"CLV calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def get_subscription_trends(self, days: int = 90) -> Dict[str, Any]:
        """
        Get subscription trends and patterns
        """
        try:
            period_start = datetime.utcnow() - timedelta(days=days)
            
            # Daily new subscriptions
            daily_new = self.db.query(
                func.date(Subscription.created_at).label('date'),
                func.count(Subscription.id).label('count')
            ).filter(
                Subscription.created_at >= period_start
            ).group_by(func.date(Subscription.created_at)).all()
            
            # Tier preference trends
            tier_trends = self.db.query(
                Subscription.tier,
                func.date(Subscription.created_at).label('date'),
                func.count(Subscription.id).label('count')
            ).filter(
                Subscription.created_at >= period_start
            ).group_by(Subscription.tier, func.date(Subscription.created_at)).all()
            
            # Upgrade/downgrade patterns
            upgrade_trends = self.db.query(
                SubscriptionChange.change_type,
                func.date(SubscriptionChange.effective_date).label('date'),
                func.count(SubscriptionChange.id).label('count')
            ).filter(
                and_(
                    SubscriptionChange.effective_date >= period_start,
                    SubscriptionChange.change_type.in_(['upgrade', 'downgrade'])
                )
            ).group_by(
                SubscriptionChange.change_type,
                func.date(SubscriptionChange.effective_date)
            ).all()
            
            return {
                'period_days': days,
                'daily_new_subscriptions': [
                    {
                        'date': day.date.isoformat(),
                        'count': day.count
                    } for day in daily_new
                ],
                'tier_trends': self._group_tier_trends(tier_trends),
                'upgrade_downgrade_trends': self._group_change_trends(upgrade_trends)
            }
            
        except Exception as e:
            logger.error(f"Subscription trends failed: {str(e)}")
            return {'error': str(e)}
    
    def generate_business_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive business analytics report
        """
        try:
            # Get all key metrics
            overview = self.get_subscription_overview()
            revenue = self.get_revenue_analytics(30)
            churn = self.get_churn_analysis(3)
            clv = self.get_customer_lifetime_value()
            
            # Calculate growth metrics
            growth_metrics = self._calculate_growth_metrics()
            
            # Market penetration by tier
            tier_penetration = self._calculate_tier_penetration()
            
            return {
                'report_date': datetime.utcnow().isoformat(),
                'overview': overview,
                'revenue_metrics': revenue,
                'churn_analysis': churn,
                'customer_lifetime_value': clv,
                'growth_metrics': growth_metrics,
                'tier_penetration': tier_penetration,
                'recommendations': self._generate_recommendations(
                    overview, revenue, churn, clv
                )
            }
            
        except Exception as e:
            logger.error(f"Business report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_mrr(self) -> float:
        """
        Calculate Monthly Recurring Revenue
        """
        # Get all active subscriptions
        monthly_subscriptions = self.db.query(func.sum(Subscription.current_price)).filter(
            and_(
                Subscription.status == SubscriptionStatus.ACTIVE,
                Subscription.billing_cycle == BillingCycle.MONTHLY
            )
        ).scalar() or 0
        
        # Convert annual to monthly
        annual_subscriptions = self.db.query(func.sum(Subscription.current_price)).filter(
            and_(
                Subscription.status == SubscriptionStatus.ACTIVE,
                Subscription.billing_cycle == BillingCycle.ANNUALLY
            )
        ).scalar() or 0
        
        annual_mrr = annual_subscriptions / 12
        
        return float(monthly_subscriptions + annual_mrr)
    
    def _calculate_cohort_retention(self, months: int) -> Dict[str, Any]:
        """
        Calculate cohort retention analysis
        """
        # This would implement detailed cohort analysis
        # For now, return placeholder data
        return {
            'cohorts': [],
            'retention_rates': [],
            'note': 'Cohort analysis requires more historical data'
        }
    
    def _group_tier_trends(self, tier_trends) -> Dict[str, List]:
        """
        Group tier trends by tier for better visualization
        """
        grouped = defaultdict(list)
        
        for trend in tier_trends:
            grouped[trend.tier.value].append({
                'date': trend.date.isoformat(),
                'count': trend.count
            })
        
        return dict(grouped)
    
    def _group_change_trends(self, change_trends) -> Dict[str, List]:
        """
        Group subscription change trends
        """
        grouped = defaultdict(list)
        
        for trend in change_trends:
            grouped[trend.change_type].append({
                'date': trend.date.isoformat(),
                'count': trend.count
            })
        
        return dict(grouped)
    
    def _calculate_growth_metrics(self) -> Dict[str, Any]:
        """
        Calculate growth metrics
        """
        # Month over month growth
        this_month_start = datetime.utcnow().replace(day=1)
        last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)
        
        this_month_new = self.db.query(func.count(Subscription.id)).filter(
            Subscription.created_at >= this_month_start
        ).scalar() or 0
        
        last_month_new = self.db.query(func.count(Subscription.id)).filter(
            and_(
                Subscription.created_at >= last_month_start,
                Subscription.created_at < this_month_start
            )
        ).scalar() or 1
        
        mom_growth = ((this_month_new - last_month_new) / last_month_new * 100) if last_month_new > 0 else 0
        
        return {
            'month_over_month_growth': round(mom_growth, 2),
            'this_month_new': this_month_new,
            'last_month_new': last_month_new
        }
    
    def _calculate_tier_penetration(self) -> Dict[str, float]:
        """
        Calculate market penetration by tier
        """
        total_active = self.db.query(func.count(Subscription.id)).filter(
            Subscription.status == SubscriptionStatus.ACTIVE
        ).scalar() or 1
        
        penetration = {}
        for tier in SubscriptionTier:
            tier_count = self.db.query(func.count(Subscription.id)).filter(
                and_(
                    Subscription.status == SubscriptionStatus.ACTIVE,
                    Subscription.tier == tier
                )
            ).scalar() or 0
            
            penetration[tier.value] = round((tier_count / total_active * 100), 2)
        
        return penetration
    
    def _generate_recommendations(
        self,
        overview: Dict,
        revenue: Dict,
        churn: Dict,
        clv: Dict
    ) -> List[str]:
        """
        Generate business recommendations based on analytics
        """
        recommendations = []
        
        # High churn rate
        avg_churn = churn.get('average_churn_rate', 0)
        if avg_churn > 10:
            recommendations.append(
                f"High churn rate ({avg_churn}%) detected. Consider improving onboarding and customer success programs."
            )
        
        # Low tier penetration
        tier_penetration = overview.get('active_subscriptions', {}).get('by_tier', {})
        premium_ratio = tier_penetration.get('premium', 0) / sum(tier_penetration.values()) if tier_penetration else 0
        
        if premium_ratio < 0.3:
            recommendations.append(
                "Low premium tier adoption. Consider highlighting premium features and offering trial upgrades."
            )
        
        # Revenue growth
        if revenue.get('mrr', 0) < 10000:
            recommendations.append(
                "Focus on customer acquisition and retention to scale MRR to $10k+ milestone."
            )
        
        return recommendations