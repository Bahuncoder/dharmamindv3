"""
Utility Functions for DharmaMind Subscription System
===================================================

Common utilities for subscription management, pricing calculations,
validation, and helper functions.
"""

import re
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum

from ..models.subscription_models import SubscriptionTier, BillingCycle, CurrencyCode

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class PricingCalculator:
    """
    Pricing calculation utilities
    """
    
    # Base pricing in USD
    BASE_PRICING = {
        SubscriptionTier.FREE: {
            BillingCycle.MONTHLY: 0.0,
            BillingCycle.ANNUALLY: 0.0,
            BillingCycle.LIFETIME: 0.0
        },
        SubscriptionTier.PREMIUM: {
            BillingCycle.MONTHLY: 19.99,
            BillingCycle.ANNUALLY: 199.99,
            BillingCycle.LIFETIME: 999.99
        },
        SubscriptionTier.ENTERPRISE: {
            BillingCycle.MONTHLY: 99.99,
            BillingCycle.ANNUALLY: 999.99,
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
        }
    }
    
    # Currency exchange rates (would be updated from external API)
    EXCHANGE_RATES = {
        CurrencyCode.USD: 1.0,
        CurrencyCode.EUR: 0.85,
        CurrencyCode.GBP: 0.75,
        CurrencyCode.INR: 83.0,
        CurrencyCode.CAD: 1.35,
        CurrencyCode.AUD: 1.50,
        CurrencyCode.SGD: 1.35,
        CurrencyCode.JPY: 150.0
    }
    
    @classmethod
    def get_price(
        cls, 
        tier: SubscriptionTier, 
        billing_cycle: BillingCycle,
        currency: CurrencyCode = CurrencyCode.USD
    ) -> float:
        """
        Get price for tier and billing cycle in specified currency
        """
        base_price = cls.BASE_PRICING[tier][billing_cycle]
        exchange_rate = cls.EXCHANGE_RATES[currency]
        
        return round(base_price * exchange_rate, 2)
    
    @classmethod
    def calculate_proration(
        cls,
        old_price: float,
        new_price: float,
        days_remaining: int,
        billing_cycle: BillingCycle
    ) -> float:
        """
        Calculate proration amount for subscription changes
        """
        if billing_cycle == BillingCycle.MONTHLY:
            daily_old = old_price / 30
            daily_new = new_price / 30
        elif billing_cycle == BillingCycle.ANNUALLY:
            daily_old = old_price / 365
            daily_new = new_price / 365
        else:
            return 0.0
        
        return round((daily_new - daily_old) * days_remaining, 2)
    
    @classmethod
    def calculate_discount(
        cls,
        base_amount: float,
        discount_percentage: float = 0,
        discount_amount: float = 0,
        max_discount: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate discount and final amount
        Returns: (discount_applied, final_amount)
        """
        discount_applied = 0.0
        
        if discount_percentage > 0:
            discount_applied = base_amount * (discount_percentage / 100)
        elif discount_amount > 0:
            discount_applied = min(discount_amount, base_amount)
        
        if max_discount and discount_applied > max_discount:
            discount_applied = max_discount
        
        final_amount = max(0.0, base_amount - discount_applied)
        
        return round(discount_applied, 2), round(final_amount, 2)
    
    @classmethod
    def calculate_tax(
        cls,
        amount: float,
        tax_rate: float,
        tax_inclusive: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate tax amount and total
        Returns: (tax_amount, total_amount)
        """
        if tax_inclusive:
            # Tax is included in the amount
            tax_amount = amount * (tax_rate / (100 + tax_rate))
            total_amount = amount
        else:
            # Tax is added to the amount
            tax_amount = amount * (tax_rate / 100)
            total_amount = amount + tax_amount
        
        return round(tax_amount, 2), round(total_amount, 2)

class SubscriptionValidator:
    """
    Validation utilities for subscription data
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_tier_upgrade(current_tier: SubscriptionTier, new_tier: SubscriptionTier) -> bool:
        """Validate if tier change is an upgrade"""
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
    
    @staticmethod
    def validate_billing_cycle(tier: SubscriptionTier, billing_cycle: BillingCycle) -> bool:
        """Validate if billing cycle is valid for tier"""
        # Free tier only supports monthly (free)
        if tier == SubscriptionTier.FREE:
            return billing_cycle == BillingCycle.MONTHLY
        
        # Lifetime tier only supports lifetime billing
        if tier == SubscriptionTier.LIFETIME:
            return billing_cycle == BillingCycle.LIFETIME
        
        # Other tiers support monthly and annual
        return billing_cycle in [BillingCycle.MONTHLY, BillingCycle.ANNUALLY]
    
    @staticmethod
    def validate_payment_amount(amount: float, min_amount: float = 0.50) -> bool:
        """Validate payment amount"""
        return amount >= min_amount and amount <= 999999.99
    
    @staticmethod
    def validate_trial_period(days: int) -> bool:
        """Validate trial period duration"""
        return 0 <= days <= 90  # Max 90 days trial

class FeatureManager:
    """
    Feature access and limit management
    """
    
    TIER_FEATURES = {
        SubscriptionTier.FREE: {
            'daily_messages': 10,
            'ai_models': ['basic'],
            'chat_history_days': 7,
            'export_chats': False,
            'priority_support': False,
            'advanced_analytics': False,
            'custom_prompts': False,
            'api_access': False,
            'dharma_insights': 'basic',
            'meditation_sessions': 3,
            'scripture_access': 'limited',
            'voice_interaction': False,
            'team_management': False
        },
        SubscriptionTier.PREMIUM: {
            'daily_messages': 200,
            'ai_models': ['basic', 'advanced'],
            'chat_history_days': 90,
            'export_chats': True,
            'priority_support': True,
            'advanced_analytics': True,
            'custom_prompts': True,
            'api_access': 'limited',
            'dharma_insights': 'advanced',
            'meditation_sessions': 20,
            'scripture_access': 'full',
            'voice_interaction': True,
            'team_management': False,
            'personalized_guidance': True
        },
        SubscriptionTier.ENTERPRISE: {
            'daily_messages': -1,  # unlimited
            'ai_models': ['basic', 'advanced', 'premium'],
            'chat_history_days': -1,  # unlimited
            'export_chats': True,
            'priority_support': True,
            'advanced_analytics': True,
            'custom_prompts': True,
            'api_access': 'full',
            'dharma_insights': 'premium',
            'meditation_sessions': -1,  # unlimited
            'scripture_access': 'complete',
            'voice_interaction': True,
            'team_management': True,
            'personalized_guidance': True,
            'white_labeling': True,
            'dedicated_support': True
        }
    }
    
    @classmethod
    def get_feature_value(cls, tier: SubscriptionTier, feature: str) -> Any:
        """Get feature value for tier"""
        return cls.TIER_FEATURES.get(tier, {}).get(feature)
    
    @classmethod
    def has_feature(cls, tier: SubscriptionTier, feature: str) -> bool:
        """Check if tier has feature enabled"""
        feature_value = cls.get_feature_value(tier, feature)
        
        if isinstance(feature_value, bool):
            return feature_value
        elif isinstance(feature_value, int):
            return feature_value != 0
        elif isinstance(feature_value, list):
            return len(feature_value) > 0
        elif isinstance(feature_value, str):
            return feature_value not in ['', 'none', 'disabled']
        
        return feature_value is not None
    
    @classmethod
    def get_usage_limit(cls, tier: SubscriptionTier, feature: str) -> int:
        """Get usage limit for feature"""
        feature_value = cls.get_feature_value(tier, feature)
        
        if isinstance(feature_value, int):
            return feature_value
        
        # Default limits for non-numeric features
        if feature_value is True:
            return -1  # unlimited
        elif feature_value is False:
            return 0
        
        return 0

class SecurityUtils:
    """
    Security utilities for subscription system
    """
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data with optional salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        return hashlib.sha256((data + salt).encode()).hexdigest()
    
    @staticmethod
    def mask_card_number(card_number: str) -> str:
        """Mask credit card number, showing only last 4 digits"""
        if len(card_number) < 4:
            return '*' * len(card_number)
        
        return '*' * (len(card_number) - 4) + card_number[-4:]
    
    @staticmethod
    def validate_webhook_signature(payload: str, signature: str, secret: str) -> bool:
        """Validate webhook signature"""
        expected_signature = hashlib.sha256(
            (payload + secret).encode()
        ).hexdigest()
        
        return secrets.compare_digest(signature, expected_signature)

class DateTimeUtils:
    """
    Date and time utilities
    """
    
    @staticmethod
    def add_billing_period(
        start_date: datetime,
        billing_cycle: BillingCycle
    ) -> datetime:
        """Add billing period to date"""
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
    
    @staticmethod
    def get_billing_period_days(billing_cycle: BillingCycle) -> int:
        """Get number of days in billing period"""
        period_days = {
            BillingCycle.DAILY: 1,
            BillingCycle.WEEKLY: 7,
            BillingCycle.MONTHLY: 30,
            BillingCycle.QUARTERLY: 90,
            BillingCycle.ANNUALLY: 365
        }
        
        return period_days.get(billing_cycle, 30)
    
    @staticmethod
    def is_date_in_period(
        check_date: datetime,
        period_start: datetime,
        period_end: datetime
    ) -> bool:
        """Check if date falls within period"""
        return period_start <= check_date <= period_end
    
    @staticmethod
    def format_datetime_for_display(dt: datetime, format_str: str = "%B %d, %Y at %I:%M %p") -> str:
        """Format datetime for user-friendly display"""
        return dt.strftime(format_str)

class NotificationTemplates:
    """
    Email and notification templates for subscription events
    """
    
    TEMPLATES = {
        'subscription_created': {
            'subject': 'Welcome to DharmaMind - Subscription Activated',
            'body': '''
            Dear {user_name},
            
            Your DharmaMind {tier} subscription has been successfully activated!
            
            Subscription Details:
            - Tier: {tier}
            - Billing Cycle: {billing_cycle}
            - Amount: {currency}{amount}
            - Next Billing Date: {next_billing_date}
            
            Start your spiritual journey today!
            
            Namaste,
            The DharmaMind Team
            '''
        },
        'subscription_upgraded': {
            'subject': 'DharmaMind Subscription Upgraded',
            'body': '''
            Dear {user_name},
            
            Your DharmaMind subscription has been upgraded to {new_tier}!
            
            You now have access to enhanced features including:
            {features_list}
            
            Thank you for your continued trust in DharmaMind.
            
            Namaste,
            The DharmaMind Team
            '''
        },
        'payment_failed': {
            'subject': 'Payment Failed - Action Required',
            'body': '''
            Dear {user_name},
            
            We were unable to process your payment for your DharmaMind subscription.
            
            Please update your payment method to continue enjoying our services.
            
            Login to your account: {login_url}
            
            If you have questions, please contact our support team.
            
            Namaste,
            The DharmaMind Team
            '''
        },
        'trial_ending': {
            'subject': 'Your DharmaMind Trial is Ending Soon',
            'body': '''
            Dear {user_name},
            
            Your DharmaMind trial period ends in {days_remaining} days.
            
            Continue your spiritual journey by upgrading to a paid subscription.
            
            Upgrade now: {upgrade_url}
            
            Namaste,
            The DharmaMind Team
            '''
        }
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> Dict[str, str]:
        """Get notification template"""
        return cls.TEMPLATES.get(template_name, {
            'subject': 'DharmaMind Notification',
            'body': 'Thank you for using DharmaMind.'
        })
    
    @classmethod
    def format_template(
        cls,
        template_name: str,
        **kwargs
    ) -> Dict[str, str]:
        """Format template with provided variables"""
        template = cls.get_template(template_name)
        
        formatted_template = {}
        for key, value in template.items():
            formatted_template[key] = value.format(**kwargs)
        
        return formatted_template

# Convenience functions

def calculate_monthly_recurring_revenue(subscriptions: List[Dict]) -> float:
    """Calculate MRR from subscription list"""
    mrr = 0.0
    
    for sub in subscriptions:
        if sub['status'] != 'active':
            continue
        
        amount = sub['amount']
        billing_cycle = sub['billing_cycle']
        
        if billing_cycle == 'monthly':
            mrr += amount
        elif billing_cycle == 'annually':
            mrr += amount / 12
        elif billing_cycle == 'quarterly':
            mrr += amount / 3
    
    return round(mrr, 2)

def get_subscription_health_score(
    active_subs: int,
    churn_rate: float,
    growth_rate: float,
    payment_success_rate: float
) -> Dict[str, Any]:
    """Calculate overall subscription health score"""
    
    # Weight factors
    weights = {
        'growth': 0.3,
        'retention': 0.3,
        'payment': 0.2,
        'scale': 0.2
    }
    
    # Normalize metrics to 0-100 scale
    growth_score = min(100, max(0, growth_rate * 10))
    retention_score = min(100, max(0, (100 - churn_rate) * 1.2))
    payment_score = payment_success_rate
    scale_score = min(100, active_subs / 10)  # 1000 subs = 100 score
    
    # Calculate weighted score
    health_score = (
        growth_score * weights['growth'] +
        retention_score * weights['retention'] +
        payment_score * weights['payment'] +
        scale_score * weights['scale']
    )
    
    # Determine health status
    if health_score >= 80:
        status = 'excellent'
    elif health_score >= 60:
        status = 'good'
    elif health_score >= 40:
        status = 'fair'
    else:
        status = 'poor'
    
    return {
        'health_score': round(health_score, 1),
        'status': status,
        'component_scores': {
            'growth': round(growth_score, 1),
            'retention': round(retention_score, 1),
            'payment': round(payment_score, 1),
            'scale': round(scale_score, 1)
        }
    }