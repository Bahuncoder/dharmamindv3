"""
Comprehensive Subscription & Payment Models for DharmaMind
=========================================================

Advanced data models for multi-tier subscription management, payment processing,
billing cycles, usage tracking, and analytics.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, backref
from sqlalchemy import ForeignKey
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
from decimal import Decimal

Base = declarative_base()

class SubscriptionTier(str, Enum):
    """Subscription tier levels with spiritual progression theme"""
    FREE = "free"              # Seeker - Basic spiritual guidance
    PREMIUM = "premium"        # Devotee - Enhanced spiritual features  
    ENTERPRISE = "enterprise"  # Guru - Complete spiritual platform
    LIFETIME = "lifetime"      # Enlightened - Permanent access
    FAMILY = "family"         # Ashram - Family sharing plan
    STUDENT = "student"       # Shishya - Educational discount
    CORPORATE = "corporate"   # Sangha - Organization plan

class SubscriptionStatus(str, Enum):
    """Subscription status states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAUSED = "paused"

class PaymentStatus(str, Enum):
    """Payment transaction status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    DISPUTED = "disputed"

class BillingCycle(str, Enum):
    """Billing frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    LIFETIME = "lifetime"

class PaymentMethod(str, Enum):
    """Supported payment methods"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    RAZORPAY = "razorpay"
    SQUARE = "square"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    UPI = "upi"
    WALLET = "wallet"

class CurrencyCode(str, Enum):
    """Supported currencies"""
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    INR = "inr"
    CAD = "cad"
    AUD = "aud"
    SGD = "sgd"
    JPY = "jpy"

class DiscountType(str, Enum):
    """Discount types"""
    PERCENTAGE = "percentage"
    FIXED_AMOUNT = "fixed_amount"
    TRIAL_EXTENSION = "trial_extension"
    UPGRADE_CREDIT = "upgrade_credit"

# Core Subscription Models

class Subscription(Base):
    """
    Core subscription model with comprehensive features
    """
    __tablename__ = 'subscriptions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Subscription Details
    tier = Column(SQLEnum(SubscriptionTier), nullable=False)
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.INACTIVE)
    billing_cycle = Column(SQLEnum(BillingCycle), default=BillingCycle.MONTHLY)
    
    # Pricing
    base_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)  # After discounts
    currency = Column(SQLEnum(CurrencyCode), default=CurrencyCode.USD)
    
    # Dates
    created_at = Column(DateTime, default=func.now())
    activated_at = Column(DateTime)
    trial_start = Column(DateTime)
    trial_end = Column(DateTime)
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancelled_at = Column(DateTime)
    ended_at = Column(DateTime)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Features & Limits
    features_enabled = Column(JSON, default=dict)  # Custom feature flags
    usage_limits = Column(JSON, default=dict)      # Usage quotas
    
    # Payment & Billing
    payment_method_id = Column(String, ForeignKey('payment_methods.id'))
    stripe_subscription_id = Column(String, unique=True)
    paypal_subscription_id = Column(String, unique=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    notes = Column(Text)
    
    # Relationships
    payments = relationship("Payment", back_populates="subscription")
    invoices = relationship("Invoice", back_populates="subscription")
    usage_records = relationship("Usage", back_populates="subscription")

    def is_active(self) -> bool:
        """Check if subscription is currently active"""
        return self.status == SubscriptionStatus.ACTIVE
    
    def is_trial(self) -> bool:
        """Check if subscription is in trial period"""
        if not self.trial_end:
            return False
        return datetime.utcnow() <= self.trial_end
    
    def days_until_renewal(self) -> int:
        """Days until next billing period"""
        if not self.current_period_end:
            return 0
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)
    
    def get_tier_features(self) -> Dict[str, Any]:
        """Get features available for current tier"""
        tier_features = {
            SubscriptionTier.FREE: {
                "daily_messages": 10,
                "ai_models": ["basic"],
                "chat_history": 7,  # days
                "export_chats": False,
                "priority_support": False,
                "advanced_analytics": False,
                "custom_prompts": False,
                "api_access": False,
                "dharma_insights": "basic",
                "meditation_sessions": 3,
                "scripture_access": "limited"
            },
            SubscriptionTier.PREMIUM: {
                "daily_messages": 200,
                "ai_models": ["basic", "advanced"],
                "chat_history": 90,  # days
                "export_chats": True,
                "priority_support": True,
                "advanced_analytics": True,
                "custom_prompts": True,
                "api_access": "limited",
                "dharma_insights": "advanced",
                "meditation_sessions": 20,
                "scripture_access": "full",
                "personalized_guidance": True,
                "voice_interaction": True
            },
            SubscriptionTier.ENTERPRISE: {
                "daily_messages": -1,  # unlimited
                "ai_models": ["basic", "advanced", "premium"],
                "chat_history": -1,  # unlimited
                "export_chats": True,
                "priority_support": True,
                "advanced_analytics": True,
                "custom_prompts": True,
                "api_access": "full",
                "dharma_insights": "premium",
                "meditation_sessions": -1,
                "scripture_access": "complete",
                "personalized_guidance": True,
                "voice_interaction": True,
                "team_management": True,
                "white_labeling": True,
                "dedicated_support": True
            }
        }
        
        base_features = tier_features.get(self.tier, tier_features[SubscriptionTier.FREE])
        # Override with custom features
        base_features.update(self.features_enabled or {})
        return base_features

class Payment(Base):
    """
    Payment transaction records
    """
    __tablename__ = 'payments'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String, ForeignKey('subscriptions.id'), nullable=False)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Payment Details
    amount = Column(Float, nullable=False)
    currency = Column(SQLEnum(CurrencyCode), default=CurrencyCode.USD)
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    payment_method = Column(SQLEnum(PaymentMethod), nullable=False)
    
    # External IDs
    stripe_payment_intent_id = Column(String, unique=True)
    stripe_charge_id = Column(String, unique=True)
    paypal_payment_id = Column(String, unique=True)
    razorpay_payment_id = Column(String, unique=True)
    
    # Transaction Info
    description = Column(Text)
    failure_reason = Column(Text)
    failure_code = Column(String)
    
    # Dates
    created_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime)
    failed_at = Column(DateTime)
    refunded_at = Column(DateTime)
    
    # Refund Info
    refunded_amount = Column(Float, default=0.0)
    refund_reason = Column(Text)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="payments")
    invoice = relationship("Invoice", back_populates="payment", uselist=False)

class Invoice(Base):
    """
    Invoice generation and management
    """
    __tablename__ = 'invoices'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String, ForeignKey('subscriptions.id'), nullable=False)
    payment_id = Column(String, ForeignKey('payments.id'))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Invoice Details
    invoice_number = Column(String, unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    tax_amount = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    currency = Column(SQLEnum(CurrencyCode), default=CurrencyCode.USD)
    
    # Dates
    issue_date = Column(DateTime, default=func.now())
    due_date = Column(DateTime, nullable=False)
    paid_date = Column(DateTime)
    
    # Status
    status = Column(String, default="draft")  # draft, sent, paid, overdue, cancelled
    
    # Billing Info
    billing_period_start = Column(DateTime)
    billing_period_end = Column(DateTime)
    
    # Content
    line_items = Column(JSON, default=list)  # Itemized billing
    notes = Column(Text)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="invoices")
    payment = relationship("Payment", back_populates="invoice")

class Usage(Base):
    """
    Usage tracking for quotas and analytics
    """
    __tablename__ = 'usage'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String, ForeignKey('subscriptions.id'), nullable=False)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Usage Metrics
    feature = Column(String, nullable=False)  # e.g., "messages", "api_calls", "minutes"
    usage_count = Column(Integer, default=0)
    usage_value = Column(Float, default=0.0)  # For measured usage like data transfer
    
    # Time Period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    recorded_at = Column(DateTime, default=func.now())
    
    # Metadata
    details = Column(JSON, default=dict)  # Additional usage context
    
    # Relationships
    subscription = relationship("Subscription", back_populates="usage_records")

class PaymentMethodRecord(Base):
    """
    Stored payment method information
    """
    __tablename__ = 'payment_methods'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Payment Method Details
    type = Column(SQLEnum(PaymentMethod), nullable=False)
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # External IDs
    stripe_payment_method_id = Column(String)
    paypal_agreement_id = Column(String)
    
    # Card Info (encrypted/tokenized)
    last_four = Column(String)
    brand = Column(String)  # visa, mastercard, etc.
    exp_month = Column(Integer)
    exp_year = Column(Integer)
    
    # Billing Address
    billing_address = Column(JSON, default=dict)
    
    # Dates
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Metadata
    metadata = Column(JSON, default=dict)

class Discount(Base):
    """
    Discounts and promotional codes
    """
    __tablename__ = 'discounts'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Discount Details
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Discount Value
    type = Column(SQLEnum(DiscountType), nullable=False)
    value = Column(Float, nullable=False)  # Percentage or fixed amount
    max_redemptions = Column(Integer)
    current_redemptions = Column(Integer, default=0)
    
    # Validity
    valid_from = Column(DateTime, default=func.now())
    valid_until = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Restrictions
    applicable_tiers = Column(JSON, default=list)  # Which tiers can use this
    first_time_only = Column(Boolean, default=False)
    
    # Dates
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Metadata
    metadata = Column(JSON, default=dict)

class SubscriptionChange(Base):
    """
    Track subscription modifications and upgrades/downgrades
    """
    __tablename__ = 'subscription_changes'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String, ForeignKey('subscriptions.id'), nullable=False)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Change Details
    change_type = Column(String, nullable=False)  # upgrade, downgrade, cancel, pause, resume
    from_tier = Column(SQLEnum(SubscriptionTier))
    to_tier = Column(SQLEnum(SubscriptionTier))
    
    # Pricing Impact
    old_price = Column(Float)
    new_price = Column(Float)
    proration_amount = Column(Float, default=0.0)
    
    # Dates
    effective_date = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    
    # Reason & Notes
    reason = Column(String)  # user_request, payment_failure, admin_action
    notes = Column(Text)
    
    # Metadata
    metadata = Column(JSON, default=dict)

# Analytics Models

class SubscriptionMetrics(Base):
    """
    Aggregated subscription metrics for analytics
    """
    __tablename__ = 'subscription_metrics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Time Period
    date = Column(DateTime, nullable=False)
    period_type = Column(String, nullable=False)  # daily, weekly, monthly
    
    # Core Metrics
    total_subscriptions = Column(Integer, default=0)
    active_subscriptions = Column(Integer, default=0)
    new_subscriptions = Column(Integer, default=0)
    cancelled_subscriptions = Column(Integer, default=0)
    
    # Revenue Metrics
    total_revenue = Column(Float, default=0.0)
    recurring_revenue = Column(Float, default=0.0)
    average_revenue_per_user = Column(Float, default=0.0)
    
    # Tier Breakdown
    tier_breakdown = Column(JSON, default=dict)
    
    # Churn & Retention
    churn_rate = Column(Float, default=0.0)
    retention_rate = Column(Float, default=0.0)
    
    # Created timestamp
    created_at = Column(DateTime, default=func.now())
    
    # Metadata
    metadata = Column(JSON, default=dict)