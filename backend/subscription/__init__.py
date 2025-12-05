"""
DharmaMind Comprehensive Subscription & Payment System
====================================================

This module provides a complete enterprise-grade subscription and payment management system
for DharmaMind spiritual AI platform, handling multi-tier subscriptions, billing, and payments.

Features:
- Multi-tier subscription management (Free, Premium, Enterprise, Lifetime)
- Payment processing with multiple providers (Stripe, PayPal, Razorpay, Square)
- Automated billing cycles and invoice generation
- Usage tracking and quota management
- Dunning management for failed payments
- Advanced analytics and reporting
- International payment support
- Webhook handling for payment events
- Subscription analytics and insights
- Discount and coupon management
- Corporate billing features
- API monetization tracking
- Compliance and audit trails
"""

from .models.subscription_models import (
    SubscriptionTier,
    SubscriptionStatus,
    PaymentMethod,
    BillingCycle,
    Subscription,
    Payment,
    Invoice,
    Usage,
    Discount
)

from .services.subscription_service import SubscriptionService
from .services.payment_service import PaymentService
from .services.billing_service import BillingService
from .services.usage_service import UsageService
from .services.analytics_service import SubscriptionAnalyticsService

from .routes.subscription_routes import subscription_bp
from .routes.payment_routes import payment_bp
from .routes.billing_routes import billing_bp

__all__ = [
    # Models
    'SubscriptionTier',
    'SubscriptionStatus', 
    'PaymentMethod',
    'BillingCycle',
    'Subscription',
    'Payment',
    'Invoice',
    'Usage',
    'Discount',
    
    # Services
    'SubscriptionService',
    'PaymentService', 
    'BillingService',
    'UsageService',
    'SubscriptionAnalyticsService',
    
    # Routes
    'subscription_bp',
    'payment_bp',
    'billing_bp'
]

# Version information
__version__ = "1.0.0"
__author__ = "DharmaMind Team"
__description__ = "Comprehensive Subscription & Payment System for DharmaMind"