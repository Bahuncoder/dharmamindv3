"""
🕉️ DharmaMind Subscription & Payment Models

Complete subscription management system with enterprise-grade security:

Features:
- Multi-tier subscription plans (Free, Pro, Max, Enterprise)
- Secure payment processing with encryption
- Usage tracking and billing management
- Subscription lifecycle management
- Payment method management
- Invoice generation and history
- Security compliance (PCI DSS standards)
- Audit trails for all financial transactions

Security Principles:
- All payment data encrypted at rest and in transit
- PCI DSS compliant data handling
- Zero-knowledge payment processing
- Complete audit trails
- Data minimization principles

May this system serve with transparency and financial dharma 💳
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from decimal import Decimal


# ===============================
# SUBSCRIPTION ENUMERATIONS
# ===============================

class SubscriptionTier(str, Enum):
    """Subscription tier levels"""
    FREE = "free"
    PRO = "pro"
    MAX = "max"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"


class BillingInterval(str, Enum):
    """Billing intervals"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    LIFETIME = "lifetime"


class PaymentStatus(str, Enum):
    """Payment status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class PaymentMethod(str, Enum):
    """Payment methods"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    CRYPTOCURRENCY = "cryptocurrency"


class Currency(str, Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    INR = "INR"


class InvoiceStatus(str, Enum):
    """Invoice status"""
    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    VOID = "void"
    REFUNDED = "refunded"


# ===============================
# SUBSCRIPTION PLAN MODELS
# ===============================

class SubscriptionFeature(BaseModel):
    """Individual subscription feature"""
    feature_id: str = Field(..., description="Unique feature identifier")
    name: str = Field(..., description="Feature display name")
    description: str = Field(..., description="Feature description")
    enabled: bool = Field(default=True, description="Feature enabled status")
    
    # Usage limits
    usage_limit: Optional[int] = Field(default=None, description="Usage limit (-1 for unlimited)")
    reset_period: Optional[str] = Field(default="monthly", description="Usage reset period")
    
    # Feature metadata
    category: str = Field(..., description="Feature category")
    priority: int = Field(default=0, description="Feature priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SubscriptionPlan(BaseModel):
    """Comprehensive subscription plan definition"""
    plan_id: str = Field(..., description="Unique plan identifier")
    tier: SubscriptionTier = Field(..., description="Subscription tier")
    name: str = Field(..., description="Plan display name")
    description: str = Field(..., description="Plan description")
    
    # Pricing
    price: Decimal = Field(..., ge=0, description="Plan price")
    currency: Currency = Field(default=Currency.USD, description="Price currency")
    billing_interval: BillingInterval = Field(..., description="Billing interval")
    
    # Trial settings
    trial_period_days: int = Field(default=0, ge=0, description="Trial period in days")
    trial_price: Decimal = Field(default=Decimal('0'), ge=0, description="Trial price")
    
    # Plan features
    features: List[SubscriptionFeature] = Field(..., description="Plan features")
    
    # Usage limits
    monthly_chat_limit: int = Field(default=-1, description="Monthly chat limit (-1 for unlimited)")
    monthly_api_calls: int = Field(default=-1, description="Monthly API calls limit")
    storage_limit_gb: int = Field(default=-1, description="Storage limit in GB")
    team_members_limit: int = Field(default=1, description="Team members limit")
    
    # Advanced features
    priority_support: bool = Field(default=False, description="Priority support access")
    custom_integrations: bool = Field(default=False, description="Custom integrations access")
    advanced_analytics: bool = Field(default=False, description="Advanced analytics access")
    white_label: bool = Field(default=False, description="White label access")
    api_access: bool = Field(default=False, description="API access")
    
    # Plan management
    active: bool = Field(default=True, description="Plan active status")
    public: bool = Field(default=True, description="Plan public visibility")
    created_at: datetime = Field(default_factory=datetime.now, description="Plan creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Plan last update")
    
    # Dharmic pricing principles
    dharmic_discount: Optional[float] = Field(default=None, ge=0, le=1, description="Dharmic principle discount")
    karma_points_value: int = Field(default=0, ge=0, description="Karma points value for this plan")


# ===============================
# SUBSCRIPTION MODELS
# ===============================

class Subscription(BaseModel):
    """User subscription model"""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Subscription ID")
    user_id: str = Field(..., description="User ID")
    plan_id: str = Field(..., description="Subscription plan ID")
    
    # Status and lifecycle
    status: SubscriptionStatus = Field(..., description="Subscription status")
    current_period_start: datetime = Field(..., description="Current billing period start")
    current_period_end: datetime = Field(..., description="Current billing period end")
    
    # Trial information
    trial_start: Optional[datetime] = Field(default=None, description="Trial start date")
    trial_end: Optional[datetime] = Field(default=None, description="Trial end date")
    
    # Cancellation information
    cancel_at_period_end: bool = Field(default=False, description="Cancel at period end")
    cancelled_at: Optional[datetime] = Field(default=None, description="Cancellation date")
    cancellation_reason: Optional[str] = Field(default=None, description="Cancellation reason")
    
    # Usage tracking
    current_usage: Dict[str, int] = Field(default_factory=dict, description="Current period usage")
    usage_alerts_sent: List[str] = Field(default_factory=list, description="Usage alerts sent")
    
    # Payment information
    payment_method_id: Optional[str] = Field(default=None, description="Payment method ID")
    next_payment_attempt: Optional[datetime] = Field(default=None, description="Next payment attempt")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Subscription metadata")
    notes: Optional[str] = Field(default=None, description="Internal notes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Subscription creation")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update")
    
    # Dharmic principles
    karma_points_earned: int = Field(default=0, ge=0, description="Karma points earned")
    dharmic_actions_count: int = Field(default=0, ge=0, description="Dharmic actions performed")


# ===============================
# PAYMENT MODELS
# ===============================

class PaymentMethodInfo(BaseModel):
    """Secure payment method information"""
    payment_method_id: str = Field(..., description="Payment method ID")
    user_id: str = Field(..., description="User ID")
    method_type: PaymentMethod = Field(..., description="Payment method type")
    
    # Card information (encrypted/tokenized)
    card_last_four: Optional[str] = Field(default=None, description="Last four digits")
    card_brand: Optional[str] = Field(default=None, description="Card brand")
    card_exp_month: Optional[int] = Field(default=None, description="Card expiry month")
    card_exp_year: Optional[int] = Field(default=None, description="Card expiry year")
    
    # Bank information (encrypted/tokenized)
    bank_name: Optional[str] = Field(default=None, description="Bank name")
    account_last_four: Optional[str] = Field(default=None, description="Account last four")
    
    # Digital wallet information
    wallet_email: Optional[str] = Field(default=None, description="Wallet email")
    
    # Status and verification
    verified: bool = Field(default=False, description="Payment method verified")
    default: bool = Field(default=False, description="Default payment method")
    
    # Security
    fingerprint: str = Field(..., description="Payment method fingerprint")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update")
    
    @validator("card_last_four")
    def validate_card_last_four(cls, v):
        if v and len(v) != 4:
            raise ValueError("Card last four must be exactly 4 digits")
        return v


class PaymentIntent(BaseModel):
    """Payment intent for secure processing"""
    payment_intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Payment intent ID")
    user_id: str = Field(..., description="User ID")
    subscription_id: Optional[str] = Field(default=None, description="Related subscription ID")
    
    # Payment details
    amount: Decimal = Field(..., gt=0, description="Payment amount")
    currency: Currency = Field(..., description="Payment currency")
    description: str = Field(..., description="Payment description")
    
    # Status
    status: PaymentStatus = Field(default=PaymentStatus.PENDING, description="Payment status")
    
    # Payment method
    payment_method_id: str = Field(..., description="Payment method ID")
    
    # Processing details
    provider_payment_id: Optional[str] = Field(default=None, description="Provider payment ID")
    provider_name: str = Field(..., description="Payment provider name")
    
    # Security
    client_secret: str = Field(..., description="Client secret for secure processing")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Payment metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    processed_at: Optional[datetime] = Field(default=None, description="Processing time")
    
    # Error information
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class PaymentRecord(BaseModel):
    """Complete payment record"""
    payment_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Payment ID")
    user_id: str = Field(..., description="User ID")
    subscription_id: Optional[str] = Field(default=None, description="Related subscription ID")
    payment_intent_id: str = Field(..., description="Payment intent ID")
    
    # Payment details
    amount: Decimal = Field(..., gt=0, description="Payment amount")
    currency: Currency = Field(..., description="Payment currency")
    description: str = Field(..., description="Payment description")
    
    # Status and processing
    status: PaymentStatus = Field(..., description="Payment status")
    processed_at: datetime = Field(..., description="Processing time")
    
    # Payment method used
    payment_method_type: PaymentMethod = Field(..., description="Payment method type")
    payment_method_details: Dict[str, Any] = Field(default_factory=dict, description="Payment method details")
    
    # Provider information
    provider_name: str = Field(..., description="Payment provider")
    provider_transaction_id: str = Field(..., description="Provider transaction ID")
    provider_fee: Optional[Decimal] = Field(default=None, description="Provider fee")
    
    # Refund information
    refunded_amount: Decimal = Field(default=Decimal('0'), ge=0, description="Refunded amount")
    refund_reason: Optional[str] = Field(default=None, description="Refund reason")
    
    # Security and compliance
    pci_compliant: bool = Field(default=True, description="PCI compliance status")
    fraud_score: Optional[float] = Field(default=None, ge=0, le=1, description="Fraud score")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Payment metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update")


# ===============================
# INVOICE MODELS
# ===============================

class InvoiceLineItem(BaseModel):
    """Invoice line item"""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Line item ID")
    description: str = Field(..., description="Item description")
    quantity: int = Field(..., gt=0, description="Quantity")
    unit_price: Decimal = Field(..., ge=0, description="Unit price")
    total_price: Decimal = Field(..., ge=0, description="Total price")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Item metadata")


class Invoice(BaseModel):
    """Comprehensive invoice model"""
    invoice_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Invoice ID")
    invoice_number: str = Field(..., description="Human-readable invoice number")
    user_id: str = Field(..., description="User ID")
    subscription_id: Optional[str] = Field(default=None, description="Related subscription ID")
    
    # Invoice details
    status: InvoiceStatus = Field(..., description="Invoice status")
    description: str = Field(..., description="Invoice description")
    
    # Amounts
    subtotal: Decimal = Field(..., ge=0, description="Subtotal amount")
    tax_amount: Decimal = Field(default=Decimal('0'), ge=0, description="Tax amount")
    discount_amount: Decimal = Field(default=Decimal('0'), ge=0, description="Discount amount")
    total_amount: Decimal = Field(..., ge=0, description="Total amount")
    currency: Currency = Field(..., description="Invoice currency")
    
    # Line items
    line_items: List[InvoiceLineItem] = Field(..., description="Invoice line items")
    
    # Dates
    invoice_date: datetime = Field(..., description="Invoice date")
    due_date: datetime = Field(..., description="Payment due date")
    paid_date: Optional[datetime] = Field(default=None, description="Payment date")
    
    # Payment information
    payment_id: Optional[str] = Field(default=None, description="Related payment ID")
    payment_method_type: Optional[PaymentMethod] = Field(default=None, description="Payment method used")
    
    # Billing address
    billing_address: Dict[str, str] = Field(default_factory=dict, description="Billing address")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Invoice metadata")
    notes: Optional[str] = Field(default=None, description="Invoice notes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update")
    
    @validator("total_amount")
    def validate_total_amount(cls, v, values):
        """Validate total amount calculation"""
        subtotal = values.get("subtotal", Decimal('0'))
        tax_amount = values.get("tax_amount", Decimal('0'))
        discount_amount = values.get("discount_amount", Decimal('0'))
        
        expected_total = subtotal + tax_amount - discount_amount
        if abs(v - expected_total) > Decimal('0.01'):  # Allow for small rounding differences
            raise ValueError("Total amount must equal subtotal + tax - discount")
        
        return v


# ===============================
# USAGE TRACKING MODELS
# ===============================

class UsageRecord(BaseModel):
    """Usage tracking record"""
    usage_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Usage ID")
    user_id: str = Field(..., description="User ID")
    subscription_id: str = Field(..., description="Subscription ID")
    
    # Usage details
    feature_id: str = Field(..., description="Feature used")
    usage_type: str = Field(..., description="Type of usage")
    quantity: int = Field(..., gt=0, description="Usage quantity")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.now, description="Usage timestamp")
    billing_period: str = Field(..., description="Billing period")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Usage metadata")
    
    # Cost information
    unit_cost: Optional[Decimal] = Field(default=None, description="Unit cost")
    total_cost: Optional[Decimal] = Field(default=None, description="Total cost")


class UsageSummary(BaseModel):
    """Usage summary for billing period"""
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Summary ID")
    user_id: str = Field(..., description="User ID")
    subscription_id: str = Field(..., description="Subscription ID")
    
    # Period information
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    
    # Usage breakdown
    feature_usage: Dict[str, int] = Field(..., description="Usage by feature")
    total_usage: int = Field(..., ge=0, description="Total usage")
    
    # Limits and overages
    usage_limits: Dict[str, int] = Field(..., description="Usage limits")
    overages: Dict[str, int] = Field(default_factory=dict, description="Usage overages")
    
    # Costs
    total_cost: Decimal = Field(default=Decimal('0'), ge=0, description="Total usage cost")
    overage_cost: Decimal = Field(default=Decimal('0'), ge=0, description="Overage cost")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update")


# ===============================
# AUDIT & SECURITY MODELS
# ===============================

class SubscriptionAuditLog(BaseModel):
    """Subscription audit log entry"""
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Audit ID")
    user_id: str = Field(..., description="User ID")
    subscription_id: str = Field(..., description="Subscription ID")
    
    # Event details
    event_type: str = Field(..., description="Event type")
    event_description: str = Field(..., description="Event description")
    
    # Changes
    old_values: Dict[str, Any] = Field(default_factory=dict, description="Old values")
    new_values: Dict[str, Any] = Field(default_factory=dict, description="New values")
    
    # Context
    ip_address: str = Field(..., description="Source IP address")
    user_agent: str = Field(..., description="User agent")
    admin_user_id: Optional[str] = Field(default=None, description="Admin user ID if applicable")
    
    # Security
    security_hash: str = Field(..., description="Security hash for integrity")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")


# ===============================
# REQUEST/RESPONSE MODELS
# ===============================

class SubscriptionCreateRequest(BaseModel):
    """Request to create subscription"""
    plan_id: str = Field(..., description="Plan ID to subscribe to")
    payment_method_id: str = Field(..., description="Payment method ID")
    trial_period: Optional[int] = Field(default=None, description="Custom trial period in days")
    promo_code: Optional[str] = Field(default=None, description="Promotional code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SubscriptionUpdateRequest(BaseModel):
    """Request to update subscription"""
    plan_id: Optional[str] = Field(default=None, description="New plan ID")
    payment_method_id: Optional[str] = Field(default=None, description="New payment method ID")
    cancel_at_period_end: Optional[bool] = Field(default=None, description="Cancel at period end")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata updates")


class PaymentMethodCreateRequest(BaseModel):
    """Request to create payment method"""
    method_type: PaymentMethod = Field(..., description="Payment method type")
    token: str = Field(..., description="Secure token from payment provider")
    set_as_default: bool = Field(default=False, description="Set as default payment method")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ===============================
# RESPONSE MODELS
# ===============================

class SubscriptionResponse(BaseModel):
    """Subscription response"""
    subscription: Subscription = Field(..., description="Subscription details")
    plan: SubscriptionPlan = Field(..., description="Plan details")
    current_usage: Dict[str, Any] = Field(..., description="Current usage information")
    next_invoice: Optional[Dict[str, Any]] = Field(default=None, description="Next invoice preview")


class PaymentResponse(BaseModel):
    """Payment response"""
    payment: PaymentRecord = Field(..., description="Payment record")
    status: str = Field(..., description="Payment status")
    message: str = Field(..., description="Status message")
    next_action: Optional[Dict[str, Any]] = Field(default=None, description="Next action required")


# ===============================
# MODEL EXPORTS
# ===============================

__all__ = [
    # Enums
    "SubscriptionTier",
    "SubscriptionStatus", 
    "BillingInterval",
    "PaymentStatus",
    "PaymentMethod",
    "Currency",
    "InvoiceStatus",
    
    # Plan models
    "SubscriptionFeature",
    "SubscriptionPlan",
    
    # Subscription models
    "Subscription",
    
    # Payment models
    "PaymentMethodInfo",
    "PaymentIntent",
    "PaymentRecord",
    
    # Invoice models
    "InvoiceLineItem",
    "Invoice",
    
    # Usage models
    "UsageRecord",
    "UsageSummary",
    
    # Audit models
    "SubscriptionAuditLog",
    
    # Request models
    "SubscriptionCreateRequest",
    "SubscriptionUpdateRequest",
    "PaymentMethodCreateRequest",
    
    # Response models
    "SubscriptionResponse",
    "PaymentResponse"
]
