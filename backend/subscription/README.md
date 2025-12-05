# DharmaMind Comprehensive Subscription & Payment System

A complete enterprise-grade subscription and payment management system for the DharmaMind spiritual AI platform, featuring multi-tier subscriptions, multiple payment providers, automated billing, usage tracking, and advanced analytics.

## 🌟 Features

### 🎯 Core Subscription Management

- **Multi-Tier System**: Free, Premium, Enterprise, Family, Student, Corporate, Lifetime tiers
- **Flexible Billing**: Monthly, quarterly, annual, and lifetime billing cycles
- **Subscription Lifecycle**: Creation, upgrades, downgrades, pauses, cancellations
- **Trial Management**: Configurable trial periods with automatic conversion
- **Proration**: Automatic proration calculations for mid-cycle changes

### 💳 Payment Processing

- **Multiple Providers**: Stripe, PayPal, Razorpay, Square support
- **Global Currencies**: USD, EUR, GBP, INR, CAD, AUD, SGD, JPY
- **Payment Methods**: Credit/debit cards, digital wallets, UPI, bank transfers
- **Secure Processing**: PCI compliant payment handling
- **Webhook Support**: Real-time payment status updates

### 📊 Usage Tracking & Quotas

- **Feature-Based Limits**: Messages, API calls, storage, team members
- **Real-Time Monitoring**: Live usage tracking and quota enforcement
- **Analytics Dashboard**: Usage patterns and trends analysis
- **Automatic Resets**: Period-based usage limit resets

### 💰 Billing & Invoicing

- **Automated Billing**: Scheduled recurring payments
- **Invoice Generation**: PDF invoices with line items and tax
- **Dunning Management**: Failed payment recovery workflows
- **Tax Calculation**: Configurable tax rates by location
- **Multiple Currencies**: Local pricing and billing

### 📈 Analytics & Reporting

- **Revenue Analytics**: MRR, ARR, revenue trends
- **Churn Analysis**: Customer retention and churn patterns
- **Cohort Analysis**: User behavior over time
- **Business Intelligence**: Comprehensive reporting dashboard
- **Customer Lifetime Value**: CLV calculations and predictions

## 🏗️ Architecture

```
backend/subscription/
├── __init__.py                 # Package initialization and exports
├── models/
│   ├── __init__.py
│   └── subscription_models.py  # SQLAlchemy models
├── services/
│   ├── __init__.py
│   ├── subscription_service.py # Core subscription management
│   ├── payment_service.py      # Payment processing
│   ├── billing_service.py      # Billing and invoicing
│   ├── usage_service.py        # Usage tracking
│   └── analytics_service.py    # Analytics and reporting
├── routes/
│   ├── __init__.py
│   ├── subscription_routes.py  # Subscription API endpoints
│   ├── payment_routes.py       # Payment API endpoints
│   └── billing_routes.py       # Billing API endpoints
└── utils/
    ├── __init__.py
    └── helpers.py              # Utility functions
```

## 📋 Subscription Tiers

### 🌱 Free (Seeker)

- 10 daily messages
- Basic AI model
- 7-day chat history
- Limited scripture access
- 3 meditation sessions

### ⭐ Premium (Devotee) - $19.99/month

- 200 daily messages
- Advanced AI models
- 90-day chat history
- Full scripture access
- 20 meditation sessions
- Personalized guidance
- Voice interaction
- Priority support

### 🏢 Enterprise (Guru) - $99.99/month

- Unlimited messages
- Premium AI models
- Unlimited chat history
- Complete scripture access
- Unlimited meditation sessions
- Team management
- White labeling
- Dedicated support
- Full API access

### 👨‍👩‍👧‍👦 Family (Ashram) - $29.99/month

- 500 daily messages
- 5 family members
- Shared libraries
- Family analytics

### 🎓 Student (Shishya) - $9.99/month

- 50% discount for students
- Educational verification required

## 🚀 Quick Start

### 1. Installation

```python
# Install dependencies
pip install -r requirements.txt

# Add to your Flask app
from backend.subscription import (
    subscription_bp,
    payment_bp,
    billing_bp
)

app.register_blueprint(subscription_bp)
app.register_blueprint(payment_bp)
app.register_blueprint(billing_bp)
```

### 2. Database Setup

```python
from backend.subscription.models import Base
from your_db_config import engine

# Create tables
Base.metadata.create_all(engine)
```

### 3. Configuration

```python
# Environment variables
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
PAYPAL_CLIENT_ID=your_client_id
PAYPAL_CLIENT_SECRET=your_client_secret
RAZORPAY_KEY_ID=rzp_test_...
RAZORPAY_KEY_SECRET=your_key_secret
```

### 4. Basic Usage

```python
from backend.subscription import SubscriptionService

# Initialize service
db_session = get_db_session()
subscription_service = SubscriptionService(db_session)

# Create subscription
subscription = await subscription_service.create_subscription(
    user_id="user123",
    tier=SubscriptionTier.PREMIUM,
    billing_cycle=BillingCycle.MONTHLY,
    trial_days=14
)

# Check feature access
has_access = subscription_service.check_feature_access(
    user_id="user123",
    feature="voice_interaction"
)
```

## 🔗 API Endpoints

### Subscription Management

#### Get Subscription Tiers

```http
GET /api/subscription/tiers
```

#### Create Subscription

```http
POST /api/subscription/create
Content-Type: application/json

{
  "tier": "premium",
  "billing_cycle": "monthly",
  "payment_method_id": "pm_1234",
  "trial_days": 14
}
```

#### Get Current Subscription

```http
GET /api/subscription/current
Authorization: Bearer <token>
```

#### Upgrade Subscription

```http
POST /api/subscription/upgrade
Content-Type: application/json

{
  "new_tier": "enterprise",
  "immediate": true
}
```

#### Cancel Subscription

```http
POST /api/subscription/cancel
Content-Type: application/json

{
  "immediate": false,
  "reason": "user_request"
}
```

### Payment Processing

#### Get Payment Methods

```http
GET /api/payment/methods
Authorization: Bearer <token>
```

#### Process Payment

```http
POST /api/payment/process
Content-Type: application/json

{
  "amount": 19.99,
  "currency": "usd",
  "payment_method": "stripe",
  "payment_method_id": "pm_1234"
}
```

#### Payment History

```http
GET /api/payment/history?limit=50&offset=0
Authorization: Bearer <token>
```

### Billing Management

#### Get Billing Summary

```http
GET /api/billing/summary
Authorization: Bearer <token>
```

#### Invoice History

```http
GET /api/billing/invoices
Authorization: Bearer <token>
```

#### Download Invoice

```http
GET /api/billing/invoices/{invoice_id}/download
Authorization: Bearer <token>
```

### Usage Tracking

#### Usage Summary

```http
GET /api/subscription/usage?period_days=30
Authorization: Bearer <token>
```

#### Check Usage Limits

```http
POST /api/subscription/usage/check
Content-Type: application/json

{
  "feature": "messages",
  "count": 1
}
```

## 🎯 Integration Examples

### Frontend Integration

```javascript
// Check feature access
async function checkFeatureAccess(feature) {
  const response = await fetch(`/api/subscription/feature-access/${feature}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  const result = await response.json();
  return result.has_access;
}

// Track usage
async function trackUsage(feature, count = 1) {
  const response = await fetch("/api/subscription/usage/track", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ feature, count }),
  });

  return response.json();
}
```

### Chat Integration

```python
from backend.subscription import SubscriptionService

async def process_chat_message(user_id: str, message: str):
    # Check message limits
    subscription_service = SubscriptionService(db_session)
    allowed, details = await subscription_service.check_usage_limit(
        user_id=user_id,
        feature="messages",
        requested_count=1
    )

    if not allowed:
        return {"error": "Message limit exceeded", "limit_details": details}

    # Process message
    response = await process_ai_message(message)

    # Track usage
    await subscription_service.track_usage(
        user_id=user_id,
        feature="messages",
        count=1
    )

    return response
```

## 🔒 Security Features

- **PCI Compliance**: Secure payment processing
- **Data Encryption**: Sensitive data protection
- **Webhook Verification**: Signature validation
- **Rate Limiting**: API endpoint protection
- **Audit Logging**: Complete transaction trails

## 🌍 Multi-Currency Support

Supported currencies with automatic exchange rate updates:

- USD (United States Dollar)
- EUR (Euro)
- GBP (British Pound)
- INR (Indian Rupee)
- CAD (Canadian Dollar)
- AUD (Australian Dollar)
- SGD (Singapore Dollar)
- JPY (Japanese Yen)

## 📊 Analytics Dashboard

The system provides comprehensive analytics:

### Business Metrics

- Monthly Recurring Revenue (MRR)
- Annual Recurring Revenue (ARR)
- Customer Lifetime Value (CLV)
- Churn rate and retention analysis
- Revenue by tier and geography

### Usage Analytics

- Feature utilization patterns
- User engagement metrics
- Quota utilization trends
- Peak usage analysis

### Financial Reporting

- Revenue trends and forecasting
- Payment success rates
- Failed payment analysis
- Tax reporting by jurisdiction

## 🛠️ Customization

### Adding New Subscription Tiers

```python
# In subscription_models.py
class SubscriptionTier(str, Enum):
    # Add new tier
    CUSTOM = "custom"

# In helpers.py - update pricing
BASE_PRICING = {
    SubscriptionTier.CUSTOM: {
        BillingCycle.MONTHLY: 49.99,
        BillingCycle.ANNUALLY: 499.99
    }
}
```

### Custom Features

```python
# In helpers.py - update tier features
TIER_FEATURES = {
    SubscriptionTier.CUSTOM: {
        'custom_feature': True,
        'special_limit': 1000
    }
}
```

### Payment Provider Integration

```python
# In payment_service.py
async def _process_new_provider_payment(self, payment, method_id, amount, currency):
    # Implement new payment provider logic
    pass
```

## 🧪 Testing

```bash
# Run subscription tests
pytest backend/subscription/tests/

# Test specific service
pytest backend/subscription/tests/test_subscription_service.py

# Integration tests
pytest backend/subscription/tests/test_integration.py
```

## 📚 Documentation

### Models

- `Subscription`: Core subscription entity
- `Payment`: Payment transaction records
- `Invoice`: Billing and invoice management
- `Usage`: Feature usage tracking
- `PaymentMethodRecord`: Stored payment methods

### Services

- `SubscriptionService`: Core subscription operations
- `PaymentService`: Payment processing
- `BillingService`: Billing and invoicing
- `UsageService`: Usage tracking and limits
- `SubscriptionAnalyticsService`: Analytics and reporting

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:

- 📧 Email: support@dharmamind.com
- 💬 Discord: [DharmaMind Community]
- 📚 Documentation: [docs.dharmamind.com]

---

**Built with ❤️ for the DharmaMind spiritual AI platform**
